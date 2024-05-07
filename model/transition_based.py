from torch.nn import Module, Linear, Dropout, CrossEntropyLoss, Parameter, Embedding
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
import torch
from treebank.tree import Tree, Relation
from treebank.token import Token
from treebank import TreeBank
from .super_token import SuperTokenEmbedding

@dataclass
class TransitionBasedModelOutput:
    output: Tensor
    loss: Tensor

class TransitionBasedModel(Module):

    def __init__(
        self,
        *,
        action_set: str,
        tag_set: list[str],
        upos_set: list[str] | None,
        transformer_path: str,
        space_token: str,
        augment: bool
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
        self.transformer = AutoModel.from_pretrained(transformer_path, add_pooling_layer=False)

        if action_set not in ("standard", "eager"):
            raise ValueError(f"Invalid action set: {action_set}")
        self.action_set = action_set
        tag_set = ["Shift"] + [f"LeftArc-{tag}" for tag in tag_set] + [f"RightArc-{tag}" for tag in tag_set]
        if action_set == "eager":
            tag_set.append("Reduce")
        self.id_to_label = dict(enumerate(tag_set))
        self.label_to_id = {label: i for i, label in enumerate(tag_set)}
        self.space_token = space_token
        vocab = self.tokenizer.get_vocab()
        self.space_ids = {vocab.get(space_token), vocab.get('▁')}
        self.space_ids.discard(None)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        config = self.transformer.config
        hidden_size = config.hidden_size
        initializer_range = config.initializer_range
        # ROOT and END embeddings
        self.root_embedding = Parameter(torch.zeros(1, hidden_size))
        self.end_embedding = Parameter(torch.zeros(1, hidden_size))
        self.root_embedding.data.normal_(mean=0.0, std=initializer_range)
        self.end_embedding.data.normal_(mean=0.0, std=initializer_range)
        # POS embeddings
        if upos_set is not None:
            upos_set = ["<ROOT>", "<END>"] + upos_set
            self.id_to_pos = dict(enumerate(upos_set))
            self.pos_to_id = {pos: i for i, pos in enumerate(upos_set)}
            self.pos_embeddings = Embedding(len(upos_set), hidden_size)
            self.use_pos = True
            additional_dim = hidden_size
        else:
            self.use_pos = False
            additional_dim = 0
        # Feature augmentation
        if augment:
            kernel_sizes = [2, 3, 4, 5]
            each_kernel_size_output_dim = hidden_size // len(kernel_sizes)
            self.super_token_embeddings = SuperTokenEmbedding(
                embed_dim=hidden_size + additional_dim,
                kernel_sizes=kernel_sizes,
                each_kernel_size_output_dim=each_kernel_size_output_dim
            )
            additional_dim += (len(kernel_sizes) * each_kernel_size_output_dim) + hidden_size
        self.augmented = augment
        # Classifier
        feature_size = (hidden_size + additional_dim) * 3
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dense = Linear(feature_size, hidden_size)
        self.dropout = Dropout(classifier_dropout)
        self.out_proj = Linear(hidden_size, len(self.id_to_label))
        # Loss
        self.loss_func = CrossEntropyLoss()

        self.to(self.device)

    def __encode_and_pool(self, trees: list[Tree]):
        # Create list of lists of strings to be tokenized
        words: list[list[str]] = [[] for _ in trees]
        for tree, word in zip(trees, words):
            for token in tree:
                word.append(token.form)
                if self.space_token != ' ' and token.miscs["SpaceAfter"] == "Yes":
                    word.append(self.space_token)
        # Tokenize
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        # Encode using transformer
        encoded = self.transformer(**tokenized).last_hidden_state
        # Select the encoding of the first non-space token of each word
        select_indice: list[list[int]] = [[] for _ in trees]
        for i, (select_index, input_ids) in enumerate(zip(select_indice, tokenized.input_ids)):
            word_ids = tokenized.word_ids(batch_index=i)
            last_word_id = None
            for j, (word_id, token_id) in enumerate(zip(word_ids, input_ids)):
                if word_id in (None, last_word_id) or token_id.item() in self.space_ids:
                    continue
                select_index.append(j)
                last_word_id = word_id
        pooled_encodings = [
            torch.cat([
                self.root_embedding,
                encoded[i, select_index],
                self.end_embedding
            ])
            for i, select_index in enumerate(select_indice)
        ]
        # POS embeddings
        if self.use_pos:
            upos_tags = [
                torch.tensor(
                    [self.pos_to_id["<ROOT>"]] +
                    [self.pos_to_id[token.upos] for token in tree] +
                    [self.pos_to_id["<END>"]]
                ).to(self.device)
                for tree in trees
            ]
            pos_embeddings = [self.pos_embeddings(tags) for tags in upos_tags]
            pooled_encodings = [
                torch.cat([
                    pooled_encoding, # Token embedding
                    pos_embedding # POS embedding
                ], dim=1)
                for pooled_encoding, pos_embedding in zip(pooled_encodings, pos_embeddings)
            ]
        # Feature augmentation
        if self.augmented:
            return [
                torch.cat([
                    pooled_encoding, # Token embedding
                    super_token_embedding, # Super token embedding
                    encoded[i, 0].expand(len(pooled_encoding), -1) # Sentence embedding
                ], dim=1)
                for i, (pooled_encoding, super_token_embedding) in enumerate(
                    zip(pooled_encodings, self.super_token_embeddings(pooled_encodings))
                )
            ]
        else:
            return pooled_encodings

    def __classifier(self, x: Tensor):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

    def forward(self, trees: list[Tree]):
        # Encode and pool
        pooled_encodings = self.__encode_and_pool(trees)
        # Create training samples
        inputs: list[Tensor] = []
        labels: list[int] = []
        if self.action_set == "standard":
            for tree, encoding in zip(trees, pooled_encodings):
                for state, action in tree.to_transitions("standard"):
                    if len(state.stack) < 2:
                        continue
                    inputs.append(torch.cat([
                        encoding[state.stack[-2].id],
                        encoding[state.stack[-1].id],
                        encoding[state.buffer[0].id]
                    ]))
                    labels.append(self.label_to_id[action])
        elif self.action_set == "eager":
            for tree, encoding in zip(trees, pooled_encodings):
                for state, action in tree.to_transitions("eager"):
                    if len(state.buffer) < 2:
                        continue
                    inputs.append(torch.cat([
                        encoding[state.stack[-1].id],
                        encoding[state.buffer[0].id],
                        encoding[state.buffer[1].id]
                    ]))
                    labels.append(self.label_to_id[action])
        x = torch.stack(inputs)
        y = torch.tensor(labels).to(self.device)
        # Classifier
        x = self.__classifier(x)
        # Loss
        loss = self.loss_func(x, y)
        # Return
        return TransitionBasedModelOutput(
            output=x,
            loss=loss
        )

    def evaluate(self, test_treebank: TreeBank):
        num_head_correct = 0
        num_head_label_correct = 0
        num_total = 0
        for tree in test_treebank:
            num_total += len(tree)
            result = self.parse(tree)
            for relation in result:
                if relation.dep.head_token is relation.head:
                    num_head_correct += 1
                    if relation.dep.deprel == relation.deprel:
                        num_head_label_correct += 1
        return {"UAS": num_head_correct / num_total, "LAS": num_head_label_correct / num_total}

    @torch.no_grad()
    def parse(self, tree: Tree, trace: bool = False):
        encoding = self.__encode_and_pool([tree])[0]
        stack = [Tree._ROOT]
        buffer = list(tree) + [Tree._END]
        relations: list[Relation] = []
        if trace: show_state(stack, buffer)
        if self.action_set == "standard":
            while len(stack) > 1 or len(buffer) > 1:
                if len(stack) < 2:
                    stack.append(buffer.pop(0))
                    if trace: print("Shift")
                else:
                    top = stack[-1]
                    second = stack[-2]
                    x = torch.stack([torch.cat([
                        encoding[second.id],
                        encoding[top.id],
                        encoding[buffer[0].id]
                    ])])
                    pred = self.__classifier(x)[0]
                    action_ranks = map(
                        self.id_to_label.__getitem__,
                        sorted(
                            range(len(self.id_to_label)),
                            key=lambda i: pred[i],
                            reverse=True
                        )
                    )
                    for action in action_ranks:
                        if action == "Shift":
                            if len(buffer) > 1:
                                stack.append(buffer.pop(0))
                                if trace: print(action)
                                break
                            else:
                                continue
                        arc, rel = action.split("-")
                        if arc == "LeftArc":
                            if second is not Tree._ROOT:
                                relations.append(
                                    Relation(
                                        head=top,
                                        dep=second,
                                        deprel=rel
                                    )
                                )
                                stack.pop(-2)
                                if trace: print(action)
                            else:
                                continue
                        elif arc == "RightArc":
                            relations.append(
                                Relation(
                                    head=second,
                                    dep=top,
                                    deprel=rel
                                )
                            )
                            stack.pop()
                            if trace: print(action)
                        break
                if trace:
                    show_state(stack, buffer)
        elif self.action_set == "eager":
            while len(stack) > 1 or len(buffer) > 1:
                if len(buffer) == 1:
                    stack.pop()
                    if trace: print("Reduce")
                else:
                    top = stack[-1]
                    front = buffer[0]
                    x = torch.stack([torch.cat([
                        encoding[top.id],
                        encoding[front.id],
                        encoding[buffer[1].id]
                    ])])
                    pred = self.__classifier(x)[0]
                    action_ranks = map(
                        self.id_to_label.__getitem__,
                        sorted(
                            range(len(self.id_to_label)),
                            key=lambda i: pred[i],
                            reverse=True
                        )
                    )
                    for action in action_ranks:
                        if action == "Reduce":
                            if (
                                len(stack) > 1 and
                                any(relation.dep is top for relation in relations)
                            ):
                                stack.pop()
                                if trace: print(action)
                                break
                            else:
                                continue
                        if action == "Shift":
                            stack.append(buffer.pop(0))
                            if trace: print(action)
                            break
                        arc, rel = action.split("-")
                        if arc == "LeftArc":
                            if (
                                top is not Tree._ROOT and
                                not any(relation.dep is top for relation in relations)
                            ):
                                relations.append(
                                    Relation(
                                        head=front,
                                        dep=top,
                                        deprel=rel
                                    )
                                )
                                stack.pop()
                                if trace: print(action)
                            else:
                                continue
                        elif arc == "RightArc":
                            relations.append(
                                Relation(
                                    head=top,
                                    dep=front,
                                    deprel=rel
                                )
                            )
                            stack.append(buffer.pop(0))
                            if trace: print(action)
                        break
                if trace:
                    show_state(stack, buffer)
        return relations

def show_state(stack: list[Token], buffer: list[Token]):
    print(f"[{', '.join(token.form for token in stack)}], [{', '.join(token.form for token in buffer)}]")
