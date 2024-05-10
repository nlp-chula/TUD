from torch.nn import Module, CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_scheduler
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataclasses import dataclass
from typing import Any
from os import path
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import torch, re
from treebank.tree import Tree
from treebank import TreeBank

@dataclass
class POSTaggerOutput:
    logits: list[Tensor]
    loss: Tensor

class POSTagger(Module):

    def __init__(
        self,
        *,
        upos_set: list[str],
        transformer_path: str,
        space_token: str
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
        self.classifier = AutoModelForTokenClassification.from_pretrained(transformer_path, num_labels=len(upos_set))
        self.loss_func = CrossEntropyLoss()

        self.id_to_label = dict(enumerate(upos_set))
        self.label_to_id = {label: i for i, label in enumerate(upos_set)}
        self.space_token = space_token
        vocab = self.tokenizer.get_vocab()
        self.space_ids = {vocab.get(space_token), vocab.get('▁')}
        self.space_ids.discard(None)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.to(self.device)

    def __tokenize_and_classify(self, trees: list[Tree]):
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
        # Classify
        logits = self.classifier(**tokenized).logits
        # Select the classification of the first non-space token of each word
        select_indice: list[list[int]] = [[] for _ in trees]
        for i, (select_index, input_ids) in enumerate(zip(select_indice, tokenized.input_ids)):
            word_ids = tokenized.word_ids(batch_index=i)
            last_word_id = None
            for j, (word_id, token_id) in enumerate(zip(word_ids, input_ids)):
                if word_id in (None, last_word_id) or token_id.item() in self.space_ids:
                    continue
                select_index.append(j)
                last_word_id = word_id
        return [
            logits[i, select_index]
            for i, select_index in enumerate(select_indice)
        ]

    def forward(self, trees: list[Tree]):
        logits = self.__tokenize_and_classify(trees)
        labels = torch.tensor([
            self.label_to_id[token.upos]
            for tree in trees
                for token in tree
        ]).to(self.device)
        loss = self.loss_func(torch.cat(logits), labels)
        return POSTaggerOutput(
            logits=logits,
            loss=loss
        )

    def evaluate(self, test_treebank: TreeBank) -> dict[str, Any]:
        y_true = [token.upos for tree in test_treebank for token in tree]
        y_pred = [label for tree in test_treebank for label in self.tag(tree)]
        return classification_report(y_true, y_pred, output_dict=True) # type: ignore

    @torch.no_grad()
    def tag(self, tree: Tree):
        logits = self.__tokenize_and_classify([tree])[0].argmax(dim=1)
        return [
            self.id_to_label[logit.item()]
            for logit in logits
        ]

def train_pos_tagger(
    *,
    pos_tagger: POSTagger,
    train_set: TreeBank,
    dev_set: TreeBank,
    test_set: TreeBank,
    num_epochs: int,
    batch_size: int,
    save_path: str
):
    optimizer = AdamW(
        params=pos_tagger.parameters(),
        lr=3e-5,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    num_batches_per_epoch, remainder = divmod(len(train_set), batch_size)
    if remainder:
        num_batches_per_epoch += 1
    num_total_steps = num_epochs * num_batches_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_total_steps),
        num_training_steps=num_total_steps
    )

    max_score = 0.0
    for i in range(1, num_epochs + 1):
        pos_tagger.train()
        print(f"EPOCH: {i}")
        start, stop = 0, batch_size
        batch = [tree for tree in train_set[start:stop]]
        progress_bar = tqdm(total=num_batches_per_epoch)
        while batch:
            loss = pos_tagger(batch).loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            start, stop = stop, stop + batch_size
            batch = [tree for tree in train_set[start:stop]]
            progress_bar.update()
        pos_tagger.eval()
        dev_metrics = pos_tagger.evaluate(dev_set)
        print(f"DEV: {dev_metrics['macro avg']['f1-score']}")
        if dev_metrics["macro avg"]["f1-score"] > max_score:
            max_score = dev_metrics["macro avg"]["f1-score"]
            torch.save(pos_tagger.state_dict(), save_path)
            print("Saved!")
    pos_tagger.load_state_dict(torch.load(save_path))
    pos_tagger.eval()
    test_metrics = pos_tagger.evaluate(test_set)
    print(f"TEST: {test_metrics['macro avg']['f1-score']}")
    return test_metrics

POS_TAGGER_FILENAME_PATTERN = re.compile(r"(?P<dataset>.+)_pos_tagger\.pt")
POS_TAGGER_TRANSFORMER_PATH = "clicknext/phayathaibert"
def load_pos_tagger(pos_tagger_path: str, space_token: str):
    pos_tagger_dirname, pos_tagger_filename = path.split(pos_tagger_path)
    match = POS_TAGGER_FILENAME_PATTERN.fullmatch(pos_tagger_filename)
    if not match:
        raise ValueError(f"Invalid model file name: {pos_tagger_filename!r}")
    upos_set = torch.load(path.join(pos_tagger_dirname, f"{match['dataset']}_upos_set.pt"))
    pos_tagger = POSTagger(
        upos_set=upos_set,
        transformer_path=POS_TAGGER_TRANSFORMER_PATH,
        space_token=space_token
    )
    print(pos_tagger.load_state_dict(torch.load(pos_tagger_path), strict=False))
    return pos_tagger.eval()

def tag_treebank(treebank: TreeBank, pos_tagger: POSTagger):
    new_treebank = treebank.copy()
    # Ensure that old tags are removed
    for tree in new_treebank:
        for token in tree:
            token.upos = '_'
    # Tag each tree
    for tree in new_treebank:
        tags = pos_tagger.tag(tree)
        for token, tag in zip(tree, tags):
            token.upos = tag
    return new_treebank
