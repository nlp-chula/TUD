from torch.optim import AdamW
from transformers import get_scheduler
from os import path
from tqdm.auto import tqdm
import re, torch
from treebank import TreeBank
from .transition_based import TransitionBasedModel
from .graph_based import GraphBasedModel

def get_train_dev_test(data_path: str):
    return (
        TreeBank.from_conllu_file(path.join(data_path, "train.conllu")),
        TreeBank.from_conllu_file(path.join(data_path, "dev.conllu")),
        TreeBank.from_conllu_file(path.join(data_path, "test.conllu"))
    )

def train_model(
    *,
    model: TransitionBasedModel | GraphBasedModel,
    train_set: TreeBank,
    dev_set: TreeBank,
    test_set: TreeBank,
    num_epochs: int,
    batch_size: int,
    save_path: str
):
    is_graph_based = isinstance(model, GraphBasedModel)

    optimizer = AdamW(
        params=model.parameters(),
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

    max_las = 0.0
    for i in range(1, num_epochs + 1):
        model.train()
        print(f"EPOCH: {i}")
        start, stop = 0, batch_size
        batch = [tree for tree in train_set[start:stop] if is_graph_based or tree.is_projective]
        progress_bar = tqdm(total=num_batches_per_epoch)
        while batch:
            loss = model(batch).loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            start, stop = stop, stop + batch_size
            batch = [tree for tree in train_set[start:stop] if is_graph_based or tree.is_projective]
            progress_bar.update()
        model.eval()
        dev_metrics = model.evaluate(dev_set)
        print(f"DEV: {dev_metrics}")
        if dev_metrics["LAS"] > max_las:
            max_las = dev_metrics["LAS"]
            torch.save(model.state_dict(), save_path)
            print("Saved!")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_metrics = model.evaluate(test_set)
    print(f"TEST: {test_metrics}")
    return test_metrics

MODEL_FILENAME_PATTERN = re.compile(r"(?P<dataset>[^\-]+)\-(?P<architecture>transition|graph)(\-(?P<action_set>standard|eager))?\-(?P<transformer>wangchan|phayathai)(?P<augmented>\-augmented)?\-(?P<pos>gold|auto|agnostic)_pos\.pt")
TRANSFORMER_NAME_TO_PATH = {
    "wangchan": "airesearch/wangchanberta-base-att-spm-uncased",
    "phayathai": "clicknext/phayathaibert"
}

def set_transformer_path(name: str, path: str):
    TRANSFORMER_NAME_TO_PATH[name] = path

def load_model(model_path: str, space_token: str):
    model_dirname, model_filename = path.split(model_path)
    match = MODEL_FILENAME_PATTERN.fullmatch(model_filename)
    if not match:
        raise ValueError(f"Invalid model file name: {model_filename!r}")
    tag_set = torch.load(path.join(model_dirname, f"{match['dataset']}_tag_set.pt"))
    transformer_path = TRANSFORMER_NAME_TO_PATH.get(match["transformer"])
    if transformer_path is None:
        raise ValueError(f"Unknown transformer name: {match['transformer']!r}\nYou might have forgotten to call {set_transformer_path.__name__}()")
    if match["architecture"] == "transition":
        if match["action_set"] is None:
            raise ValueError(f"Missing transition action set in file name: {model_filename!r}")
        if match["action_set"] not in ("standard", "eager"):
            raise ValueError(f"Invalid transition action set: {match['action_set']!r}")
        model = TransitionBasedModel(
            action_set=match["action_set"],
            tag_set=tag_set,
            upos_set=None if match["pos"] == "agnostic" else torch.load(path.join(model_dirname, f"{match['dataset']}_upos_set.pt")),
            transformer_path=transformer_path,
            space_token=space_token,
            augment=match["augmented"] is not None
        )
    elif match["architecture"] == "graph":
        model = GraphBasedModel(
            tag_set=tag_set,
            upos_set=None if match["pos"] == "agnostic" else torch.load(path.join(model_dirname, f"{match['dataset']}_upos_set.pt")),
            transformer_path=transformer_path,
            space_token=space_token,
            augment=match["augmented"] is not None
        )
    else:
        raise ValueError(f"Invalid architecture: {match['architecture']!r}")

    print(model.load_state_dict(torch.load(model_path), strict=False))
    return model.eval()

def parse_treebank(treebank: TreeBank, model: TransitionBasedModel | GraphBasedModel):
    new_treebank = treebank.copy()
    # Ensure that old parses are removed
    for tree in new_treebank:
        for token in tree:
            token.head = -1
            token.deprel = '_'
    # Parse each tree
    for tree in new_treebank:
        result = model.parse(tree)
        for relation in result:
            relation.dep.head = relation.head.id
            relation.dep.deprel = relation.deprel
    return new_treebank
