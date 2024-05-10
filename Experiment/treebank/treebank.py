from collections import Counter
from collections.abc import Iterable
from typing import overload, Literal
from random import Random
from os import walk, path
import json
from .tree import Tree
from .brat2conllu import generate_conllu_from_brat

class TreeBank:

    def __init__(self, trees: Iterable[Tree]):
        self.__trees = list(trees)
        identifiers = Counter((tree.filename, tree.sent_id) for tree in self)
        non_unique = [key for key, count in identifiers.items() if count > 1]
        assert len(non_unique) == 0, f"Duplicate {Tree.__name__} identifiers (filename, sent_id) in {self}\n" + '\n'.join(f"({filename}, {sent_id})" for filename, sent_id in non_unique)
        self.num_tokens = sum(len(tree) for tree in self)
        self.num_non_projective_trees = sum(not tree.is_projective for tree in self)
        self.num_non_projective_arcs = sum(tree.num_non_projective_arcs for tree in self)
        tag_set: list[str] = []
        upos_set: list[str] = []
        for tree in self:
            for token in tree:
                if token.deprel not in tag_set:
                    tag_set.append(token.deprel)
                if token.upos not in upos_set:
                    upos_set.append(token.upos)
        self.tag_set = tag_set
        self.upos_set = upos_set

    def copy(self):
        return type(self)(tree.copy() for tree in self)

    def __len__(self):
        return len(self.__trees)

    def __iter__(self):
        return iter(self.__trees)

    def __reversed__(self):
        return reversed(self.__trees)

    @overload
    def __getitem__(self, index: int) -> Tree: ...
    @overload
    def __getitem__(self, index: slice) -> list[Tree]: ...
    def __getitem__(self, index: int | slice):
        return self.__trees[index]

    def __repr__(self):
        return f"<{type(self).__name__} containing {len(self)} tree(s)>"

    @classmethod
    def from_conllu_file(cls, conllu_file_path: str):
        trees: list[Tree] = []
        with open(conllu_file_path, encoding="utf-8") as conllu_file:
            for raw_conllu in conllu_file.read().split("\n\n"):
                if raw_conllu != '':
                    tree = Tree(raw_conllu)
                    if tree.filename is None:
                        tree.filename = conllu_file_path
                    trees.append(tree)
        return cls(trees)

    @classmethod
    def from_conllu_dir(cls, conllu_dir_path: str):
        trees: list[Tree] = []
        for dirpath, _, filenames in walk(conllu_dir_path):
            for filename in filenames:
                if filename.endswith(".conllu"):
                    file_path = path.join(dirpath, filename)
                    with open(file_path, encoding="utf-8") as conllu_file:
                        for raw_conllu in conllu_file.read().split("\n\n"):
                            if raw_conllu != '':
                                try:
                                    tree = Tree(raw_conllu)
                                    if tree.filename is None:
                                        tree.filename = filename
                                    trees.append(tree)
                                except AssertionError as e:
                                    print(f"The following error occured in '{file_path}':")
                                    print(e)
        return cls(trees)

    @classmethod
    def from_brat_dir(cls, brat_dir_path: str):
        return cls(Tree(raw_conllu) for raw_conllu in generate_conllu_from_brat(brat_dir_path))

    def to_conllu(self):
        return ''.join(tree.to_conllu() + "\n\n" for tree in self)

    def to_list_of_dict(self):
        return [tree.to_dict() for tree in self]

    def to_json(self):
        return json.dumps(self.to_list_of_dict(), indent='\t')

    def save(self, format: Literal["conllu", "json"], file_path: str):
        if format not in ("conllu", "json"):
            raise ValueError(f"Unknown format: {format!r}")
        if not file_path.endswith(f".{format}"):
            file_path += f".{format}"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.to_conllu() if format == "conllu" else self.to_json())

    def train_dev_test_split(
        self,
        train_ratio: int,
        dev_ratio: int,
        test_ratio: int,
        *,
        seed: int | None = None
    ):
        sum_ratio = train_ratio + dev_ratio + test_ratio
        train_size = max(int(len(self) * (train_ratio / sum_ratio)), 1)
        dev_size = max(int(len(self) * (dev_ratio / sum_ratio)), 1)
        temp_trees = self.__trees.copy()
        Random(seed).shuffle(temp_trees)
        return (
            type(self)(temp_trees[:train_size]),
            type(self)(temp_trees[train_size:train_size+dev_size]),
            type(self)(temp_trees[train_size+dev_size:])
        )
