from os import walk, path
from dataclasses import dataclass
from typing import Any
from logging import warning

@dataclass
class BratToken:
    token_id: str
    word: str
    pos: str
    special_attr: str
    head_position: str
    relation_type: str
    space_after: str

    def to_conllu(self):
        return f"{self.token_id}\t{self.word}\t_\t{self.pos}\t{self.pos}\t{self.special_attr}\t{self.head_position}\t{self.relation_type}\t_\tSpaceAfter={self.space_after}"

class BratTree:

    def __init__(self, filename: str, sent_id: int, text: str, token_dict: dict[str, dict[str, str]]):
        self.filename = filename
        self.sent_id = sent_id
        self.text = text

        to_token_id = {
            ann_id: str(token_id)
                for token_id, ann_id in enumerate(
                    sorted(
                        token_dict,
                        key=lambda x: token_dict[x]["position"]
                    ),
                    start=1
                )
        }

        self.tokens = sorted(
            (
                BratToken(
                    to_token_id[ann_id],
                    attrs.get("word", "_"),
                    attrs.get("pos", "_"),
                    attrs.get("special_attr", "_"),
                    ','.join(to_token_id[i] for i in attrs["head_id"]) if "head_id" in attrs else "_",
                    ','.join(r for r in attrs["relation_type"]) if "relation_type" in attrs else "_",
                    attrs.get("space_after", "_")
                )
                for ann_id, attrs in token_dict.items()
            ),
            key=lambda token: int(token.token_id)
        )

        not_have_head = [token for token in self.tokens if token.head_position == "_"]
        if len(not_have_head) == 1:
            root_token = not_have_head[0]
            root_token.head_position = "0"
            root_token.relation_type = "root"

    def to_conllu(self):
        first_line = f"# filename = {self.filename}"
        second_line = f"# sent_id = {self.sent_id}"
        third_line = f"# text = {self.text}"
        body = '\n'.join(token.to_conllu() for token in self.tokens)
        return f"{first_line}\n{second_line}\n{third_line}\n{body}"


def convert_ann_txt_pair_to_conllu(annfilepath: str, txtfilepath: str, sent_id: int):

    with open(txtfilepath, encoding="utf-8") as txtfile:
        text = txtfile.read().replace('\n', ' ').strip()
        text_len = len(text)

    with open(annfilepath, encoding="utf-8") as annfile:
        token_dict: dict[str, dict[str, Any]] = {}
        for annotation in annfile:
            annotation = annotation.strip()
            fields = annotation.split('\t')
            annotation_type = fields[0][0]
            if annotation_type == 'T':
                ann_id, (pos, start, *_, end), word = fields[0], fields[1].split(' '), fields[2]
                start, end = int(start), int(end)
                space_after = "Yes" if end < text_len and text[end] == ' ' else "No"
                if ann_id in token_dict:
                    token_dict[ann_id]["word"] = word
                    token_dict[ann_id]["pos"] = pos
                    token_dict[ann_id]["space_after"] = space_after
                    token_dict[ann_id]["position"] = (start, -end)
                else:
                    token_dict[ann_id] = {
                        "word": word,
                        "pos": pos,
                        "space_after": space_after,
                        "position": (start, -end)
                    }
                # NOTE: We sort 'end' descendingly so that overlapped annotations appear separate in the conllu file
                # EXAMPLE: When "ประเทศไทย", "ประเทศ", and "ไทย" are annotated simultaneously,
                # we want them to appear in that order and not "ประเทศ", "ประเทศไทย", "ไทย"
            elif annotation_type == 'A':
                attr_name, ann_id, *value = fields[1].split(' ')
                if '-' in attr_name:
                    special_attr = attr_name.replace('-', '=')
                elif len(value) == 0:
                    special_attr = f"{attr_name}=Yes"
                elif len(value) == 1:
                    special_attr = f"{attr_name}={value[0]}"
                else:
                    special_attr = "_"
                if ann_id in token_dict:
                    if "special_attr" in token_dict[ann_id]:
                        token_dict[ann_id]["special_attr"] += f"|{special_attr}"
                    else:
                        token_dict[ann_id]["special_attr"] = special_attr
                else:
                    token_dict[ann_id] = {
                        "special_attr": special_attr
                    }
            elif annotation_type == 'R':
                relation_type, arg1, arg2, *_ = fields[1].split(' ')
                head_id = arg1.split(':')[1]
                ann_id = arg2.split(':')[1]
                if ann_id in token_dict:
                    if "head_id" in token_dict[ann_id]:
                        token_dict[ann_id]["head_id"].append(head_id)
                        token_dict[ann_id]["relation_type"].append(relation_type)
                    else:
                        token_dict[ann_id]["head_id"] = [head_id]
                        token_dict[ann_id]["relation_type"] = [relation_type]
                else:
                    token_dict[ann_id] = {
                        "head_id": [head_id],
                        "relation_type": [relation_type]
                    }
    return BratTree(path.basename(annfilepath), sent_id, text, token_dict).to_conllu()

def find_ann_txt_pairs(root_directory: str):
    for dir, _, filenames in walk(root_directory):
        for filename in filenames:
            if filename.endswith(".ann"):
                txtfilename = f"{filename[:-4]}.txt"
                if txtfilename in filenames:
                    yield path.join(dir, filename), path.join(dir, txtfilename)
                else:
                    warning(f"'{filename}' doesn't have a corresponding '{txtfilename}' in {dir}")

def generate_conllu_from_brat(brat_dir_path: str):
    for sent_id, (annfilepath, txtfilepath) in enumerate(find_ann_txt_pairs(brat_dir_path), start=1):
        yield convert_ann_txt_pair_to_conllu(annfilepath, txtfilepath, sent_id)
