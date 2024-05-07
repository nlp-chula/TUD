from typing import TypedDict, Required, NotRequired, Self
from .token import Token
from .tree import Tree

class SplitToken(TypedDict):
    token: Required[str]
    pos: Required[str]
    special_attr: Required[str]
    head_index: Required[int]
    relation_type: Required[str]
    space_after: Required[bool]
    token_id: NotRequired[int]
    head: NotRequired[Self]

def split_conllu_into_sentences(*, input_path: str, output_path: str):
    with open(input_path, encoding="utf-8") as input, open(output_path, "w", encoding="utf-8") as output:
        raw_trees = [raw_tree for raw_tree in input.read().split("\n\n") if raw_tree != '']
        for raw_tree in raw_trees:
            try:
                raw_lines = raw_tree.split('\n')
                headers: dict[str, str] = {}
                for i, raw_line in enumerate(raw_lines):
                    match = Tree.header_pattern.fullmatch(raw_line)
                    if match:
                        headers[match["key"]] = match["value"]
                    else:
                        raw_lines = raw_lines[i:]
                        break
                filename = headers.get("filename")
                sent_id = headers["sent_id"]
                _ = headers["text"]

                ## Process body ##
                assert all(Token.pattern.fullmatch(line) for line in raw_lines), f"Wrong body format"
                body = [line.split('\t') for line in raw_lines]
                tokens: list[SplitToken] = [
                    {
                        "token": token,
                        "pos": pos,
                        "special_attr": special_attr,
                        "head_index": int(head_id) - 1 if head_id not in ('_', '0') else -1, # int(head_id) - 1 because Python's indices start at 0 instead of 1
                        "relation_type": relation_type,
                        "space_after": space_after == "SpaceAfter=Yes"
                    }
                    for _, token, _, pos, pos, special_attr, head_id, relation_type, _, space_after in body
                ]
                # head_index is None iff relation_type == root
                assert all((token["relation_type"] == "root") == (token["head_index"] == -1) for token in tokens), f"Incomplete or incorrect dependency in {filename}"

                ## Split into sentences ##
                # The assumptions are:
                # 1. Sentences cannot overlap
                # 2. Every token in a sentence can be traced back to a common root token
                sentences: list[list[SplitToken]] = []
                current_position_in_sentence = 1
                current_root = None
                for old_token_id, token in enumerate(tokens, start=1):
                    # Trace back to root token
                    root = token
                    loop_count = 0
                    while root["relation_type"] != "root":
                        root = tokens[root["head_index"]]
                        loop_count += 1
                        assert loop_count < len(tokens), f"Maximum loop count reached at token {old_token_id} ({token['token']}) in {filename}. There might be a cycle creating infinite loop."
                    if root is not current_root:
                        # New sentence reached
                        current_root = root
                        current_position_in_sentence = 1
                        sentences.append([])
                    else:
                        # Still the same sentence
                        current_position_in_sentence += 1
                    # Assign new ID
                    token["token_id"] = current_position_in_sentence
                    # Since we might not yet know what ID the head token will have in the new sentence,
                    # We store a reference to the head Token dict so that we can reach it to find out later once every token is assigned new ID
                    token["head"] = tokens[token["head_index"]] if token["relation_type"] != "root" else token
                    # Append token to the last sentence
                    sentences[-1].append(token)

                ## Last token should not be marked SpaceAfter=Yes ##
                for sentence in sentences:
                    sentence[-1]["space_after"] = False

                ## Write to output ##
                for new_sent_id, sentence in enumerate(sentences):
                    output.write(f"# filename = {filename}\n")
                    output.write(f"# sent_id = {sent_id}" + (f".{new_sent_id}" if new_sent_id != 0 else '') + "\n")
                    output.write(f"# text = {''.join(token['token'] + (' ' if token['space_after'] else '') for token in sentence)}\n")
                    for token_id, token in enumerate(sentence, start=1):
                        output.write(f"{token_id}\t{token['token']}\t_\t{token['pos']}\t{token['pos']}\t{token['special_attr']}\t{token['head']['token_id'] if token['head'] is not token else '0'}\t{token['relation_type']}\t_\tSpaceAfter={'Yes' if token['space_after'] else 'No'}\n")
                    output.write('\n')

            except Exception:
                pass
    