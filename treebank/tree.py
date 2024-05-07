from typing import overload, TypedDict, Required, Literal, NamedTuple
from collections import Counter
from itertools import chain
import re, torch
from .token import Token, TokenDict

class TreeDict(TypedDict):
    filename: Required[str | None]
    sent_id: Required[str]
    text: Required[str]
    tokens: Required[list[TokenDict]]
    is_projective: Required[bool]
    num_non_projective_arcs: Required[int]

class Tree:

    header_pattern = re.compile(r"^# (?P<key>.+) = (?P<value>.+)$")
    conllu_format = "# filename = {filename}\n# sent_id = {sent_id}\n# text = {text}\n{body}"
    _ROOT = Token.create_dummy_token(0, "ROOT", True)
    _END = Token.create_dummy_token(-1, "END", False)

    def __init__(self, raw_conllu: str):
        raw_lines = raw_conllu.split('\n')
        headers: dict[str, str] = {}
        for i, raw_line in enumerate(raw_lines):
            match = self.header_pattern.fullmatch(raw_line)
            if match:
                headers[match["key"]] = match["value"]
            else:
                raw_lines = raw_lines[i:]
                break
        # assert "filename" in headers, f"Missing 'filename' header\n{headers}"
        assert "sent_id" in headers, f"Missing 'sent_id' header\n{headers}"
        assert "text" in headers, f"Missing 'text' header\n{headers}"
        self.filename = headers.get("filename")
        self.sent_id = headers["sent_id"]
        self.text = headers["text"]
        self.__tokens: list[Token] = []
        found_root = False
        for i, raw_line in enumerate(raw_lines, start=1):
            token = Token(raw_line)
            assert token.id == i, f"Non-sequential token id in {self} at {token}"
            if token.is_root:
                assert not found_root, f"Multiple root in {self}. Second root at {token}"
                found_root = True
                token.head_token = self._ROOT
                self.root_token = token
            self.__tokens.append(token)
        assert found_root, f"Root not found in {self}"
        self[-1].miscs["SpaceAfter"] = "No"
        reconstructed_text = ''.join(token.form if token.miscs["SpaceAfter"] == "No" else token.form + ' ' for token in self)
        assert reconstructed_text == self.text, f"Text mismatch in {self}\nHeader: {self.text}\nActual: {reconstructed_text}"
        # Link head tokens
        for token in self:
            if not token.is_root:
                token.head_token = self[token.head - 1]
                token.head_token.dep_tokens.append(token)
        # Check for cycles
        for start_token in self:
            token = start_token
            loop_counter = 0
            while not token.is_root:
                token = token.head_token
                loop_counter += 1
                assert loop_counter < len(self), f"Loop in {self} starting at {start_token}"
        # Check for projectivity
        tree_is_projective = True
        for dep in self:
            if dep.is_root:
                continue
            head = dep.head_token
            start, stop = (dep.id, head.id - 1) if dep.id < head.id else (head.id, dep.id - 1)
            arc_is_projective = True
            for token in self[start:stop]:
                token = token.head_token
                while token is not head:
                    if token.is_root:
                        arc_is_projective = tree_is_projective = False
                        break
                    token = token.head_token
                if not arc_is_projective:
                    break
            dep.arc_is_projective = arc_is_projective
        self.is_projective = tree_is_projective
        self.num_non_projective_arcs = 0 if tree_is_projective else sum(not token.arc_is_projective for token in self)

    def copy(self):
        return type(self)(self.to_conllu())

    def __len__(self):
        return len(self.__tokens)

    def __iter__(self):
        return iter(self.__tokens)

    def __reversed__(self):
        return reversed(self.__tokens)

    @overload
    def __getitem__(self, index: int) -> Token: ...
    @overload
    def __getitem__(self, index: slice) -> list[Token]: ...
    def __getitem__(self, index: int | slice):
        return self.__tokens[index]

    def __repr__(self):
        return f"<{type(self).__name__}{' ' + self.filename if self.filename else ''}: {self.sent_id}>"

    def to_conllu(self):
        return self.conllu_format.format(
            filename=self.filename,
            sent_id=self.sent_id,
            text=self.text,
            body='\n'.join(token.to_conllu() for token in self)
        )

    def to_dict(self):
        return TreeDict(
            filename=self.filename,
            sent_id=self.sent_id,
            text=self.text,
            tokens=[token.to_dict() for token in self],
            is_projective=self.is_projective,
            num_non_projective_arcs=self.num_non_projective_arcs
        )

    def to_adjacency_matrix(self):
        matrix = torch.zeros(len(self) + 1, len(self) + 1)
        for token in self:
            matrix[token.head, token.id] = 1
        return matrix

    def to_transitions(self, action_set: Literal["standard", "eager"]):
        stack = [self._ROOT]
        buffer = self.__tokens + [self._END]
        relations: list[Relation] = []
        transitions: list[tuple[TransitionState, str]] = []
        def add_transition(action: str):
            transitions.append((
                TransitionState(
                    stack=stack.copy(),
                    buffer=buffer.copy(),
                    relations=relations.copy()
                ),
                action
            ))
        if action_set == "standard":
            if not self.is_projective:
                raise ValueError(f"{action_set!r} action set cannot be used with non-projective trees")
            while len(stack) > 1 or len(buffer) > 1:
                top = stack[-1]
                second = stack[-2] if len(stack) > 1 else None
                if (
                    second and
                    second.head_token is top
                ):
                    add_transition(f"LeftArc-{second.deprel}")
                    relations.append(
                        Relation(
                            head=top,
                            dep=second,
                            deprel=second.deprel
                        )
                    )
                    stack.pop(-2)
                elif (
                    second and
                    top.head_token is second and
                    not any(token in top.dep_tokens for token in chain(stack, buffer))
                ):
                    add_transition(f"RightArc-{top.deprel}")
                    relations.append(
                        Relation(
                            head=second,
                            dep=top,
                            deprel=top.deprel
                        )
                    )
                    stack.pop()
                else:
                    add_transition("Shift")
                    stack.append(buffer.pop(0))
        elif action_set == "eager":
            if not self.is_projective:
                raise ValueError(f"{action_set!r} action set cannot be used with non-projective trees")
            while len(stack) > 1 or len(buffer) > 1:
                top = stack[-1]
                front = buffer[0] if len(buffer) > 1 else None
                if (
                    front and
                    top.head_token is front
                ):
                    add_transition(f"LeftArc-{top.deprel}")
                    relations.append(
                        Relation(
                            head=front,
                            dep=top,
                            deprel=top.deprel
                        )
                    )
                    stack.pop()
                elif (
                    front and
                    front.head_token is top
                ):
                    add_transition(f"RightArc-{front.deprel}")
                    relations.append(
                        Relation(
                            head=top,
                            dep=front,
                            deprel=front.deprel
                        )
                    )
                    stack.append(buffer.pop(0))
                elif (
                    any(relation.dep is top for relation in relations) and
                    not any(token in top.dep_tokens for token in chain(stack, buffer))
                ):
                    add_transition("Reduce")
                    stack.pop()
                else:
                    add_transition("Shift")
                    stack.append(buffer.pop(0))
        else:
            raise ValueError(f"Unknown action set: {action_set!r}")
        token_count = Counter(relation.dep for relation in relations)
        assert all(count == 1 for count in token_count.values()), f"Invalid {action_set!r} transition sequence in {self}: Multiple head tokens for the same token"
        assert set(token_count) == set(self.__tokens), f"Invalid {action_set!r} transition sequence in {self}: Not all tokens have a head token"
        return transitions

class Relation(NamedTuple):
    head: Token
    dep: Token
    deprel: str

    def __repr__(self):
        return f"({self.head.form} -{self.deprel}-> {self.dep.form})"

class TransitionState(NamedTuple):
    stack: list[Token]
    buffer: list[Token]
    relations: list[Relation]

    def __repr__(self):
        return f"([{', '.join(token.form for token in self.stack)}], [{', '.join(token.form for token in self.buffer)}], [{', '.join(repr(relation) for relation in self.relations)}])"
