"""Conversation plumbing of the agent without a model: scripted turns stand in for generation."""
from pathlib import Path

import pytest

from vlmbaseline.spatial.agent import Agent


class Scripted(Agent):
    def __init__(self, idx, replies):
        self.idx, self.max_steps, self.base, self.replies, self.seen = idx, 5, "sys", list(replies), []

    def _gen(self, msgs, max_new_tokens=384, prefix=""):
        self.seen.append(msgs)
        return self.replies.pop(0)


def call(name, **args):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


@pytest.fixture(scope="module")
def idx():
    root = Path("data") / "lab_03"
    if not root.exists():
        pytest.skip("no export data")
    from vlmbaseline.scene import load_export
    from vlmbaseline.spatial.tools import SpatialIndex
    return SpatialIndex(load_export(root))


def two(idx):
    a, b = idx.s.objects[0].symbol, idx.s.objects[-1].symbol
    return a, b, idx.obj[b].label


def test_history_reaches_model(idx):
    a, _, _ = two(idx)
    ag = Scripted(idx, [call("object_info", symbol=a), f"{a} 옆입니다."])
    ag.ask("그 옆에 뭐가 있어?", history=[{"q": "의자 어디 있어?", "a": f"의자 {a}가 있습니다."}])
    texts = [m["content"][0]["text"] for m in ag.seen[0]]
    assert texts[1:4] == ["의자 어디 있어?", f"의자 {a}가 있습니다.", "그 옆에 뭐가 있어?"]


def test_start_from_previous_answer_only_when_pointed_back(idx):
    a, _, lab = two(idx)
    hist = [{"q": "어디 있어?", "a": f"{a}에 있습니다."}]
    ag = Scripted(idx, [call("guide_to", start=a, label=lab), "안내", "안내 (목적지 포함)"])
    _, tr = ag.ask(f"거기서 {lab}까지 가는 길 알려줘.", history=hist)
    assert "error" not in tr[0]["result"]  # '거기서' = the object of the answer before
    ag = Scripted(idx, [call("guide_to", start=a, label=lab), "어디에 계신가요?"])
    _, tr = ag.ask(f"{lab}까지 가는 길 알려줘.", history=hist)
    assert "error" in tr[0]["result"]  # no pointing back: the start was not given by the user


def test_unknown_tool_is_retried(idx):
    lab = idx.s.objects[0].label
    ag = Scripted(idx, [call("labels", label=lab), "1", call("count", label=lab), "답"])
    _, tr = ag.ask(f"{lab}는 몇 개야?")
    assert [t["tool"] for t in tr] == ["labels", "count"]
