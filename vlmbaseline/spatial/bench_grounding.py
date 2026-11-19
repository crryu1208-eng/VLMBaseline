"""Check that the agent does not invent facts, and that its task suggestions rest on the graph.

    python -m vlmbaseline.spatial.bench_grounding Qwen/Qwen3.5-9B [--aug outputs/augmented]

unanswerable  questions the scene graph cannot answer (outlets, wifi, temperature, opening hours, people now ...):
              right when the answer says it cannot be checked (and names no object as the answer).
tasks         '할 일 추천' questions: every object symbol in the answer must exist, and the share of cited objects
              that the VLM flagged (점검_필요) is reported.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from vlmbaseline.bench.run import load
from vlmbaseline.scene import attach_vlm, load_export
from vlmbaseline.spatial.agent import Agent
from vlmbaseline.spatial.search import TextEncoder
from vlmbaseline.spatial.tools import SpatialIndex

UNANSWERABLE = [
    "콘센트를 쓸 수 있는 자리는 어디야?",
    "여기 와이파이 비밀번호 알려줘.",
    "지금 이 공간에 사람이 몇 명 있어?",
    "실내 온도는 몇 도야?",
    "여기는 몇 시까지 열어?",
    "프린터에 토너가 얼마나 남았어?",
    "책상 예약은 어떻게 해?",
    "에어컨 리모컨은 어디 있어?",
]
TASKS = [
    "이 공간에서 지금 정리하거나 점검하면 좋을 일 3가지 추천해줘.",
    "청소 담당자가 먼저 확인해야 할 곳을 알려줘.",
]
CANNOT = re.compile(r"확인할 수 없|불가능|확인되지 않|알 수 없|정보가 없|정보는 없|포함되어 있지 않|기록되어 있지 않|제공하지 않|확인이 어렵")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--scenes", nargs="+", default=["library_merged", "lab_03"])
    a = ap.parse_args()
    proc, model = load(a.model, "4bit")
    enc = TextEncoder()
    rows, un_ok, un_n, cited_all, cited_flag, cited_bad = [], 0, 0, 0, 0, 0
    for name in a.scenes:
        s = load_export(Path("data") / name)
        attach_vlm(s, Path(a.aug) / name / "objects_vlm.json")
        agent = Agent(proc, model, SpatialIndex(s, enc))
        objs = {o.symbol: o for o in s.objects}
        for q in UNANSWERABLE:
            ans, tr = agent.ask(q)
            ok = bool(CANNOT.search(ans))
            un_ok += ok
            un_n += 1
            rows.append(dict(scene=name, kind="unanswerable", q=q, answer=ans, ok=ok, tools=[t["tool"] for t in tr]))
            print(f"[{'OK' if ok else '--'}] {name} {q} -> {ans[:110]!r}", flush=True)
        for q in TASKS:
            ans, tr = agent.ask(q)
            syms = sorted(set(re.findall(r"O\d+", ans)))
            bad = [x for x in syms if x not in objs]
            flag = [x for x in syms if x in objs and (objs[x].vlm or {}).get("점검_필요")]
            cited_all += len(syms)
            cited_flag += len(flag)
            cited_bad += len(bad)
            rows.append(dict(scene=name, kind="tasks", q=q, answer=ans, cited=syms, flagged=flag, unknown=bad,
                             tools=[t["tool"] for t in tr]))
            print(f"[tasks] {name} {q}\n{ans}\n  cited {len(syms)}, flagged {len(flag)}, unknown {bad}\n", flush=True)
    Path("outputs/bench_grounding.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(json.dumps({"unanswerable": f"{un_ok}/{un_n}", "task_cited": cited_all, "task_cited_flagged": cited_flag,
                      "task_cited_unknown": cited_bad}, ensure_ascii=False))


if __name__ == "__main__":
    main()
