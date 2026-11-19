"""Benchmark free-text object search (no symbols, everyday Korean descriptions of use / kind).

    python -m vlmbaseline.spatial.bench_search [--aug outputs/augmented] [--agent Qwen/Qwen3.5-9B]

Each query has a set of acceptable labels; a hit = the top result (top-1) or one of the top 3 has such a label.
Compared: SigLIP2 image-text only, VLM descriptions only, both. With --agent, the tool-using agent answers
'<query> 어디 있어?' and is right when the first object symbol in its answer has an acceptable label.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from vlmbaseline.scene import attach_vlm, load_export
from vlmbaseline.spatial.search import TextEncoder, rank

Q = [
    ("앉아서 쉴 수 있는 곳", {"chair", "armchair", "bench", "office chair"}),
    ("초록색 식물", {"potted plant", "plant"}),
    ("불이 났을 때 쓰는 것", {"fire extinguisher", "fire alarm"}),
    ("컴퓨터 화면", {"monitor", "computer", "screen", "blank monitor"}),
    ("쓰레기 버리는 곳", {"trash can"}),
    ("문서를 출력하는 기계", {"printer"}),
    ("책이 꽂혀 있는 가구", {"bookshelf", "shelf"}),
    ("다른 층으로 이동하는 곳", {"elevator"}),
    ("음식을 데우는 기계", {"microwave"}),
    ("커피를 내리는 기계", {"coffee machine"}),
    ("마커로 글을 쓰는 판", {"whiteboard"}),
    ("서류를 넣어 두는 가구", {"filing cabinet", "cabinet", "drawer", "locker", "shelf"}),
    ("음료를 차갑게 보관하는 것", {"refrigerator", "small refrigerator"}),
    ("공부하거나 일하는 책상", {"desk", "table"}),
    ("밤에 불을 밝히는 스탠드", {"lamp"}),
    ("드나드는 출입구", {"door", "glass door"}),
    ("주황색 안내 표지", {"sign stand", "sign"}),
    ("자전거", {"bicycle"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--scenes", nargs="+", default=["library_merged", "lab_03"])
    ap.add_argument("--agent", default="")
    a = ap.parse_args()
    enc = TextEncoder()
    res = {"siglip": [], "vlm_text": [], "both": []}
    rows = []
    scenes = {}
    for name in a.scenes:
        s = load_export(Path("data") / name)
        n = attach_vlm(s, Path(a.aug) / name / "objects_vlm.json")
        scenes[name] = s
        present = {o.label for o in s.objects}
        for q, ok in Q:
            if not ok & present:
                continue
            for mode in res:
                r = rank(s.objects, q, enc if mode != "vlm_text" else None, k=3) if mode != "siglip" else \
                    rank([_no_vlm(o) for o in s.objects], q, enc, k=3)
                labels = [o.label for _, o, _ in r]
                res[mode].append((labels[0] in ok, bool(set(labels) & ok)))
                if mode == "both":
                    rows.append({"scene": name, "q": q, "top3": [f"{o.symbol} {o.label}" for _, o, _ in r],
                                 "hit1": labels[0] in ok})
        print(f"{name}: {n} objects with VLM descriptions")
    for mode, v in res.items():
        print(f"{mode:9s} top-1 {sum(x for x, _ in v)}/{len(v)}  top-3 {sum(y for _, y in v)}/{len(v)}")
    for r in rows:
        if not r["hit1"]:
            print("  miss:", r["scene"], r["q"], r["top3"])
    if a.agent:
        from vlmbaseline.bench.run import load
        from vlmbaseline.spatial.agent import Agent
        from vlmbaseline.spatial.tools import SpatialIndex
        proc, model = load(a.agent, "4bit")
        ok_n, tot = 0, 0
        for name, s in scenes.items():
            agent = Agent(proc, model, SpatialIndex(s, enc))
            lab = {o.symbol: o.label for o in s.objects}
            present = set(lab.values())
            for q, ok in Q:
                if not ok & present:
                    continue
                ans, trace = agent.ask(f"{q}는 어디 있어?")
                syms = [x for x in re.findall(r"O\d+", ans) if x in lab]
                hit = bool(syms) and lab[syms[0]] in ok
                ok_n += hit
                tot += 1
                print(f"[{'OK' if hit else '--'}] {name} {q} -> {ans[:100]!r} tools={[t['tool'] for t in trace]}")
        print(f"agent: {ok_n}/{tot}")
    Path("outputs/bench_search.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def _no_vlm(o):
    import copy
    c = copy.copy(o)
    c.vlm = None
    return c


if __name__ == "__main__":
    main()
