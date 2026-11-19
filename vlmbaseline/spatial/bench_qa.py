"""Benchmark the tool-using agent on scene questions.

    python -m vlmbaseline.spatial.bench_qa Qwen/Qwen3.5-9B --set outputs/bench_set --out outputs/bench_runs

Two question sets per scene:
  structured  the 40 questions of bench 1 (scene_qa.jsonl, English labels, '숫자만 / 기호만')
  natural     Korean everyday wording (Korean object names, no label hints): counts, nearest, walking
              distance, guidance ('...까지 가는 길 알려줘'); ground truth from SpatialIndex.
Writes <out>/<model>/agent_qa.jsonl and prints accuracy per kind.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from vlmbaseline.bench.run import load
from vlmbaseline.scene import load_export
from vlmbaseline.spatial.agent import Agent
from vlmbaseline.spatial.tools import SpatialIndex

KO = {"chair": "의자", "bookshelf": "책장", "potted plant": "화분", "monitor": "모니터", "lamp": "스탠드 조명",
      "door": "문", "printer": "프린터", "fire extinguisher": "소화기", "whiteboard": "화이트보드",
      "trash can": "쓰레기통", "desk": "책상", "filing cabinet": "서류 캐비닛", "elevator": "엘리베이터",
      "bench": "벤치", "table": "테이블", "ceiling light": "천장 조명", "box": "상자", "cabinet": "수납장",
      "partition": "파티션", "armchair": "안락의자", "sign stand": "안내판"}


def natural_set(idx: SpatialIndex, name, rng, n=4):
    labels = [l for l in KO if idx.labels.get(l, 0) >= 1]
    multi = [l for l in labels if idx.labels[l] >= 2]
    objs = [o for o in idx.s.objects if o.label in KO]
    qs = []
    for l in rng.sample(labels, min(n, len(labels))):
        qs.append(dict(kind="n_count", q=f"여기 {KO[l]}는 몇 개 있어?", gt=idx.labels[l],
                       alt=idx.count(l)["count"]))
    for _ in range(n):
        a = rng.choice(objs)
        l = rng.choice([m for m in multi if m != a.label])
        gt = idx.nearest(a.symbol, l)["nearest"][0]["symbol"]
        qs.append(dict(kind="n_nearest", q=f"{KO[a.label]} {a.symbol} 근처에서 제일 가까운 {KO[l]}는 어느 거야?", gt=gt))
    for _ in range(n):
        a, b = rng.sample(objs, 2)
        gt = idx.distance(a.symbol, b.symbol)["walking_m"]
        qs.append(dict(kind="n_walk", q=f"{KO[a.label]} {a.symbol}에서 {KO[b.label]} {b.symbol}까지 걸어가면 몇 m 정도야?",
                       gt=gt))
    for _ in range(n):
        a = rng.choice(objs)
        l = rng.choice([m for m in labels if m != a.label])
        tgt = idx.nearest(a.symbol, l)["nearest"][0]["symbol"]
        qs.append(dict(kind="n_guide", q=f"지금 {KO[a.label]} {a.symbol} 앞에 있어. 가장 가까운 {KO[l]}까지 가는 길 알려줘.",
                       gt=tgt, gt_walk=idx.route(a.symbol, tgt)["walking_m"]))
    for i, q in enumerate(qs):
        q.update(id=f"{name}_n{i:02d}", scene=name)
    return qs


def first_num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m[0]) if m else None


def score(r):
    a, k, gt = r["answer"], r["kind"], r["gt"]
    syms = re.findall(r"O\d+", a)
    if k in ("nearest", "largest", "highest"):
        return bool(syms) and syms[0] == gt
    if k == "n_nearest":
        return gt in syms
    if k == "n_guide":
        return gt in syms and any(t["tool"] in ("route", "guide_to") for t in r["trace"])
    nums = [float(x) for x in re.findall(r"(?<![O\d])-?\d+(?:\.\d+)?", a.replace(",", ""))]
    if k in ("count", "n_count"):
        return any(v in (gt, r.get("alt", gt)) for v in nums)
    if k in ("distance",):
        return bool(nums) and abs(nums[0] - gt) <= max(0.3, 0.1 * gt)
    if k == "n_walk":
        return any(abs(v - gt) <= max(0.3, 0.1 * gt) for v in nums)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--set", default="outputs/bench_set")
    ap.add_argument("--out", default="outputs/bench_runs")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    set_dir, rng = Path(a.set), random.Random(a.seed)
    structured = [json.loads(l) for l in (set_dir / "scene_qa.jsonl").read_text().splitlines()]
    proc, model = load(a.model, a.quant)
    rows = []
    for name in sorted({q["scene"] for q in structured}):
        idx = SpatialIndex(load_export(Path("data") / name))
        agent = Agent(proc, model, idx)
        qs = [q for q in structured if q["scene"] == name] + natural_set(idx, name, rng)
        for q in qs:
            t = time.perf_counter()
            ans, trace = agent.ask(q["q"])
            r = dict(q, answer=ans, trace=trace, sec=round(time.perf_counter() - t, 2))
            r["ok"] = score(r)
            rows.append(r)
            print(f"[{'OK' if r['ok'] else '--'}] {q['kind']:9s} {q['q'][:50]} -> {ans[:90]!r} ({r['sec']}s)", flush=True)
    out = Path(a.out) / a.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "agent_qa.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r["ok"])
    s = [r["ok"] for r in rows if not r["kind"].startswith("n_")]
    n = [r["ok"] for r in rows if r["kind"].startswith("n_")]
    print(json.dumps({"structured": f"{sum(s)}/{len(s)}", "natural": f"{sum(n)}/{len(n)}",
                      **{k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())},
                      "sec_median": float(np.median([r["sec"] for r in rows]))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
