"""Benchmark questions about areas (zones named by the VLM, augment/describe_areas.py).

    python -m vlmbaseline.spatial.bench_areas Qwen/Qwen3.5-9B [--aug outputs/augmented]

where     '지금 여기가 어디야?' from keyframe poses: right when the answer names the area the pose is in.
to_name   '여기서 <area name>까지 어떻게 가?': right when a guidance tool went to that area.
to_kind   the same asked by purpose ('쉴 수 있는 곳으로 안내해줘'), hand-written per scene with the expected area.
list      '이 층에는 어떤 공간들이 있어?': right when 3 or more area names are in the answer.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from vlmbaseline.bench.run import load
from vlmbaseline.scene import attach_areas, attach_vlm, load_export
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.agent import Agent
from vlmbaseline.spatial.search import TextEncoder
from vlmbaseline.spatial.tools import SpatialIndex

# question by purpose -> expected area name (names from areas_vlm.json of the current augmentation)
BY_KIND = {
    "library_merged": [("쉴 수 있는 휴게 공간으로 안내해줘.", "창가 휴게 공간"),
                       ("컴퓨터를 쓸 수 있는 곳으로 가는 길 알려줘.", "컴퓨터 열람 공간")],
    "lab_03": [("엘리베이터 타는 곳으로 가는 길 알려줘.", "엘리베이터 앞"),
               ("로비로 가려면 어떻게 가?", "원형 유리문 로비")],
}


def went_to(trace, aid):
    for t in trace:
        r = t["result"]
        if t["tool"] in ("guide_me", "guide_to") and isinstance(r, dict) and \
                isinstance(r.get("goal_area"), dict) and r["goal_area"].get("area") == aid:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--scenes", nargs="+", default=["library_merged", "lab_03"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    proc, model = load(a.model, "4bit")
    enc = TextEncoder()
    rows = []

    def rec(kind, name, q, ans, tr, ok, **kw):
        rows.append(dict(scene=name, kind=kind, q=q, answer=ans, ok=ok, tools=[t["tool"] for t in tr],
                         args=[t["args"] for t in tr], **kw))
        print(f"[{'OK' if ok else '--'}] {kind:8s} {name} {q} -> {ans[:130]!r} {[t['tool'] for t in tr]}", flush=True)

    for name in a.scenes:
        s = load_export(Path("data") / name)
        attach_vlm(s, Path(a.aug) / name / "objects_vlm.json")
        attach_areas(s, Path(a.aug) / name / "areas_vlm.json")
        idx = SpatialIndex(s, enc)
        agent = Agent(proc, model, idx)
        by_name = {}
        for aid, r in s.areas.items():
            by_name.setdefault(r.get("이름"), []).append(aid)
        kfs = s.keyframes[len(s.keyframes) // 12::len(s.keyframes) // 6][:6]
        for kf in kfs:
            pose = nav.pose_from_T(kf.world_T_sensor)
            idx.pose = pose
            gt = s.areas[idx.area_of[idx._pose_place()]]["이름"]
            q = "지금 여기가 어디야?"
            ans, tr = agent.ask(q, pose=pose)
            rec("where", name, q, ans, tr, gt in ans, gt=gt)
        unique = [n for n, ids in by_name.items() if len(ids) == 1]
        for tgt in rng.sample(unique, min(4, len(unique))):
            aid = by_name[tgt][0]
            kf = rng.choice(s.keyframes)
            pose = nav.pose_from_T(kf.world_T_sensor)
            idx.pose = pose
            if idx.area_of.get(idx._pose_place()) == aid:
                continue
            q = f"여기서 {tgt}까지 어떻게 가?"
            ans, tr = agent.ask(q, pose=pose)
            rec("to_name", name, q, ans, tr, went_to(tr, aid), gt=tgt)
        for q, tgt in BY_KIND.get(name, []):
            if tgt not in by_name:
                print(f"skip {q}: no area named {tgt}")
                continue
            aid = by_name[tgt][0]
            for _ in range(5):  # a start pose outside the goal area
                pose = nav.pose_from_T(rng.choice(s.keyframes).world_T_sensor)
                idx.pose = pose
                if idx.area_of.get(idx._pose_place()) != aid:
                    break
            ans, tr = agent.ask(q, pose=pose)
            rec("to_kind", name, q, ans, tr, went_to(tr, aid), gt=tgt)
        q = "이 층에는 어떤 공간들이 있어?"
        ans, tr = agent.ask(q)
        hit = sum(1 for n in by_name if n and n in ans)
        rec("list", name, q, ans, tr, hit >= 3, named=hit)
    Path("outputs/bench_areas.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    out = {}
    for k in ("where", "to_name", "to_kind", "list"):
        v = [r["ok"] for r in rows if r["kind"] == k]
        out[k] = f"{sum(v)}/{len(v)}"
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
