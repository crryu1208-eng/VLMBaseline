"""Score benchmark runs and print a table.

    python -m vlmbaseline.bench.score outputs/bench_runs
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# accepted answers per reviewed label (same object, a common synonym or the direct parent class)
SYN = {
    "air conditioner": ["air conditioner", "ac unit", "air conditioning"],
    "armchair": ["armchair", "chair", "lounge chair", "sofa chair", "accent chair"],
    "bench": ["bench", "sofa", "couch", "ottoman", "seat"],
    "bicycle": ["bicycle", "bike"],
    "blank monitor": ["monitor", "screen", "tv", "television", "display"],
    "book": ["book", "magazine", "brochure", "journal"],
    "bookshelf": ["bookshelf", "bookcase", "book shelf", "shelf", "shelves", "shelving"],
    "box": ["box", "boxes", "bin", "container", "crate", "storage"],
    "cabinet": ["cabinet", "cupboard", "locker", "storage unit", "sideboard"],
    "cart": ["cart", "hand truck", "dolly", "trolley"],
    "ceiling light": ["ceiling light", "light", "lamp", "light fixture", "fluorescent"],
    "chair": ["chair"],
    "coffee machine": ["coffee machine", "coffee maker", "espresso"],
    "coffee table": ["coffee table", "table"],
    "computer": ["computer", "imac", "monitor", "all-in-one"],
    "desk": ["desk", "table", "counter", "workstation"],
    "document": ["document", "poster", "paper", "notice", "flyer", "sign"],
    "door": ["door"],
    "drawer": ["drawer", "cabinet"],
    "elevator": ["elevator", "lift"],
    "filing cabinet": ["filing cabinet", "file cabinet", "cabinet", "drawer"],
    "fire alarm": ["fire alarm", "alarm", "smoke detector", "call point"],
    "fire extinguisher": ["fire extinguisher", "extinguisher"],
    "folder": ["folder", "binder"],
    "glass door": ["glass door", "door"],
    "image sign": ["sign", "pictogram", "icon", "symbol", "plaque", "emblem"],
    "keyboard": ["keyboard"],
    "lamp": ["lamp", "light", "lampshade", "lamp shade"],
    "locker": ["locker", "cabinet"],
    "microwave": ["microwave"],
    "monitor": ["monitor", "screen", "display", "computer", "tablet", "imac"],
    "name plate": ["name plate", "nameplate", "plaque", "sign", "logo"],
    "notice": ["notice", "sign", "poster", "board", "paper", "flyer"],
    "office chair": ["office chair", "chair"],
    "paper": ["paper", "document", "poster", "notice", "sign", "flyer", "note"],
    "partition": ["partition", "divider", "cubicle", "panel"],
    "plant": ["plant"],
    "potted plant": ["potted plant", "plant", "flower pot", "pot"],
    "printer": ["printer", "copier", "photocopier", "copy machine"],
    "refrigerator": ["refrigerator", "fridge"],
    "screen": ["screen", "monitor", "display", "panel", "thermostat", "controller"],
    "shelf": ["shelf", "shelves", "shelving", "cabinet", "bookshelf", "bookcase", "rack"],
    "sign": ["sign", "signboard", "board", "stand"],
    "sign stand": ["sign stand", "sign", "stand", "signboard"],
    "small refrigerator": ["refrigerator", "fridge"],
    "sticky note": ["sticky note", "post-it", "note"],
    "table": ["table", "desk"],
    "trash can": ["trash", "bin", "wastebasket", "garbage", "waste basket", "basket"],
    "whiteboard": ["whiteboard", "board", "calendar"],
    "window": ["window"],
}


def norm(s):
    return " " + re.sub(r"[^a-z0-9\- ]", " ", s.lower()).strip() + " "


def recog_ok(r):
    a = norm(r["answer"])
    return any(re.search(rf"\b{re.escape(w)}(s|es)?\b", a) for w in SYN.get(r["label"], [r["label"]]))


def qa_score(r):
    gt, a = r["gt"], r["answer"]
    if r["kind"] in ("nearest", "largest", "highest"):
        m = re.findall(r"O\d+", a)
        return bool(m) and m[0] == gt
    nums = re.findall(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
    if not nums:
        return False
    v = float(nums[0])
    if r["kind"] == "count":
        return v == gt
    return abs(v - gt) <= max(0.3, 0.1 * gt)   # distance: 10% or 0.3 m


def describe_ok(r):
    s = r["answer"]
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return False, False
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False, False
    keys = {"이름", "색", "재질", "상태", "용도"}
    korean = all(re.search(r"[가-힣]", json.dumps(j.get(k, ""), ensure_ascii=False)) for k in ("이름", "용도"))
    return keys <= set(j), korean


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/bench_runs")
    print("| model | recog acc | recog lib / lab | describe JSON ok | describe Korean | scene QA acc "
          "(count / nearest / distance / max) | s/item recog / describe / QA | VRAM peak GB | load s |")
    print("|" + "---|" * 9)
    for d in sorted(p for p in root.iterdir() if (p / "run.json").exists()):
        st = json.loads((d / "run.json").read_text())
        rec = load(d / "recog.jsonl")
        ok = [recog_ok(r) for r in rec]
        lib = [o for o, r in zip(ok, rec) if r["scene"] == "library_merged"]
        lab = [o for o, r in zip(ok, rec) if r["scene"] == "lab_03"]
        des = [describe_ok(r) for r in load(d / "describe.jsonl")]
        qa = load(d / "scene_qa.jsonl")
        qs = [qa_score(r) for r in qa]
        kinds = {}
        for r, s in zip(qa, qs):
            k = "max" if r["kind"] in ("largest", "highest") else r["kind"]
            kinds.setdefault(k, []).append(s)
        pct = lambda v: f"{100 * sum(v) / len(v):.0f}%" if v else "-"  # noqa: E731
        print(f"| {st['model'].split('/')[-1]} ({st['quant']}) | {pct(ok)} ({sum(ok)}/{len(ok)}) | {pct(lib)} / {pct(lab)} "
              f"| {pct([x[0] for x in des])} | {pct([x[1] for x in des])} "
              f"| {pct(qs)} ({' / '.join(pct(kinds.get(k, [])) for k in ('count', 'nearest', 'distance', 'max'))}) "
              f"| {st.get('recog_sec_median')} / {st.get('describe_sec_median')} / {st.get('scene_qa_sec_median')} "
              f"| {st['vram_peak_gb']} | {st['load_s']} |")


if __name__ == "__main__":
    main()
