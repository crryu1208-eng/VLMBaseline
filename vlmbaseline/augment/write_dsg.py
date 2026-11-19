"""Write the VLM descriptions into a copy of the export's scene graph (spark_dsg JSON).

    python -m vlmbaseline.augment.write_dsg data/library_merged [--aug outputs/augmented]

Each described object node gets attributes.metadata.vlm = {이름, 라벨_맞음, 색, 재질, 상태, 점검_필요, 점검_이유,
용도, 설명, views, model}; the graph metadata gets vlm = {model, date, n}. Everything else is unchanged, so
spark_dsg and the DSG viewer read it as before. Writes <aug>/<scene>/dsg_vlm.json and dsg_with_mesh_vlm.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

KEYS = ("이름", "라벨_맞음", "색", "재질", "상태", "점검_필요", "점검_이유", "용도", "설명", "views")


def symbol(node_id):
    return f"{chr(node_id >> 56)}{node_id & ((1 << 56) - 1)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B (4-bit)")
    a = ap.parse_args()
    root, name = Path(a.export), Path(a.export).name
    vlm = json.loads((Path(a.aug) / name / "objects_vlm.json").read_text())
    for src, dst in (("dsg.json", "dsg_vlm.json"), ("dsg_with_mesh.json", "dsg_with_mesh_vlm.json")):
        p = root / "hydra/backend" / src
        if not p.exists():
            continue
        g = json.loads(p.read_text())
        n = 0
        for node in g["nodes"]:
            if node["layer"] != 2 or node.get("partition", 0) != 0:
                continue
            r = vlm.get(symbol(node["id"]))
            if r and r.get("ok"):
                meta = node["attributes"].setdefault("metadata", {})
                meta["vlm"] = {k: r[k] for k in KEYS if k in r} | {"model": a.model}
                n += 1
        g.setdefault("metadata", {})["vlm"] = {"model": a.model, "date": datetime.date.today().isoformat(),
                                                "objects_described": n}
        out = Path(a.aug) / name / dst
        out.write_text(json.dumps(g, ensure_ascii=False))
        print(f"{out}: {n} objects with metadata.vlm")


if __name__ == "__main__":
    main()
