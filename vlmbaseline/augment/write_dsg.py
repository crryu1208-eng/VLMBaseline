"""Write the VLM descriptions into a copy of the export's scene graph (spark_dsg JSON).

    python -m vlmbaseline.augment.write_dsg data/library_merged [--aug outputs/augmented]

Each described object node gets attributes.metadata.vlm = {이름, 라벨_맞음, 색, 재질, 상태, 점검_필요, 점검_이유,
용도, 설명, views, model}; with areas_vlm.json (describe_areas.py) each 2D place node gets
attributes.metadata.vlm_area = {area, 이름} and the graph metadata vlm.areas = {id: {이름, 종류, 설명, 용도, 특징,
center, neighbours, n_places}}; the graph metadata gets vlm = {model, date, n, areas}. Everything else is unchanged, so
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
    ap_ = Path(a.aug) / name / "areas_vlm.json"
    areas = {int(k): r for k, r in json.loads(ap_.read_text()).items() if r.get("ok")} if ap_.exists() else {}
    area_of = {p: k for k, r in areas.items() for p in r.get("places", [])}
    for src, dst in (("dsg.json", "dsg_vlm.json"), ("dsg_with_mesh.json", "dsg_with_mesh_vlm.json")):
        p = root / "hydra/backend" / src
        if not p.exists():
            continue
        g = json.loads(p.read_text())
        n = 0
        n_places = 0
        for node in g["nodes"]:
            if node["layer"] == 3 and node.get("partition", 0) == 1 and node["id"] in area_of:
                k = area_of[node["id"]]
                node["attributes"].setdefault("metadata", {})["vlm_area"] = {"area": k, "이름": areas[k].get("이름")}
                n_places += 1
                continue
            if node["layer"] != 2 or node.get("partition", 0) != 0:
                continue
            r = vlm.get(symbol(node["id"]))
            if r and r.get("ok"):
                meta = node["attributes"].setdefault("metadata", {})
                meta["vlm"] = {k: r[k] for k in KEYS if k in r} | {"model": a.model}
                n += 1
        g.setdefault("metadata", {})["vlm"] = {"model": a.model, "date": datetime.date.today().isoformat(),
                                                "objects_described": n}
        if areas:
            g["metadata"]["vlm"]["areas"] = {
                str(k): {kk: r[kk] for kk in ("이름", "종류", "설명", "용도", "특징", "center", "neighbours") if kk in r}
                | {"n_places": len(r.get("places", []))} for k, r in areas.items()}
        out = Path(a.aug) / name / dst
        out.write_text(json.dumps(g, ensure_ascii=False))
        print(f"{out}: {n} objects with metadata.vlm, {n_places} places with vlm_area, {len(areas)} areas")


if __name__ == "__main__":
    main()
