"""Split the floor of an export into areas and describe each area with the VLM (Korean).

    python -m vlmbaseline.augment.describe_areas data/library_merged [--model Qwen/Qwen3.5-9B] [--out outputs/augmented]

Areas: spatial/areas.split_areas (walkable regions of ~7 m radius). For each area: 3 keyframes taken inside it,
facing different ways, plus the objects in it (names from objects_vlm.json when present); the VLM returns
name, kind (from a fixed list), one sentence, uses and notable things.
Writes <out>/<scene>/areas_vlm.json  {area id: {places, center, neighbours, objects, views, 이름, 종류, ...}}
and area_views/<id>.jpg (the images given to the model) for checking.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from vlmbaseline.augment.describe_objects import parse
from vlmbaseline.bench.run import generate, load
from vlmbaseline.scene import attach_vlm, load_export
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.areas import split_areas
from vlmbaseline.spatial.tools import SKIP_LANDMARK, SpatialIndex

KINDS = ["복도", "출입구·로비", "엘리베이터·계단 앞", "열람·학습 공간", "서가", "사무 공간", "회의 공간", "휴게 공간",
         "실험·작업 공간", "창고·보관 공간", "기타"]

PROMPT = """사진 {n}장은 건물 안 같은 구역 하나에서 여러 방향을 바라보고 찍은 것입니다.
3D 씬그래프로 잰 이 구역의 모양: {shape}
이 구역에 있는 물체: {objects}
사진들을 합쳐서 이 구역 하나를 아래 JSON 객체 하나로만 설명하고 값은 한국어로 쓰세요(사진별로 따로 쓰지 마세요).
{{"종류": "", "이름": "", "설명": "한 문장", "용도": ["", ""], "특징": ["", ""]}}
- 종류: 다음 중 하나: {kinds}. 폭이 좁고 긴 통로 모양이면 대개 복도입니다.
- 이름: 사람이 길 안내에서 부를 만한 짧은 이름. '눈에 띄는 특징 + 종류' 꼴로 짓고, 물체 하나의 이름만으로 짓지 마세요
  (물체 기호나 영어 금지).
- 특징: 이 구역을 알아보게 해 주는 눈에 띄는 것 2~3개
- 사진과 물체 목록에서 확인되지 않는 내용은 지어내지 마세요."""


def area_of_places(areas):
    return {p: a for a, ps in areas.items() for p in ps}


def pick_views(kfs, k=3, min_gap_s=2.0):
    """k keyframes facing different ways (greedy: largest smallest heading difference)."""
    if not kfs:
        return []
    yaw = {id(f): nav.pose_from_T(f.world_T_sensor)[2] for f in kfs}
    out = [kfs[len(kfs) // 2]]
    while len(out) < k:
        best, bd = None, -1.0
        for f in kfs:
            if any(abs(f.timestamp_ns - g.timestamp_ns) < min_gap_s * 1e9 for g in out):
                continue
            d = min(abs((yaw[id(f)] - yaw[id(g)] + math.pi) % (2 * math.pi) - math.pi) for g in out)
            if d > bd:
                best, bd = f, d
        if best is None:
            break
        out.append(best)
    return out


def shape(idx, places, cell=0.2):
    """Floor area (m², polygons rasterised), longest walk inside the area (m), mean width = area / length."""
    from PIL import ImageDraw
    pts = np.vstack([idx.place[p].boundary[:, :2] for p in places])
    lo = pts.min(0)
    lo = lo - 1.0
    wh = np.ceil((pts.max(0) - lo) / cell).astype(int) + 10
    im = Image.new("1", (int(wh[0]), int(wh[1])), 0)
    d = ImageDraw.Draw(im)
    for p in places:
        d.polygon([tuple((q[:2] - lo) / cell) for q in idx.place[p].boundary], fill=1)
    from PIL import ImageFilter
    im = im.convert("L").filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))  # close gaps ~1.2 m
    area = float((np.asarray(im) > 0).sum()) * cell * cell
    inside = set(places)
    adj = {p: [(v, w) for v, w in idx.adj[p] if v in inside] for p in places}

    def far(src):
        dist, pq = {src: 0.0}, [(0.0, src)]
        import heapq
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist[u]:
                continue
            for v, w in adj[u]:
                if dd + w < dist.get(v, 1e18):
                    dist[v] = dd + w
                    heapq.heappush(pq, (dd + w, v))
        u = max(dist, key=dist.get)
        return u, dist[u]

    u, _ = far(places[0])
    _, length = far(u)
    length = max(length, 1.0)
    width = area / length
    r = {"area_m2": round(area), "length_m": round(length, 1), "width_m": round(width, 1)}
    r["corridor_like"] = bool(width < 3.0 and length > 2.5 * width)
    return r


def shape_text(sh):
    t = f"바닥 넓이 약 {sh['area_m2']} m², 걸어서 가장 긴 길이 약 {sh['length_m']} m, 평균 폭 약 {sh['width_m']} m"
    return t + (" (폭이 좁고 긴 통로 모양)" if sh["corridor_like"] else "")


def area_table(idx, areas):
    """Per area: places, centre, neighbour areas, objects, keyframes taken inside."""
    owner = area_of_places(areas)
    ids, cs = list(idx.adj), None
    cs = np.array([idx.place[p].center[:2] for p in ids])
    kf_in = {a: [] for a in areas}
    for f in idx.s.keyframes:
        x, y, _ = nav.pose_from_T(f.world_T_sensor)
        kf_in[owner[ids[int(np.linalg.norm(cs - [x, y], axis=1).argmin())]]].append(f)
    tab = {}
    for a, ps in areas.items():
        nb = Counter(owner[v] for p in ps for v, _ in idx.adj[p] if owner[v] != a)
        objs = [o for o in idx.s.objects if owner.get(o.place) == a]
        tab[a] = {"places": ps, "shape": shape(idx, ps), "center": np.mean([idx.place[p].center[:2] for p in ps], 0).round(2).tolist(),
                  "neighbours": sorted(nb), "objects": [o.symbol for o in objs], "kfs": kf_in[a]}
    return tab


def object_list(idx, syms, n=25):
    names = Counter()
    for s in syms:
        o = idx.obj[s]
        if o.label in SKIP_LANDMARK:
            continue
        names[(o.vlm or {}).get("이름") or o.label] += 1
    return ", ".join(f"{k} {v}개" if v > 1 else k for k, v in names.most_common(n)) or "(없음)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", default="outputs/augmented")
    ap.add_argument("--radius", type=float, default=7.0)
    a = ap.parse_args()
    proc, model = load(a.model, "4bit")
    for e in a.exports:
        s, name = load_export(e), Path(e).name
        out = Path(a.out) / name
        attach_vlm(s, out / "objects_vlm.json")
        idx = SpatialIndex(s)
        areas = split_areas(idx, radius=a.radius)
        tab = area_table(idx, areas)
        (out / "area_views").mkdir(parents=True, exist_ok=True)
        res, t0 = {}, time.perf_counter()
        for aid, t in tab.items():
            views = pick_views(t.pop("kfs"))
            ims = [Image.open(f.rgb).convert("RGB") for f in views]
            r = dict(t, views=[f.node for f in views])
            if not ims:
                res[aid] = dict(r, ok=False, error="no keyframe inside")
                continue
            sheet = Image.new("RGB", (sum(i.width for i in ims), max(i.height for i in ims)))
            x = 0
            for i in ims:
                sheet.paste(i, (x, 0))
                x += i.width
            sheet.thumbnail((1800, 400))
            sheet.save(out / "area_views" / f"{aid}.jpg", quality=85)
            objs = object_list(idx, t["objects"])
            text, dt, _, _ = generate(proc, model, ims, PROMPT.format(n=len(ims), shape=shape_text(t["shape"]), objects=objs,
                                                                 kinds=", ".join(KINDS)), 260)
            j = parse(text)
            if isinstance(j, list):  # one object per image despite the prompt: keep the first
                j = j[0] if j and isinstance(j[0], dict) else None
            if j and j.get("종류") not in KINDS:
                j["종류"] = "기타"
            res[aid] = dict(r, **(j or {}), object_list=objs, raw=text, ok=j is not None, sec=round(dt, 2))
            print(f"{name} area {aid} ({shape_text(t['shape'])}, {len(t['objects'])} objects): "
                  f"{json.dumps(j, ensure_ascii=False)[:160] if j else text[:80]!r} ({dt:.1f}s)", flush=True)
        (out / "areas_vlm.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"== {name}: {sum(r['ok'] for r in res.values())}/{len(res)} areas, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
