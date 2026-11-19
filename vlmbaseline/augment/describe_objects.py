"""Augment the objects of a scene graph export with VLM descriptions (Korean).

    python -m vlmbaseline.augment.describe_objects data/library_merged [--model Qwen/Qwen3.5-9B] [--out outputs/augmented]

For each object: the 2 keyframes that show it best (box drawn in red + a crop of each), the reviewed label as a hint;
the VLM returns name, whether the label fits, colour, material, state, uses and one sentence.
Writes <out>/<scene>/objects_vlm.json  {symbol: {..., "views": [keyframe nodes], "raw": model text}}
and views/<symbol>.jpg (the images given to the model) for checking.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw

from vlmbaseline.bench.run import generate, load
from vlmbaseline.scene import load_export
from vlmbaseline.scene.views import best_views

# visible states to choose from, per label (others: DEFAULT_STATES)
STATES = {
    ("potted plant", "plant"): ["건강함", "잎이 시들거나 누렇게 변함", "알 수 없음"],
    ("chair", "armchair", "office chair", "bench"): ["비어 있음", "사람이 앉아 있음", "물건이 놓여 있음", "알 수 없음"],
    ("desk", "table", "coffee table"): ["비어 있음", "물건이 정돈되어 놓여 있음", "물건이 어지럽게 놓여 있음",
                                        "사람이 사용 중", "알 수 없음"],
    ("trash can",): ["비어 있음", "여유 있음", "거의 가득 참", "넘침", "알 수 없음"],
    ("lamp", "ceiling light"): ["켜짐", "꺼짐", "알 수 없음"],
    ("monitor", "computer", "screen", "blank monitor"): ["화면 켜짐", "화면 꺼짐", "알 수 없음"],
    ("door", "glass door"): ["열림", "닫힘", "알 수 없음"],
    ("bookshelf", "shelf"): ["정돈됨", "어지러움", "일부 비어 있음", "알 수 없음"],
    ("box", "cabinet", "filing cabinet", "locker", "drawer"): ["닫힘", "열림", "위에 물건이 쌓여 있음", "알 수 없음"],
    ("whiteboard",): ["깨끗함", "글씨나 그림이 있음", "알 수 없음"],
}
DEFAULT_STATES = ["정상으로 보임", "이상이 보임", "알 수 없음"]


def states_for(label):
    return next((v for k, v in STATES.items() if label in k), DEFAULT_STATES)


PROMPT = """사진들은 같은 물체를 다른 시점에서 본 것입니다. 빨간 상자 안의 물체를 보세요(뒤의 사진들은 확대).
이 물체는 3D 씬그래프에서 '{label}'(으)로 분류되어 있습니다. 아래 JSON 형식으로만 답하고 값은 한국어로 쓰세요.
{{"이름": "", "라벨_맞음": true, "색": "", "재질": "", "상태": "", "점검_필요": false, "점검_이유": "", "용도": ["", ""], "설명": "한 문장"}}
- 이름: 한국어 일반 명사(영어 금지)
- 라벨_맞음: 상자 안 물체가 '{label}'이 맞으면 true, 아니면 false
- 상태: 다음 중 사진에서 확인되는 하나만 고르세요: {states}
- 점검_필요: 정리·청소·수리·교체가 필요해 보이면 true, 이유를 점검_이유에 짧게. 아니면 false, 점검_이유는 ""
- 사진에 보이지 않는 내용은 지어내지 마세요."""


def views_for(scene, o, cache, k=2):
    v = best_views(scene, o, cache, k=k, strict=True) or best_views(scene, o, cache, k=k, strict=False)
    fulls, crops = [], []
    for _, kf, (x0, y0, x1, y1) in v:
        im = Image.open(kf.rgb).convert("RGB")
        f = im.copy()
        ImageDraw.Draw(f).rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        m = 0.25 * max(x1 - x0, y1 - y0) + 10
        crops.append(im.crop((max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, im.width), min(y1 + m, im.height))))
        fulls.append(f)
    return v, fulls, crops


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--out", default="outputs/augmented")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    proc, model = load(a.model, a.quant)
    for e in a.exports:
        scene, name = load_export(e), Path(e).name
        out = Path(a.out) / name
        (out / "views").mkdir(parents=True, exist_ok=True)
        res, cache, t0 = {}, {}, time.perf_counter()
        objs = scene.objects[:a.limit] if a.limit else scene.objects
        for o in objs:
            v, fulls, crops = views_for(scene, o, cache)
            if not v:
                res[o.symbol] = {"label": o.label, "views": [], "error": "no view"}
                print(f"{name} {o.symbol} {o.label}: no view", flush=True)
                continue
            sheet = Image.new("RGB", (sum(i.width for i in fulls + crops), max(i.height for i in fulls + crops)))
            x = 0
            for i in fulls + crops:
                sheet.paste(i, (x, 0))
                x += i.width
            sheet.thumbnail((1600, 480))
            sheet.save(out / "views" / f"{o.symbol}.jpg", quality=85)
            text, dt, _, _ = generate(proc, model, fulls + crops, PROMPT.format(label=o.label, states=", ".join(states_for(o.label))), 260)
            j = parse(text)
            res[o.symbol] = dict(j or {}, label=o.label, views=[kf.node for _, kf, _ in v], raw=text,
                                 ok=j is not None, sec=round(dt, 2))
            print(f"{name} {o.symbol} {o.label}: {json.dumps(j, ensure_ascii=False)[:150] if j else text[:80]!r} "
                  f"({dt:.1f}s)", flush=True)
        (out / "objects_vlm.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        n_ok = sum(r.get("ok", False) for r in res.values())
        print(f"== {name}: {n_ok}/{len(res)} described, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
