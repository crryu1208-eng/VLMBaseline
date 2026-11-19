"""Run one VLM on the benchmark set (built by build_set.py).

    python -m vlmbaseline.bench.run Qwen/Qwen3.5-9B --set outputs/bench_set --out outputs/bench_runs [--quant 4bit]

Tasks
  recog     image with the object in a red box -> short English name           (scored against the reviewed label)
  describe  same image + crop -> Korean JSON: name, colour, material, state, uses (checked for valid JSON / Korean)
  scene_qa  object table of the scene (text) + Korean question                  (scored against the graph)
Writes <out>/<model>/{recog,describe,scene_qa}.jsonl and run.json (load time, peak VRAM, latency).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

RECOG = ("What is the object inside the red box? Answer with a short English noun phrase only "
         "(for example: office chair). No sentence.")
DESCRIBE = ("첫 번째 사진의 빨간 상자 안 물체(두 번째 사진은 확대)를 설명하세요. 아래 JSON 형식으로만 답하고, 값은 한국어로 쓰세요.\n"
            '{"이름": "", "색": "", "재질": "", "상태": "", "용도": ["", ""]}')
SCENE = ("다음은 실내 공간의 3D 씬그래프 물체 목록입니다. 좌표 단위는 m이고 바닥이 z = 0입니다.\n\n{table}\n\n"
         "질문: {q}\n답만 짧게 쓰세요.")


def load(model_id, quant):
    q = None
    if quant == "4bit":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    elif quant == "8bit":
        q = BitsAndBytesConfig(load_in_8bit=True)
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, quantization_config=q,
                                                        device_map="cuda")
    model.eval()
    return proc, model


def generate(proc, model, images, text, max_new_tokens):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        inputs = proc.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        inputs = proc.apply_chat_template(msgs, **kw)
    inputs = inputs.to(model.device)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    n_in = inputs["input_ids"].shape[1]
    ans = proc.decode(out[0, n_in:], skip_special_tokens=True).strip()
    return ans, dt, n_in, out.shape[1] - n_in


def boxed(set_dir, r):
    full = Image.open(set_dir / "img" / r["full"]).convert("RGB")
    crop = Image.open(set_dir / "img" / r["crop"]).convert("RGB")
    return full, crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--set", default="outputs/bench_set")
    ap.add_argument("--out", default="outputs/bench_runs")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--tasks", default="recog,describe,scene_qa")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    set_dir = Path(a.set)
    out = Path(a.out) / a.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    excluded = {e["key"] for e in json.loads((set_dir / "excluded.json").read_text())}
    objs = [json.loads(l) for l in (set_dir / "objects.jsonl").read_text().splitlines()]
    objs = [r for r in objs if r["key"] not in excluded]
    qa = [json.loads(l) for l in (set_dir / "scene_qa.jsonl").read_text().splitlines()]
    if a.limit:
        objs, qa = objs[:a.limit], qa[:a.limit]

    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    proc, model = load(a.model, a.quant)
    load_s = time.perf_counter() - t
    vram_load = torch.cuda.max_memory_allocated() / 2**30
    stats = dict(model=a.model, quant=a.quant, load_s=round(load_s, 1), vram_load_gb=round(vram_load, 2))

    tasks = a.tasks.split(",")
    for task in tasks:
        rows, times = [], []
        items = qa if task == "scene_qa" else objs
        for r in items:
            if task == "recog":
                full, _ = boxed(set_dir, r)
                ans, dt, n_in, n_out = generate(proc, model, [full], RECOG, 16)
            elif task == "describe":
                full, crop = boxed(set_dir, r)
                ans, dt, n_in, n_out = generate(proc, model, [full, crop], DESCRIBE, 220)
            else:
                table = (set_dir / f"graph_{r['scene']}.txt").read_text()
                ans, dt, n_in, n_out = generate(proc, model, [], SCENE.format(table=table, q=r["q"]), 24)
            times.append(dt)
            rows.append(dict(r, answer=ans, sec=round(dt, 3), n_in=n_in, n_out=n_out))
            print(f"[{task}] {r.get('key', r.get('id'))}: {ans[:80]!r} ({dt:.2f}s)", flush=True)
        with open(out / f"{task}.jsonl", "w") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        stats[f"{task}_sec_median"] = round(sorted(times)[len(times) // 2], 3) if times else None
    stats["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "run.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
