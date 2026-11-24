"""Voice input benchmark: the benchmark questions spoken by the TTS voices, through the page's conversion, recognized.

    python -m vlmbaseline.demo.bench_voice [-o outputs/bench_voice.json] [--voices F1 M1 ...] [--snr 10]

Each question (bench_pose / bench_areas / bench_dialog) is spoken by every Supertonic voice, resampled to 48 kHz
(a microphone) and then to 16 kHz the way index.html does (nearest sample), optionally with white noise at --snr dB,
and recognized by Whisper (stt.Listener). Character error rate after removing spaces and punctuation, exact
questions, and time per question. Synthetic speech is cleaner than a person in a corridor: an upper bound.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import numpy as np

SOURCES = ["outputs/bench_pose.json", "outputs/bench_areas.json", "outputs/bench_dialog.json"]


def questions():
    qs = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("q", "question") and isinstance(v, str):
                    qs.add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for f in SOURCES:
        if Path(f).exists():
            walk(json.load(open(f)))
    return sorted(qs)


def norm(t):
    return re.sub(r"[\s.,?!~'\"()]", "", t)


def cer(ref, hyp):
    a, b = norm(ref), norm(hyp)
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(1, len(a))


def page_16k(x, rate):
    """index.html wav16k: nearest sample, no filter."""
    m = int(len(x) * 16000 / rate)
    return x[np.minimum(len(x) - 1, (np.arange(m) * rate / 16000).astype(int))]


def wav_bytes(x, rate):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="outputs/bench_voice.json")
    ap.add_argument("--voices", nargs="+", default=["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--snr", type=float, nargs="+", default=[None, 10.0], help="dB of white noise; omit for clean")
    ap.add_argument("--hint", action="store_true", help="give Whisper the area and object names of the scenes")
    a = ap.parse_args()
    from supertonic import TTS

    from vlmbaseline.demo.stt import Listener
    from vlmbaseline.demo.tts import Speaker
    tts = TTS(auto_download=False)
    lis = Listener()
    while not lis.ready and not lis.error:
        time.sleep(0.5)
    if lis.error:
        raise SystemExit(lis.error)
    rng = np.random.default_rng(0)
    qs = questions()
    hint = None
    if a.hint:
        from vlmbaseline.demo.stt import scene_hint
        from vlmbaseline.scene import attach_areas, attach_vlm, load_export
        from vlmbaseline.spatial.tools import SpatialIndex
        parts = []
        for sc in ("lab_03", "library_merged"):
            s = load_export(Path("data") / sc)
            attach_vlm(s, Path("outputs/augmented") / sc / "objects_vlm.json")
            attach_areas(s, Path("outputs/augmented") / sc / "areas_vlm.json")
            parts.append(scene_hint(SpatialIndex(s)))
        hint = ", ".join(parts)
    rows = []
    for v in a.voices:
        style = tts.get_voice_style(v)
        for q in qs:
            w, _ = tts.synthesize(Speaker.clean(q), voice_style=style, lang="ko")
            w = np.asarray(w, np.float32).reshape(-1)
            sr = tts.sample_rate
            mic = np.interp(np.linspace(0, len(w) - 1, int(len(w) * 48000 / sr)), np.arange(len(w)), w)
            mic = np.concatenate([np.zeros(4800), mic, np.zeros(9600)]).astype(np.float32)  # quiet before / after
            for snr in a.snr:
                x = mic
                if snr is not None:
                    p = float(np.mean(mic[mic != 0] ** 2))
                    x = mic + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(mic)).astype(np.float32)
                t0 = time.time()
                hyp = lis.text(wav_bytes(page_16k(x, 48000), 16000), hint)
                rows.append({"voice": v, "snr": snr, "q": q, "text": hyp, "cer": round(cer(q, hyp), 3),
                             "exact": norm(q) == norm(hyp), "s": round(time.time() - t0, 2)})
    out = {}
    for snr in a.snr:
        R = [r for r in rows if r["snr"] == snr]
        c = np.array([r["cer"] for r in R])
        out["clean" if snr is None else f"snr{snr:g}dB"] = {
            "clips": len(R), "cer_mean": round(float(c.mean()), 3), "cer_90": round(float(np.percentile(c, 90)), 3),
            "exact": f"{sum(r['exact'] for r in R)}/{len(R)}",
            "s_median": round(float(np.median([r["s"] for r in R])), 2)}
    print(json.dumps(out, ensure_ascii=False))
    Path(a.out).write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
