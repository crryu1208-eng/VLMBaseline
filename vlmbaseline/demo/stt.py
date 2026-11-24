"""Korean questions by voice, recognized on this computer (no network once the model is downloaded).

Whisper large-v3-turbo (OpenAI, MIT; transformers, already a dependency of the model), fp16 on the GPU: ~1.8 GB,
~0.3 s for a short question. The model is downloaded once from Hugging Face and loaded offline afterwards.
The page sends 16 kHz mono 16-bit wav (recorded and converted in the browser, no ffmpeg needed).
"""
from __future__ import annotations

import io
import threading

MODEL = "openai/whisper-large-v3-turbo"
MAX_S = 15
HINT_TOKENS = 150  # the hint's first words (area names come first)


class Listener:
    def __init__(self, model=MODEL, device="cuda:0"):
        self.model_id, self.device = model, device
        self.asr = None
        self.error = None
        self.lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import torch
            from transformers import pipeline
            dev = self.device if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if dev != "cpu" else torch.float32
            self.asr = pipeline("automatic-speech-recognition", model=self.model_id, dtype=dtype, device=dev)
            print("voice input ready", flush=True)
        except Exception as e:  # the page hides the button
            self.error = f"{type(e).__name__}: {e}"
            print("voice input not loaded:", self.error, flush=True)

    @property
    def ready(self):
        return self.asr is not None

    def text(self, wav_bytes, hint=None):
        """Korean text of a wav (any rate; resampled to 16 kHz), or None when not loaded. hint: words of the scene
        (area and object names) given to Whisper as the previous text, so they are spelled as in the map."""
        if not self.ready:
            return None
        import numpy as np
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        x = x.mean(axis=1)[: int(MAX_S * sr)]
        if sr != 16000:
            n = int(len(x) * 16000 / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        if len(x) < 1600 or float(np.abs(x).max()) < 1e-3:  # under 0.1 s or silence: Whisper would make text up
            return ""
        kw = {"language": "korean", "task": "transcribe"}
        if hint:
            ids = self.asr.tokenizer.get_prompt_ids(hint, return_tensors="pt")
            kw["prompt_ids"] = ids[: 1 + HINT_TOKENS].to(self.asr.device)  # Whisper: prompt + answer under 448
        with self.lock:
            r = self.asr({"raw": x, "sampling_rate": 16000}, generate_kwargs=kw)
        t = r["text"].strip()
        if hint and t.startswith(hint.strip()):  # some versions return the prompt too
            t = t[len(hint.strip()):].strip()
        return t


def scene_hint(idx, limit=400):
    """Area and object names of a scene (SpatialIndex), as one line for Whisper."""
    from vlmbaseline.spatial.names import KO
    s = idx.s
    areas = s.areas.values() if isinstance(s.areas, dict) else s.areas
    words = sorted({a.get("이름") for a in areas if a.get("이름")})
    words += sorted({KO.get(o.label, o.label) for o in s.objects} - set(words))
    return ", ".join(words)[:limit]
