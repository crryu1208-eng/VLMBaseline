"""Korean speech for the guidance, made on this computer (no network once the model is downloaded).

Supertonic (Supertone; code MIT, model weights OpenRAIL-M), ONNX on the CPU: ~1.5 s for one guidance sentence.
The model (~390 MB) is downloaded once from Hugging Face into ~/.cache/supertonic3 and loaded from there afterwards
without network. Install: uv pip install supertonic onnxruntime  (pip install -e .[tts]).
Sentences are cached (the same guidance phrases come back often), so a repeated one is played at once.
"""
from __future__ import annotations

import io
import re
import threading
from collections import OrderedDict

VOICE = "F1"
CACHE = 300  # sentences kept as wav
_DIG = "영일이삼사오육칠팔구"
_COUNT = ["영", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]  # '3개' -> '세 개'
_NATIVE = (_COUNT, ["영", "첫", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"])  # '1번째' -> '첫 번째'


def read_number(s):
    """'12' -> '십이', '1.5' -> '일 점 오', '2026' -> '이천이십육' (Sino-Korean, as distances are read)."""
    whole, _, frac = s.partition(".")
    n = int(whole)
    if n == 0:
        out = "영"
    elif n >= 100_000_000:
        out = " ".join(_DIG[int(c)] for c in whole)
    else:
        out = ""
        for big, name in ((10_000, "만"), (1, "")):
            k, n = divmod(n, big) if big > 1 else (n, 0)
            if not k:
                continue
            part = ""
            for unit, uname in ((1000, "천"), (100, "백"), (10, "십"), (1, "")):
                d, k = divmod(k, unit)
                if d:
                    part += ("" if d == 1 and uname else _DIG[d]) + uname
            out += ("" if part == "일" and name else part) + name
    if frac:
        out += " 점 " + "".join(_DIG[int(c)] for c in frac)
    return out


class Speaker:
    def __init__(self, voice=VOICE, speed=1.1):
        self.voice, self.speed = voice, speed
        self.tts = self.style = None
        self.error = None
        self.lock = threading.Lock()  # one synthesis at a time (ONNX runs on all cores)
        self.cache = OrderedDict()
        threading.Thread(target=self._load, daemon=True).start()  # the server starts without waiting

    def _load(self):
        try:
            from supertonic import TTS
            try:
                tts = TTS(auto_download=False)  # already downloaded: no network at all
            except Exception:
                tts = TTS()  # first run: download the model
            self.style = tts.get_voice_style(self.voice)
            self.tts = tts
            print("speech ready", flush=True)
        except Exception as e:  # the page falls back to the browser's voice
            self.error = f"{type(e).__name__}: {e}"
            print("speech not loaded:", self.error, flush=True)

    @property
    def ready(self):
        return self.tts is not None

    @staticmethod
    def clean(text):
        text = re.sub(r"\s*\(O\d+\)", "", text)  # object symbols are for the screen
        text = re.sub(r"(\d+(?:\.\d+)?)\s*m(?![A-Za-z])", r"\1 미터", text)  # '12 m' would be read as the letter
        text = re.sub(r"\b(\d)\s*(번째|개|명|곳)", lambda m: _NATIVE[m.group(2) == "번째"][int(m.group(1))] + " " + m.group(2), text)
        text = re.sub(r"\d+(?:\.\d+)?", lambda m: read_number(m.group()), text)
        return re.sub(r"\s+", " ", text).strip()[:300]

    def wav(self, text):
        """16-bit mono wav bytes of the sentence, or None when speech is not loaded."""
        text = self.clean(text)
        if not self.ready or not text:
            return None
        with self.lock:
            if text in self.cache:
                self.cache.move_to_end(text)
                return self.cache[text]
            import numpy as np
            import soundfile as sf
            w, _ = self.tts.synthesize(text, voice_style=self.style, lang="ko", speed=self.speed)
            buf = io.BytesIO()
            sf.write(buf, np.asarray(w).reshape(-1), self.tts.sample_rate, format="WAV", subtype="PCM_16")
            self.cache[text] = buf.getvalue()
            if len(self.cache) > CACHE:
                self.cache.popitem(last=False)
            return self.cache[text]
