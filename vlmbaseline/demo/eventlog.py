"""What happened in a demo session, one JSON line per event, for looking at a walk afterwards.

    outputs/logs/demo_<start>.jsonl     (outputs/ is not tracked: the questions are the user's words)
    python -m vlmbaseline.demo.eventlog [file]     # the session as a timeline (default: the latest)

Events: start (settings), ask (question, pose, tools called with their arguments, answer, seconds; ask_error),
stt (recognized text, audio length, seconds), say (an announcement queued for the page, with why: follow / hold /
resume), follow_start / follow_stop / arrived, live (the position state changed: localized, uncertain and why, or the
signal came back after a gap; with the pose), page_say (said by the page itself: searching, signal lost, back).
With --keep-audio the spoken questions are kept next to the log (stt_<n>.wav) to check recognition on real voices.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("outputs/logs")


class EventLog:
    def __init__(self, log_dir=LOG_DIR, keep_audio=False):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"demo_{stamp}.jsonl"
        self.audio_dir = self.dir / f"demo_{stamp}_audio" if keep_audio else None
        self.lock = threading.Lock()
        self.n_audio = 0

    def add(self, kind, **kw):
        rec = {"t": datetime.now().isoformat(timespec="milliseconds"), "kind": kind, **kw}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self.lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
        return rec

    def audio(self, wav_bytes):
        """Keep a spoken question (only with keep_audio); returns its file name or None."""
        if self.audio_dir is None:
            return None
        with self.lock:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            self.n_audio += 1
            p = self.audio_dir / f"stt_{self.n_audio:03d}.wav"
        p.write_bytes(wav_bytes)
        return p.name


def wav_seconds(wav_bytes):
    try:
        import io

        import soundfile as sf
        return round(sf.info(io.BytesIO(wav_bytes)).duration, 1)
    except Exception:
        return None


def compact_trace(trace, limit=300):
    """Tools called, with their arguments and, cut short, what came back (errors in full)."""
    out = []
    for tr in trace or []:
        r = tr.get("result")
        e = {"tool": tr.get("tool"), "args": tr.get("args")}
        if isinstance(r, dict) and "error" in r:
            e["error"] = r["error"]
        else:
            s = json.dumps(r, ensure_ascii=False, default=str)
            e["result"] = s if len(s) <= limit else s[:limit] + "…"
        out.append(e)
    return out


def show(path):
    """The session as lines: time, kind, the words."""
    t0 = None
    for line in open(path):
        e = json.loads(line)
        t = datetime.fromisoformat(e["t"])
        t0 = t0 or t
        s = (t - t0).total_seconds()
        k = e["kind"]
        if k == "ask":
            tools = ", ".join(x["tool"] for x in e.get("tools", []))
            ans = " / ".join(x.strip() for x in str(e.get("answer", "")).splitlines() if x.strip())
            txt = f"Q {e['q']}  [{tools}] {e.get('sec')} s\n{'':23}A {ans}"
        elif k == "stt":
            txt = f"말: {e.get('text')!r} ({e.get('audio_s')} s 음성, {e.get('sec')} s)"
        elif k in ("say", "page_say"):
            txt = f"{'' if k == 'say' else '(페이지) '}{e.get('text')}" + (f"  <{e['why']}>" if e.get("why") else "")
        elif k == "live":
            txt = f"위치 {e.get('state')}" + (f" {e.get('pose')}" if e.get("pose") else "") + \
                  (f" (신호 {e['gap_s']} s 끊김 뒤)" if e.get("gap_s") else "")
        else:
            txt = json.dumps({x: v for x, v in e.items() if x not in ("t", "kind")}, ensure_ascii=False)
        print(f"{int(s // 60):3d}:{s % 60:05.2f}  {k:12s} {txt}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(LOG_DIR.glob("demo_*.jsonl"), key=lambda p: p.stat().st_mtime)
    print(path)
    show(path)


if __name__ == "__main__":
    main()
