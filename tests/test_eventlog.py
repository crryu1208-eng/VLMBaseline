"""The demo session log (demo/eventlog.py)."""
import json

from vlmbaseline.demo.eventlog import EventLog, compact_trace, show


def test_log_lines_and_audio(tmp_path, capsys):
    log = EventLog(tmp_path, keep_audio=True)
    log.add("ask", q="여기서 엘리베이터 앞까지 어떻게 가?", pose=(1.0, 2.0, 0.5), answer="직진 후 왼쪽", tools=[], sec=3.1)
    log.add("say", text="약 5 m 앞에서 왼쪽으로 도세요.", why="follow")
    assert log.audio(b"RIFF....") == "stt_001.wav" and (log.audio_dir / "stt_001.wav").exists()
    rows = [json.loads(x) for x in open(log.path)]
    assert [r["kind"] for r in rows] == ["ask", "say"] and rows[0]["q"].startswith("여기서")
    show(log.path)
    assert "왼쪽으로 도세요" in capsys.readouterr().out


def test_no_audio_without_keep(tmp_path):
    assert EventLog(tmp_path).audio(b"x") is None


def test_compact_trace_cuts_results_keeps_errors():
    tr = compact_trace([{"tool": "find", "args": {"label": "chair"}, "result": {"objects": ["O1"] * 500}},
                        {"tool": "route", "args": {}, "result": {"error": "no path"}}], limit=50)
    assert len(tr[0]["result"]) == 51 and tr[1]["error"] == "no path"


def test_wav_seconds():
    import io

    import numpy as np
    import soundfile as sf

    from vlmbaseline.demo.eventlog import wav_seconds
    b = io.BytesIO()
    sf.write(b, np.zeros(44100 * 2, np.float32), 44100, format="WAV", subtype="PCM_16")
    assert wav_seconds(b.getvalue()) == 2.0 and wav_seconds(b"junk") is None
