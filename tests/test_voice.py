"""Voice input helpers (no model): error rate, the page's resampling, speech text cleanup."""
import numpy as np

from vlmbaseline.demo.bench_voice import cer, norm, page_16k
from vlmbaseline.demo.tts import Speaker, read_number


def test_cer():
    assert cer("여기서 원형 복도까지 어떻게 가?", "여기서 원형 복도까지 어떻게 가.") == 0
    assert norm("내 왼쪽에 뭐가 있어?") == "내왼쪽에뭐가있어"
    assert abs(cer("원형복도", "원형보") - 0.5) < 1e-9  # one substitution + one deletion of 4


def test_page_16k():
    x = np.arange(48000, dtype=np.float32)
    y = page_16k(x, 48000)
    assert len(y) == 16000 and y[1] == 3


def test_speech_text():
    assert read_number("26") == "이십육" and read_number("1.5") == "일 점 오"
    assert Speaker.clean("화이트보드(O378)까지 26 m입니다. 3번째 문") == "화이트보드까지 이십육 미터입니다. 세 번째 문"
