import hashlib
import math
import struct
import wave
from pathlib import Path

from ai_vocab_video_generator.rendering.audio import create_countdown_wav


def _rms(samples: tuple[int, ...]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def test_countdown_preserves_the_original_project_audio(tmp_path: Path) -> None:
    path = create_countdown_wav(tmp_path / "countdown.wav")

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "adfc9c570670fee6eeef35309035c12a1d9facc925dad5f8b2f464baff7dfbee"
    )

    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 2
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 22_050
        duration = audio.getnframes() / audio.getframerate()
        frames = audio.readframes(audio.getnframes())

    assert 1.45 <= duration <= 1.50
    interleaved = struct.unpack(f"<{len(frames) // 2}h", frames)
    left = interleaved[::2]
    sample_rate = 22_050
    middle = left[round(0.60 * sample_rate) : round(0.80 * sample_rate)]
    tail = left[round(1.20 * sample_rate) : round(1.40 * sample_rate)]

    assert _rms(middle) > 150
    assert _rms(tail) > 30
    assert _rms(left[: round(0.30 * sample_rate)]) > _rms(tail)
