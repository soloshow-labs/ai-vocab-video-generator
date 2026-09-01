import os
import stat
from pathlib import Path

import pytest

from ai_vocab_video_generator.errors import ProviderError
from ai_vocab_video_generator.providers.tts import EdgeSpeechProvider


class FakeCommunicator:
    def __init__(self, text: str, voice: str, rate: str, volume: str) -> None:
        self.payload = f"{text}|{voice}|{rate}|{volume}".encode()

    async def save(self, destination: str) -> None:
        Path(destination).write_bytes(self.payload)


def test_edge_speech_provider_saves_audio(tmp_path: Path) -> None:
    provider = EdgeSpeechProvider(communicator_factory=FakeCommunicator)
    destination = tmp_path / "apple.mp3"

    result = provider.synthesize(
        "apple",
        destination,
        voice="en-US-JennyNeural",
        rate="-25%",
        volume="+10%",
    )

    assert result == destination
    assert destination.read_bytes() == b"apple|en-US-JennyNeural|-25%|+10%"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_edge_speech_provider_marks_audio_owner_only(tmp_path: Path) -> None:
    destination = tmp_path / "private" / "apple.mp3"

    EdgeSpeechProvider(communicator_factory=FakeCommunicator).synthesize(
        "apple", destination, voice="en-US-JennyNeural", rate="+0%"
    )

    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


class OversizedCommunicator(FakeCommunicator):
    async def save(self, destination: str) -> None:
        with Path(destination).open("wb") as output:
            output.truncate(32 * 1024 * 1024 + 1)


def test_edge_speech_provider_rejects_oversized_output(tmp_path: Path) -> None:
    destination = tmp_path / "oversized.mp3"

    with pytest.raises(ProviderError, match="Speech generation failed"):
        EdgeSpeechProvider(communicator_factory=OversizedCommunicator).synthesize(
            "apple", destination, voice="en-US-JennyNeural", rate="+0%"
        )

    assert not destination.exists()
