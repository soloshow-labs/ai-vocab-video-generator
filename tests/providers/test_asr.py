import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_vocab_video_generator.errors import ProviderError
from ai_vocab_video_generator.providers.asr import FunASRTranscriptionProvider


def test_funasr_rejects_missing_audio_before_loading_optional_model(tmp_path: Path) -> None:
    provider = FunASRTranscriptionProvider(model_cache=tmp_path / "models")

    with pytest.raises(ProviderError, match="does not exist"):
        provider.transcribe(tmp_path / "missing.wav")

    assert not (tmp_path / "models").exists()


def test_funasr_creates_a_private_model_cache(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "models"
    monkeypatch.setattr(
        "ai_vocab_video_generator.providers.asr.importlib.import_module",
        lambda _name: SimpleNamespace(AutoModel=lambda **_kwargs: object()),
    )

    FunASRTranscriptionProvider(model_cache=cache)._load_model()

    assert cache.is_dir()
    if os.name == "posix":
        assert stat.S_IMODE(cache.stat().st_mode) == 0o700
