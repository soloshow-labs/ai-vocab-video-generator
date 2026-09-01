"""Optional local speech transcription through FunASR."""

import importlib
from pathlib import Path
from typing import Any

from ai_vocab_video_generator.errors import ConfigurationError, ProviderError
from ai_vocab_video_generator.private_fs import ensure_private_directory


class FunASRTranscriptionProvider:
    def __init__(
        self,
        *,
        model_id: str = "iic/SenseVoiceSmall",
        model_cache: Path = Path("model_cache"),
    ) -> None:
        self._model_id = model_id
        self._model_cache = model_cache
        self._model: Any = None

    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.is_file():
            raise ProviderError("The recorded audio file does not exist.")
        if self._model is None:
            self._model = self._load_model()
        try:
            results = self._model.generate(input=str(audio_path), cache={}, language="auto")
            return str(results[0].get("text", "")).strip()
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ProviderError(
                "Speech transcription returned an invalid result.",
                diagnostic=type(exc).__name__,
            ) from exc

    def _load_model(self) -> Any:
        try:
            funasr = importlib.import_module("funasr")
        except ImportError as exc:
            raise ConfigurationError(
                "Voice input requires the optional ASR dependencies. Run 'uv sync --extra asr'."
            ) from exc
        ensure_private_directory(self._model_cache)
        return funasr.AutoModel(
            model=self._model_id,
            cache_dir=str(self._model_cache),
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cpu",
        )
