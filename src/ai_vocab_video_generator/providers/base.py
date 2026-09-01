"""Protocols implemented by external service adapters."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ai_vocab_video_generator.domain import MaterialAsset, SelectionMode, VideoAspect, WordEntry


@dataclass(frozen=True, slots=True)
class ImageSelectionContext:
    entry_index: int
    pool_size: int
    mode: SelectionMode
    seed: int

    def __post_init__(self) -> None:
        if self.entry_index < 0:
            raise ValueError("Entry index must not be negative.")
        if not 1 <= self.pool_size <= 20:
            raise ValueError("Image candidate pool size must be between 1 and 20.")


class VocabularyProvider(Protocol):
    def check_connection(self) -> None: ...

    def generate(self, topic: str, count: int) -> list[WordEntry]: ...

    def complete_phonetics(self, entries: Sequence[WordEntry]) -> list[WordEntry]: ...


class ImageProvider(Protocol):
    def fetch(
        self,
        query: str,
        destination_stem: Path,
        aspect: VideoAspect,
        context: ImageSelectionContext | None = None,
    ) -> MaterialAsset: ...


class SpeechProvider(Protocol):
    def synthesize(
        self,
        text: str,
        destination: Path,
        *,
        voice: str,
        rate: str,
        volume: str = "+0%",
    ) -> Path: ...


class TranscriptionProvider(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...
