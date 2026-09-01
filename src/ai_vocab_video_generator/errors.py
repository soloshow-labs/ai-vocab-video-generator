"""Safe application errors suitable for user-facing surfaces."""

from typing import Literal

from ai_vocab_video_generator.media_limits import MIB


class UploadSizeError(ValueError):
    """A local file exceeds its type's byte budget, without exposing its name."""

    def __init__(self, size_bytes: int, limit_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"The selected upload exceeds the size limit: "
            f"{size_bytes / MIB:.2f} MiB; maximum {limit_bytes / MIB:g} MiB."
        )


class ApplicationError(RuntimeError):
    def __init__(self, safe_message: str, *, diagnostic: str = "") -> None:
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.diagnostic = diagnostic


class ConfigurationError(ApplicationError):
    """A required local setting or credential is missing."""


class ProviderError(ApplicationError):
    """A third-party provider failed without exposing its credential."""


SpeechFailureReason = Literal[
    "connection",
    "timeout",
    "service",
    "rejected",
    "empty",
    "settings",
    "file",
    "output",
    "certificate",
    "unknown",
]
SpeechTrack = Literal["question", "zh", "fast", "slow"]

_SPEECH_REASONS = {
    "connection": "Could not connect to Edge TTS. Check your network and try again.",
    "timeout": "Edge TTS timed out. Check your network and try again.",
    "service": "Edge TTS is temporarily unavailable or rate-limited. Try again later.",
    "rejected": "Edge TTS rejected the request. Check the voice and service availability.",
    "empty": "Edge TTS returned no audio. Check the voice and text, then try again.",
    "settings": "Check the voice, rate, volume, and text settings.",
    "file": "Could not save the audio file. Check available disk space and storage permissions.",
    "output": "The audio output is missing, empty, or exceeds the size limit.",
    "certificate": "The secure connection to Edge TTS failed. Check your network and certificates.",
    "unknown": "Check your network and narration settings, then try again.",
}
_SPEECH_TRACKS = {
    "question": "question narration",
    "zh": "Chinese narration",
    "fast": "English narration",
    "slow": "slow English narration",
}


class SpeechGenerationError(ProviderError):
    """Safe speech failure metadata; never includes input text, paths, or URLs."""

    def __init__(
        self,
        reason: SpeechFailureReason,
        *,
        attempts: int,
        diagnostic: str = "",
        track: SpeechTrack | None = None,
        word_index: int | None = None,
    ) -> None:
        self.reason = reason
        self.attempts = attempts
        self.track = track
        self.word_index = word_index
        context = ""
        if track is not None:
            context = (
                f" for word {word_index + 1} ({_SPEECH_TRACKS[track]})"
                if word_index is not None
                else f" during {_SPEECH_TRACKS[track]}"
            )
        super().__init__(
            f"Speech generation failed{context}. {_SPEECH_REASONS[reason]} Attempts: {attempts}.",
            diagnostic=diagnostic,
        )

    def with_context(
        self, track: SpeechTrack, word_index: int | None = None
    ) -> "SpeechGenerationError":
        return SpeechGenerationError(
            self.reason,
            attempts=self.attempts,
            diagnostic=self.diagnostic,
            track=track,
            word_index=word_index,
        )


class RenderingError(ApplicationError):
    """A card or video could not be rendered."""


class JobBusyError(ApplicationError):
    """A generation or regeneration already owns the per-job lock."""
