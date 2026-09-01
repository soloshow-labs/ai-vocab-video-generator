"""Text-to-speech provider backed by Microsoft Edge voices."""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from pathlib import Path
from time import sleep
from typing import Any, Protocol

import aiohttp
import edge_tts
from edge_tts.exceptions import NoAudioReceived, WebSocketError

from ai_vocab_video_generator.errors import SpeechFailureReason, SpeechGenerationError
from ai_vocab_video_generator.private_fs import ensure_private_directory, mark_private_file

_MAX_SPEECH_BYTES = 32 * 1024 * 1024
_MAX_ATTEMPTS = 3


def _failure_reason(exc: Exception) -> tuple[SpeechFailureReason, bool]:
    if isinstance(exc, aiohttp.ClientSSLError):
        return "certificate", False
    if isinstance(exc, TimeoutError):
        return "timeout", True
    if isinstance(exc, (aiohttp.ClientConnectionError, ConnectionError, WebSocketError)):
        return "connection", True
    if isinstance(exc, aiohttp.ClientResponseError):
        transient = exc.status == 429 or 500 <= exc.status < 600
        return ("service" if transient else "rejected"), transient
    if isinstance(exc, NoAudioReceived):
        return "empty", True
    if isinstance(exc, (ValueError, TypeError)):
        return "settings", False
    if isinstance(exc, OSError):
        return "file", False
    return "unknown", False


def _discard_audio(destination: Path) -> None:
    with contextlib.suppress(OSError):
        destination.unlink(missing_ok=True)


class _Communicator(Protocol):
    def save(self, destination: str) -> Coroutine[Any, Any, None]: ...


CommunicatorFactory = Callable[[str, str, str, str], _Communicator]


def _edge_communicator(text: str, voice: str, rate: str, volume: str) -> _Communicator:
    return edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)


class EdgeSpeechProvider:
    def __init__(self, communicator_factory: CommunicatorFactory = _edge_communicator) -> None:
        self._communicator_factory = communicator_factory

    def synthesize(
        self,
        text: str,
        destination: Path,
        *,
        voice: str,
        rate: str,
        volume: str = "+0%",
    ) -> Path:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                ensure_private_directory(destination.parent)
                # Edge communicators are single-use: each retry needs a fresh one.
                communicator = self._communicator_factory(text, voice, rate, volume)
                asyncio.run(communicator.save(str(destination)))
                if (
                    not destination.is_file()
                    or not 0 < destination.stat().st_size <= _MAX_SPEECH_BYTES
                ):
                    raise SpeechGenerationError("output", attempts=attempt)
                mark_private_file(destination)
                return destination
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                _discard_audio(destination)
                raise
            except SpeechGenerationError:
                _discard_audio(destination)
                raise
            except Exception as exc:
                _discard_audio(destination)
                reason, retryable = _failure_reason(exc)
                if not retryable or attempt == _MAX_ATTEMPTS:
                    raise SpeechGenerationError(
                        reason, attempts=attempt, diagnostic=type(exc).__name__
                    ) from exc
                sleep(2 ** (attempt - 1))
        raise AssertionError("Unreachable speech retry state")
