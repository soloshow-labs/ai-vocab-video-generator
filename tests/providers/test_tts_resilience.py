import asyncio
import os
import stat
from pathlib import Path

import aiohttp
import pytest
from edge_tts.exceptions import NoAudioReceived, WebSocketError

import ai_vocab_video_generator.providers.tts as tts_module
from ai_vocab_video_generator.errors import ProviderError
from ai_vocab_video_generator.i18n import Locale
from ai_vocab_video_generator.providers.tts import EdgeSpeechProvider
from ai_vocab_video_generator.webui import _safe_message


def _connector_error():
    return aiohttp.ClientConnectorError(None, ConnectionResetError(54, "private detail"))


@pytest.mark.parametrize(
    "failure",
    [
        _connector_error(),
        TimeoutError("private timeout"),
        aiohttp.ServerDisconnectedError("private disconnect"),
        ConnectionResetError("private reset"),
        WebSocketError("private websocket failure"),
        aiohttp.ClientResponseError(None, (), status=429),
        aiohttp.ClientResponseError(None, (), status=503),
        NoAudioReceived("private service response"),
    ],
)
def test_transient_speech_errors_retry_with_fresh_communicator_and_clean_output(
    tmp_path, monkeypatch, failure
):
    delays = []
    monkeypatch.setattr(tts_module, "sleep", delays.append)
    instances = []
    destination = tmp_path / "speech.mp3"

    class FlakyCommunicator:
        def __init__(self, text, voice, rate, volume):
            assert (text, voice, rate, volume) == ("胡萝卜", "zh-CN-XiaoxiaoNeural", "+0%", "+0%")
            self.used = False
            instances.append(self)

        async def save(self, target):
            assert not self.used, "Edge communicators cannot be reused"
            self.used = True
            assert not Path(target).exists(), "Partial audio must be removed before retry"
            if len(instances) < 3:
                Path(target).write_bytes(b"partial")
                raise failure
            Path(target).write_bytes(b"complete-audio")

    result = EdgeSpeechProvider(FlakyCommunicator).synthesize(
        "胡萝卜", destination, voice="zh-CN-XiaoxiaoNeural", rate="+0%"
    )
    assert result.read_bytes() == b"complete-audio"
    assert len(instances) == 3
    assert delays == [1, 2]
    if os.name != "nt":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("failure", "attempts", "reason"),
    [
        (_connector_error(), 3, "connection"),
        (TimeoutError("private detail"), 3, "timeout"),
        (aiohttp.ClientResponseError(None, (), status=503), 3, "service"),
        (aiohttp.ClientResponseError(None, (), status=403), 1, "rejected"),
        (NoAudioReceived("private detail"), 3, "empty"),
        (ValueError("sk-private-example voice"), 1, "settings"),
        (PermissionError("/private/example/speech.mp3"), 1, "file"),
        (aiohttp.ClientSSLError(None, OSError("private certificate")), 1, "certificate"),
        (RuntimeError("private failure"), 1, "unknown"),
    ],
)
def test_final_speech_error_is_bounded_localized_and_does_not_expose_raw_errors(
    tmp_path, monkeypatch, failure, attempts, reason
):
    delays = []
    monkeypatch.setattr(tts_module, "sleep", delays.append)
    calls = []
    destination = tmp_path / "speech.mp3"

    class FailingCommunicator:
        async def save(self, target):
            calls.append(target)
            Path(target).write_bytes(b"partial")
            raise failure

    with pytest.raises(ProviderError) as caught:
        EdgeSpeechProvider(lambda *_: FailingCommunicator()).synthesize(
            "carrot", destination, voice="en-US-JennyNeural", rate="+0%"
        )
    error = caught.value
    assert len(calls) == attempts
    assert len(delays) == attempts - 1
    assert getattr(error, "attempts", None) == attempts
    assert getattr(error, "reason", None) == reason
    assert error.diagnostic == type(failure).__name__
    assert not destination.exists()
    zh = _safe_message(error, Locale.ZH_CN)
    en = _safe_message(error, Locale.EN_US)
    assert "语音" in zh and "Speech" in en
    assert str(attempts) in zh and str(attempts) in en
    for message in (zh, en, error.safe_message):
        assert "private" not in message
        assert "sk-" not in message


def test_invalid_communicator_settings_are_safe_and_not_retried(tmp_path):
    attempts = []

    def invalid_factory(*_args):
        attempts.append(1)
        raise ValueError("private voice parameter")

    with pytest.raises(ProviderError) as caught:
        EdgeSpeechProvider(invalid_factory).synthesize(
            "carrot", tmp_path / "speech.mp3", voice="invalid", rate="+0%"
        )
    assert len(attempts) == 1
    assert "private" not in caught.value.safe_message
    assert getattr(caught.value, "reason", None) == "settings"


def test_keyboard_interrupt_is_not_retried_and_partial_audio_is_removed(tmp_path):
    destination = tmp_path / "speech.mp3"

    class InterruptedCommunicator:
        async def save(self, target):
            Path(target).write_bytes(b"partial")
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        EdgeSpeechProvider(lambda *_: InterruptedCommunicator()).synthesize(
            "carrot", destination, voice="en-US-JennyNeural", rate="+0%"
        )
    assert not destination.exists()


def test_generic_speech_failure_is_also_localized():
    assert "语音" in _safe_message(ProviderError("Speech generation failed."), Locale.ZH_CN)


@pytest.mark.parametrize("mode", ["missing", "empty", "oversized"])
def test_unusable_speech_output_is_not_retried(tmp_path, mode):
    calls = []
    destination = tmp_path / "speech.mp3"

    class BadOutput:
        async def save(self, target):
            calls.append(target)
            if mode != "missing":
                with Path(target).open("wb") as output:
                    output.truncate(0 if mode == "empty" else 32 * 1024 * 1024 + 1)

    with pytest.raises(ProviderError) as caught:
        EdgeSpeechProvider(lambda *_: BadOutput()).synthesize(
            "carrot", destination, voice="en-US-JennyNeural", rate="+0%"
        )
    assert len(calls) == 1
    assert caught.value.reason == "output"
    assert not destination.exists()
    assert "音频" in _safe_message(caught.value, Locale.ZH_CN)


def test_cancelled_speech_request_is_not_retried(tmp_path):
    calls = []
    destination = tmp_path / "speech.mp3"

    class CancelledCommunicator:
        async def save(self, target):
            calls.append(target)
            Path(target).write_bytes(b"partial")
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        EdgeSpeechProvider(lambda *_: CancelledCommunicator()).synthesize(
            "carrot", destination, voice="en-US-JennyNeural", rate="+0%"
        )
    assert len(calls) == 1
    assert not destination.exists()


def _voice_preview_app(locale_value, storage):
    from pathlib import Path

    from ai_vocab_video_generator.config import AppSettings, SecretSettings
    from ai_vocab_video_generator.i18n import Locale
    from ai_vocab_video_generator.webui import _narration_panel

    _narration_panel(
        title_key="chinese_narration",
        prefix="chinese_narration",
        locale=Locale(locale_value),
        settings=AppSettings(storage_dir=Path(storage), secrets=SecretSettings(_env_file=None)),
        voice="zh-CN-XiaoxiaoNeural",
        repeats=1,
        rate=0,
        sample="胡萝卜",
        field_prefix_key="chinese_label",
        voice_language="zh",
    )


@pytest.mark.parametrize("locale", [Locale.ZH_CN, Locale.EN_US])
def test_voice_preview_renders_localized_retry_failure_without_a_traceback(
    tmp_path, monkeypatch, locale
):
    import edge_tts
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(tts_module, "sleep", lambda _: None)

    class UnreachableCommunicator:
        async def save(self, target):
            raise _connector_error()

    monkeypatch.setattr(edge_tts, "Communicate", lambda **_: UnreachableCommunicator())
    app = AppTest.from_function(_voice_preview_app, args=(locale.value, str(tmp_path))).run()
    assert not app.exception
    app.button(key="chinese_narration_play").click().run()
    assert not app.exception
    assert len(app.error) == 1
    message = app.error[0].value
    assert ("中文朗读失败" if locale is Locale.ZH_CN else "Chinese narration") in message
    assert "3" in message
    assert "private" not in message
