import pytest
from pydantic import ValidationError

from ai_vocab_video_generator.config import AppSettings, LLMPreset, LLMSettings, SecretSettings
from ai_vocab_video_generator.domain import GenerationRequest


def test_secret_settings_do_not_reveal_values(monkeypatch) -> None:
    secret = "openai-secret-value"
    monkeypatch.setenv("AIVVG_OPENAI_API_KEY", secret)

    settings = SecretSettings()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == secret
    assert secret not in repr(settings)
    assert secret not in settings.model_dump_json()


def test_llm_provider_secrets_are_independent(monkeypatch) -> None:
    values = {
        LLMPreset.OPENAI: "openai-session-secret",
        LLMPreset.DEEPSEEK: "deepseek-session-secret",
        LLMPreset.MOONSHOT: "moonshot-session-secret",
        LLMPreset.QWEN: "qwen-session-secret",
        LLMPreset.CUSTOM: "custom-session-secret",
    }
    monkeypatch.setenv("AIVVG_OPENAI_API_KEY", values[LLMPreset.OPENAI])
    monkeypatch.setenv("AIVVG_DEEPSEEK_API_KEY", values[LLMPreset.DEEPSEEK])
    monkeypatch.setenv("AIVVG_MOONSHOT_API_KEY", values[LLMPreset.MOONSHOT])
    monkeypatch.setenv("AIVVG_QWEN_API_KEY", values[LLMPreset.QWEN])
    monkeypatch.setenv("AIVVG_CUSTOM_API_KEY", values[LLMPreset.CUSTOM])

    settings = SecretSettings()

    for preset, expected in values.items():
        selected = settings.for_preset(preset)
        assert selected is not None
        assert selected.get_secret_value() == expected
    assert settings.for_preset(LLMPreset.OLLAMA) is None


@pytest.mark.parametrize(
    ("preset", "base_url", "model"),
    [
        (LLMPreset.OPENAI, "https://api.openai.com/v1", "gpt-5.6-terra"),
        (LLMPreset.DEEPSEEK, "https://api.deepseek.com", "deepseek-v4-flash"),
        (LLMPreset.MOONSHOT, "https://api.moonshot.cn/v1", "kimi-k2.6"),
        (
            LLMPreset.QWEN,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen3.7-flash",
        ),
        (LLMPreset.OLLAMA, "http://localhost:11434/v1", "qwen3.5:9b"),
    ],
)
def test_llm_presets_have_safe_openai_compatible_defaults(
    preset: LLMPreset, base_url: str, model: str
) -> None:
    settings = LLMSettings.for_preset(preset)

    assert settings.base_url == base_url
    assert settings.model == model


def test_openai_default_model_is_consistent_across_app_and_job_settings() -> None:
    assert AppSettings().llm.model == "gpt-5.6-terra"
    assert GenerationRequest(topic="daily life").vocabulary.model == "gpt-5.6-terra"


def test_custom_llm_rejects_unencrypted_remote_urls() -> None:
    with pytest.raises(ValidationError, match="HTTPS"):
        LLMSettings(preset=LLMPreset.CUSTOM, base_url="http://llm.example/v1", model="demo")

    local = LLMSettings(
        preset=LLMPreset.CUSTOM,
        base_url="http://127.0.0.1:8000/v1",
        model="demo",
    )
    assert local.base_url == "http://127.0.0.1:8000/v1"
