"""Configuration loading with a hard boundary between preferences and secrets."""

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_vocab_video_generator.domain import validate_llm_base_url


class SecretSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIVVG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    moonshot_api_key: SecretStr | None = None
    qwen_api_key: SecretStr | None = None
    custom_api_key: SecretStr | None = None
    pexels_api_key: SecretStr | None = None
    pixabay_api_key: SecretStr | None = None

    @classmethod
    def empty(cls) -> "SecretSettings":
        """Return settings that ignore every process and dotenv credential."""
        return cls(
            openai_api_key=None,
            deepseek_api_key=None,
            moonshot_api_key=None,
            qwen_api_key=None,
            custom_api_key=None,
            pexels_api_key=None,
            pixabay_api_key=None,
        )

    def values(self) -> tuple[str, ...]:
        configured = (
            self.openai_api_key,
            self.deepseek_api_key,
            self.moonshot_api_key,
            self.qwen_api_key,
            self.custom_api_key,
            self.pexels_api_key,
            self.pixabay_api_key,
        )
        return tuple(
            secret.get_secret_value()
            for secret in configured
            if secret is not None and secret.get_secret_value()
        )

    def for_preset(self, preset: "LLMPreset") -> SecretStr | None:
        return {
            LLMPreset.OPENAI: self.openai_api_key,
            LLMPreset.DEEPSEEK: self.deepseek_api_key,
            LLMPreset.MOONSHOT: self.moonshot_api_key,
            LLMPreset.QWEN: self.qwen_api_key,
            LLMPreset.OLLAMA: None,
            LLMPreset.CUSTOM: self.custom_api_key,
        }[preset]


class LLMPreset(StrEnum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    QWEN = "qwen"
    OLLAMA = "ollama"
    CUSTOM = "custom"


LLM_PRESETS: dict[LLMPreset, tuple[str, str]] = {
    LLMPreset.OPENAI: ("https://api.openai.com/v1", "gpt-5.6-terra"),
    LLMPreset.DEEPSEEK: ("https://api.deepseek.com", "deepseek-v4-flash"),
    LLMPreset.MOONSHOT: ("https://api.moonshot.cn/v1", "kimi-k2.6"),
    LLMPreset.QWEN: (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen3.7-flash",
    ),
    LLMPreset.OLLAMA: ("http://localhost:11434/v1", "qwen3.5:9b"),
    LLMPreset.CUSTOM: ("https://api.openai.com/v1", "gpt-4o-mini"),
}


class LLMSettings(BaseModel):
    preset: LLMPreset = LLMPreset.OPENAI
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.6-terra"

    @field_validator("base_url")
    @classmethod
    def secure_base_url(cls, value: str) -> str:
        return validate_llm_base_url(value)

    @field_validator("model")
    @classmethod
    def nonempty_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Enter an LLM model name.")
        return normalized

    @classmethod
    def for_preset(cls, preset: LLMPreset) -> "LLMSettings":
        base_url, model = LLM_PRESETS[preset]
        return cls(preset=preset, base_url=base_url, model=model)


class AppSettings(BaseModel):
    storage_dir: Path = Path("storage")
    model_cache_dir: Path = Path("model_cache")
    font_path: Path | None = None
    llm: LLMSettings = Field(default_factory=LLMSettings)
    secrets: SecretSettings = Field(default_factory=SecretSettings, exclude=True)

    @classmethod
    def from_toml(cls, path: Path | None = None) -> "AppSettings":
        config_path = path or Path("config.toml")
        if not config_path.is_file():
            return cls()
        with config_path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        app = raw.get("app", {})
        llm = raw.get("llm", {})
        return cls(
            storage_dir=app.get("storage_dir", "storage"),
            model_cache_dir=app.get("model_cache_dir", "model_cache"),
            font_path=app.get("font_path") or None,
            llm=LLMSettings.model_validate(llm),
        )
