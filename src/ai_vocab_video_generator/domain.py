"""Validated domain models shared by generation and presentation layers."""

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import parse_qsl, unquote, urlsplit

from PIL import ImageColor
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

_CREDENTIAL_COMPONENT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:api[_-]?key|access[_-]?(?:key|token)|auth(?:orization)?|bearer|"
    r"client[_-]?secret|credential|key|password|secret|sig(?:nature)?|token|x[_-]?api[_-]?key)"
    r"\s*[:=][^/#?&;\s]+",
    re.IGNORECASE,
)
_CREDENTIAL_PATH_PATTERN = re.compile(
    r"(?:^|[/;])(?:api[_-]?key|access[_-]?(?:key|token)|client[_-]?secret|password|"
    r"secret|x[_-]?api[_-]?key)/[^/#?&;\s]+",
    re.IGNORECASE,
)
_SECRET_URL_PATTERN = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:"
    r"sk-(?:proj|svcacct|admin|live)-[A-Za-z0-9_-]{20,}|"
    r"sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,})"
    r"(?![A-Za-z0-9_-])"
)

MAX_TOPIC_LENGTH = 240
MAX_VOCABULARY_ENTRIES = 50
MAX_SCRIPT_LENGTH = 30_000


def _decode_url_component(value: str) -> str:
    decoded = value
    for _ in range(3):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def contains_sensitive_text(value: str, *, active_secrets: tuple[str, ...] = ()) -> bool:
    """Return whether untrusted text contains a credential or an active secret."""
    return any(secret and secret in value for secret in active_secrets) or bool(
        _SECRET_URL_PATTERN.search(value) or _CREDENTIAL_COMPONENT_PATTERN.search(value)
    )


def validate_llm_base_url(value: str) -> str:
    """Normalize an LLM endpoint and reject unsafe credential transport."""
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
    except ValueError as exc:
        raise ValueError("LLM base URL is invalid.") from exc
    is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback):
        raise ValueError("Use HTTPS, except for a loopback HTTP endpoint.")
    if not parsed.hostname:
        raise ValueError("LLM base URL is invalid.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("LLM base URL is invalid.") from exc
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("LLM base URL is invalid.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("LLM base URL must not contain credentials.")
    credential_keys = {
        "apikey",
        "api_key",
        "access_key",
        "access_token",
        "auth",
        "authorization",
        "bearer",
        "client_secret",
        "credential",
        "key",
        "password",
        "secret",
        "signature",
        "sig",
        "token",
        "x_api_key",
    }
    for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = _decode_url_component(key).casefold().replace("-", "_")
        decoded_value = _decode_url_component(query_value)
        if normalized_key in credential_keys or _SECRET_URL_PATTERN.search(decoded_value):
            raise ValueError("LLM base URL must not contain credentials.")
        if normalized_key != "api_version":
            raise ValueError(
                "LLM base URL query parameters are limited to the non-secret api-version option."
            )
    decoded_components = tuple(
        _decode_url_component(component)
        for component in (parsed.path, parsed.query, parsed.fragment)
    )
    if any(
        _CREDENTIAL_COMPONENT_PATTERN.search(component)
        or _CREDENTIAL_PATH_PATTERN.search(component)
        or _SECRET_URL_PATTERN.search(component)
        for component in decoded_components
    ):
        raise ValueError("LLM base URL must not contain credentials.")
    if parsed.fragment:
        raise ValueError("LLM base URL fragments are not supported.")
    return normalized


class VideoAspect(StrEnum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"

    @property
    def resolution(self) -> tuple[int, int]:
        return (1080, 1920) if self is self.PORTRAIT else (1920, 1080)


class PhoneticMode(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    DISABLED = "disabled"


class MaterialSource(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class RemoteMaterialProvider(StrEnum):
    PEXELS = "pexels"
    PIXABAY = "pixabay"


class SelectionMode(StrEnum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"


class MaterialShape(StrEnum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"


class MaterialFitMode(StrEnum):
    CONTAIN = "contain"
    COVER = "cover"
    STRETCH = "stretch"


class MaterialKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class RenderSettings(BaseModel):
    fps: Annotated[int, Field(ge=12, le=60)] = 24


class BackgroundMusicSettings(BaseModel):
    enabled: bool = False
    path: Path | None = None
    volume_percent: Annotated[int, Field(ge=0, le=100)] = 12
    ducking_percent: Annotated[int, Field(ge=0, le=100)] = 65

    @model_validator(mode="after")
    def validate_enabled_music(self) -> Self:
        if self.enabled and (
            self.path is None
            or not self.path.is_file()
            or self.path.suffix.lower() not in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}
        ):
            raise ValueError("Enabled background music requires an existing supported music file.")
        return self


class MaterialAsset(BaseModel):
    path: Path
    kind: MaterialKind
    source_id: str | None = None


class PinnedMaterial(BaseModel):
    entry_index: Annotated[int, Field(ge=0, le=49)]
    asset: MaterialAsset


class WordEntry(BaseModel):
    english: Annotated[str, Field(min_length=1, max_length=120)]
    phonetic: Annotated[str, Field(max_length=160)] = ""
    chinese: Annotated[str, Field(max_length=240)] = ""

    @field_validator("english", "phonetic", "chinese", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class Lesson(BaseModel):
    topic: str = ""
    entries: list[WordEntry]


class CanvasSettings(BaseModel):
    aspect: VideoAspect = VideoAspect.PORTRAIT
    width: Annotated[int, Field(ge=240, le=3840, multiple_of=2)] = 1080
    height: Annotated[int, Field(ge=240, le=3840, multiple_of=2)] = 1920

    @classmethod
    def for_aspect(cls, aspect: VideoAspect) -> Self:
        width, height = aspect.resolution
        return cls(aspect=aspect, width=width, height=height)


class AnchorOffsets(BaseModel):
    top: Annotated[int | None, Field(ge=0, le=4096)] = None
    bottom: Annotated[int | None, Field(ge=0, le=4096)] = None
    left: Annotated[int | None, Field(ge=0, le=4096)] = None
    right: Annotated[int | None, Field(ge=0, le=4096)] = None

    def resolve(
        self,
        canvas_size: tuple[int, int],
        element_size: tuple[int, int],
    ) -> tuple[int, int]:
        canvas_width, canvas_height = canvas_size
        element_width, element_height = element_size
        x = self._axis(canvas_width, element_width, self.left, self.right)
        y = self._axis(canvas_height, element_height, self.top, self.bottom)
        return x, y

    @staticmethod
    def _axis(
        canvas: int,
        element: int,
        leading: int | None,
        trailing: int | None,
    ) -> int:
        if element <= 0 or element > canvas:
            raise ValueError("Element creates an empty layout box.")
        if leading is not None and trailing is not None:
            available = canvas - leading - trailing
            if available < element:
                raise ValueError("Offsets create an empty layout box.")
            return leading + (available - element) // 2
        if leading is not None:
            if leading + element > canvas:
                raise ValueError("Offsets place the element outside the layout box.")
            return leading
        if trailing is not None:
            result = canvas - trailing - element
            if result < 0:
                raise ValueError("Offsets place the element outside the layout box.")
            return result
        return (canvas - element) // 2


def _validate_color(value: str) -> str:
    normalized = value.strip()
    try:
        ImageColor.getrgb(normalized)
    except ValueError as exc:
        raise ValueError("Use a valid Pillow color value.") from exc
    return normalized.upper() if normalized.startswith("#") else normalized


class TextElementStyle(BaseModel):
    _verified_font_bytes: bytes | None = PrivateAttr(default=None)

    enabled: bool = True
    font_path: Path | None = None
    font_size: Annotated[int, Field(ge=8, le=400)] = 80
    fill_color: str = "#000000"
    weight: Annotated[float, Field(ge=0, le=10)] = 1
    stroke_color: str = "#FFFFFF"
    stroke_width: Annotated[float, Field(ge=0, le=20)] = 1.5
    offsets: AnchorOffsets = Field(default_factory=AnchorOffsets)

    @field_validator("fill_color", "stroke_color")
    @classmethod
    def valid_color(cls, value: str) -> str:
        return _validate_color(value)

    @field_validator("font_path")
    @classmethod
    def existing_font(cls, value: Path | None) -> Path | None:
        if value is not None and not value.expanduser().is_file():
            raise ValueError("The selected font file does not exist.")
        return value.expanduser() if value is not None else None

    def bind_verified_font_bytes(self, contents: bytes) -> None:
        self._verified_font_bytes = bytes(contents)

    @property
    def verified_font_bytes(self) -> bytes | None:
        return self._verified_font_bytes


class ProgressBarStyle(BaseModel):
    enabled: bool = True
    width: Annotated[int, Field(ge=1, le=3840)] = 756
    height: Annotated[int, Field(ge=1, le=400)] = 20
    start_color: str = "#FFA500"
    end_color: str = "#ADFF2F"
    offsets: AnchorOffsets = Field(default_factory=lambda: AnchorOffsets(top=1422))

    @field_validator("start_color", "end_color")
    @classmethod
    def valid_color(cls, value: str) -> str:
        return _validate_color(value)


class MaterialStyle(BaseModel):
    enabled: bool = True
    width: Annotated[int, Field(ge=1, le=3840)] = 648
    height: Annotated[int, Field(ge=1, le=3840)] = 648
    shape: MaterialShape = MaterialShape.CIRCLE
    fit_mode: MaterialFitMode = MaterialFitMode.COVER
    source: MaterialSource = MaterialSource.REMOTE
    remote_provider: RemoteMaterialProvider = RemoteMaterialProvider.PEXELS
    selection_mode: SelectionMode = SelectionMode.SEQUENTIAL
    pool_size: Annotated[int, Field(ge=1, le=20)] = 8
    offsets: AnchorOffsets = Field(default_factory=lambda: AnchorOffsets(top=384))


class NarrationTrackSettings(BaseModel):
    enabled: bool = True
    repeats: Annotated[int, Field(ge=0, le=10)] = 0
    voice: Annotated[str, Field(min_length=1, max_length=120)]
    volume: Annotated[int, Field(ge=-100, le=100)] = 0
    rate: Annotated[int, Field(ge=-100, le=100)] = 0

    @field_validator("voice", mode="before")
    @classmethod
    def strip_voice(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @property
    def rate_value(self) -> str:
        return f"{self.rate:+d}%"

    @property
    def volume_value(self) -> str:
        return f"{self.volume:+d}%"


def _chinese_narration() -> NarrationTrackSettings:
    return NarrationTrackSettings(enabled=False, voice="zh-CN-XiaoxiaoNeural")


def _fast_english_narration() -> NarrationTrackSettings:
    return NarrationTrackSettings(voice="en-US-JennyNeural", repeats=1, rate=-20)


def _slow_english_narration() -> NarrationTrackSettings:
    return NarrationTrackSettings(enabled=False, voice="en-US-JennyNeural")


def _question_narration() -> NarrationTrackSettings:
    return NarrationTrackSettings(voice="en-US-JennyNeural", repeats=1, rate=-20)


class NarrationSettings(BaseModel):
    chinese: NarrationTrackSettings = Field(default_factory=_chinese_narration)
    fast_english: NarrationTrackSettings = Field(default_factory=_fast_english_narration)
    slow_english: NarrationTrackSettings = Field(default_factory=_slow_english_narration)
    question: NarrationTrackSettings = Field(default_factory=_question_narration)


class VocabularySettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    preset: Literal["openai", "deepseek", "moonshot", "qwen", "ollama", "custom"] = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.6-terra"
    prompt_version: str = "v1"

    @field_validator("base_url")
    @classmethod
    def reject_embedded_credentials(cls, value: str) -> str:
        return validate_llm_base_url(value)


def _question_style(aspect: VideoAspect = VideoAspect.PORTRAIT) -> TextElementStyle:
    top, size = (240, 80) if aspect is VideoAspect.PORTRAIT else (50, 80)
    return TextElementStyle(
        enabled=False,
        font_size=size,
        fill_color="#000000",
        stroke_color="#FFFFFF",
        offsets=AnchorOffsets(top=top),
    )


def _english_style(aspect: VideoAspect = VideoAspect.PORTRAIT) -> TextElementStyle:
    top, size = (1052, 100) if aspect is VideoAspect.PORTRAIT else (670, 100)
    return TextElementStyle(font_size=size, offsets=AnchorOffsets(top=top))


def _phonetic_style(aspect: VideoAspect = VideoAspect.PORTRAIT) -> TextElementStyle:
    top, size = (1252, 90) if aspect is VideoAspect.PORTRAIT else (770, 80)
    return TextElementStyle(font_size=size, offsets=AnchorOffsets(top=top))


def _chinese_style(aspect: VideoAspect = VideoAspect.PORTRAIT) -> TextElementStyle:
    top, size = (1422, 80) if aspect is VideoAspect.PORTRAIT else (880, 60)
    return TextElementStyle(font_size=size, offsets=AnchorOffsets(top=top))


def _material_style(aspect: VideoAspect = VideoAspect.PORTRAIT) -> MaterialStyle:
    width, height, top = (648, 648, 384) if aspect is VideoAspect.PORTRAIT else (400, 400, 200)
    return MaterialStyle(width=width, height=height, offsets=AnchorOffsets(top=top))


def _progress_style(aspect: VideoAspect = VideoAspect.PORTRAIT) -> ProgressBarStyle:
    top = 1422 if aspect is VideoAspect.PORTRAIT else 880
    return ProgressBarStyle(offsets=AnchorOffsets(top=top))


class GenerationRequest(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    topic: Annotated[str, Field(max_length=MAX_TOPIC_LENGTH)] = ""
    word_count: Annotated[int, Field(ge=1, le=MAX_VOCABULARY_ENTRIES)] = 10
    entries: Annotated[list[WordEntry], Field(max_length=MAX_VOCABULARY_ENTRIES)] = Field(
        default_factory=list
    )
    phonetic_mode: PhoneticMode = PhoneticMode.AUTOMATIC
    canvas: CanvasSettings = Field(default_factory=CanvasSettings)
    question_text: Annotated[str, Field(max_length=120)] = "What is this?"
    question: TextElementStyle = Field(default_factory=_question_style)
    material: MaterialStyle = Field(default_factory=_material_style)
    progress: ProgressBarStyle = Field(default_factory=_progress_style)
    english_text: TextElementStyle = Field(default_factory=_english_style)
    phonetic_text: TextElementStyle = Field(default_factory=_phonetic_style)
    chinese_text: TextElementStyle = Field(default_factory=_chinese_style)
    narration: NarrationSettings = Field(default_factory=NarrationSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    background_music: BackgroundMusicSettings = Field(default_factory=BackgroundMusicSettings)
    vocabulary: VocabularySettings = Field(default_factory=VocabularySettings)
    background_image: Path | None = None
    local_materials: list[Path] = Field(default_factory=list)
    pinned_materials: list[PinnedMaterial] = Field(default_factory=list)
    material_queries: dict[
        Annotated[int, Field(ge=0, le=49)], Annotated[str, Field(max_length=120)]
    ] = Field(default_factory=dict)
    job_seed: int | None = None

    @field_validator("material_queries")
    @classmethod
    def normalize_material_queries(cls, values: dict[int, str]) -> dict[int, str]:
        return {index: " ".join(query.split()) for index, query in values.items() if query.strip()}

    @field_validator("topic", mode="before")
    @classmethod
    def strip_topic(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.topic and not self.entries:
            raise ValueError("Provide a topic or at least one vocabulary entry.")
        tracks = (
            self.narration.chinese,
            self.narration.fast_english,
            self.narration.slow_english,
        )
        if not any(track.enabled and track.repeats > 0 for track in tracks):
            raise ValueError("At least one answer narration repeat must be enabled.")
        if len({material.entry_index for material in self.pinned_materials}) != len(
            self.pinned_materials
        ):
            raise ValueError("Pinned material entry indices must be unique.")
        if any(index >= len(self.entries) for index in self.material_queries):
            raise ValueError("Material search keywords must refer to an existing vocabulary entry.")
        return self

    @classmethod
    def with_aspect_defaults(cls, aspect: VideoAspect, **values: Any) -> Self:
        return cls(
            canvas=CanvasSettings.for_aspect(aspect),
            question=_question_style(aspect),
            material=_material_style(aspect),
            progress=_progress_style(aspect),
            english_text=_english_style(aspect),
            phonetic_text=_phonetic_style(aspect),
            chinese_text=_chinese_style(aspect),
            **values,
        )


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class PipelineStage(StrEnum):
    PREPARING = "preparing"
    VOCABULARY = "vocabulary"
    IMAGES = "images"
    SPEECH = "speech"
    CARDS = "cards"
    COMPOSING = "composing"
    COMPLETE = "complete"


class PipelineProgress(BaseModel):
    stage: PipelineStage
    percent: Annotated[int, Field(ge=0, le=100)]
    message: str


class GenerationResult(BaseModel):
    job_id: str
    status: JobStatus
    video_path: Path | None = None
    manifest_path: Path
