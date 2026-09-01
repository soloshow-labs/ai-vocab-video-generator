"""Stable Edge TTS voice catalog for the languages used by the app."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EdgeVoice:
    short_name: str
    locale: str
    gender: Literal["Female", "Male"]

    @property
    def display_name(self) -> str:
        return self.short_name.removeprefix(f"{self.locale}-").removesuffix("Neural")


EDGE_TTS_VOICES: tuple[EdgeVoice, ...] = (
    EdgeVoice("zh-HK-HiuGaaiNeural", "zh-HK", "Female"),
    EdgeVoice("zh-HK-HiuMaanNeural", "zh-HK", "Female"),
    EdgeVoice("zh-HK-WanLungNeural", "zh-HK", "Male"),
    EdgeVoice("zh-CN-XiaoxiaoNeural", "zh-CN", "Female"),
    EdgeVoice("zh-CN-XiaoyiNeural", "zh-CN", "Female"),
    EdgeVoice("zh-CN-YunjianNeural", "zh-CN", "Male"),
    EdgeVoice("zh-CN-YunxiNeural", "zh-CN", "Male"),
    EdgeVoice("zh-CN-YunxiaNeural", "zh-CN", "Male"),
    EdgeVoice("zh-CN-YunyangNeural", "zh-CN", "Male"),
    EdgeVoice("zh-CN-liaoning-XiaobeiNeural", "zh-CN-liaoning", "Female"),
    EdgeVoice("zh-TW-HsiaoChenNeural", "zh-TW", "Female"),
    EdgeVoice("zh-TW-YunJheNeural", "zh-TW", "Male"),
    EdgeVoice("zh-TW-HsiaoYuNeural", "zh-TW", "Female"),
    EdgeVoice("zh-CN-shaanxi-XiaoniNeural", "zh-CN-shaanxi", "Female"),
    EdgeVoice("en-AU-WilliamMultilingualNeural", "en-AU", "Male"),
    EdgeVoice("en-AU-NatashaNeural", "en-AU", "Female"),
    EdgeVoice("en-CA-ClaraNeural", "en-CA", "Female"),
    EdgeVoice("en-CA-LiamNeural", "en-CA", "Male"),
    EdgeVoice("en-HK-YanNeural", "en-HK", "Female"),
    EdgeVoice("en-HK-SamNeural", "en-HK", "Male"),
    EdgeVoice("en-IN-NeerjaExpressiveNeural", "en-IN", "Female"),
    EdgeVoice("en-IN-NeerjaNeural", "en-IN", "Female"),
    EdgeVoice("en-IN-PrabhatNeural", "en-IN", "Male"),
    EdgeVoice("en-IE-ConnorNeural", "en-IE", "Male"),
    EdgeVoice("en-IE-EmilyNeural", "en-IE", "Female"),
    EdgeVoice("en-KE-AsiliaNeural", "en-KE", "Female"),
    EdgeVoice("en-KE-ChilembaNeural", "en-KE", "Male"),
    EdgeVoice("en-NZ-MitchellNeural", "en-NZ", "Male"),
    EdgeVoice("en-NZ-MollyNeural", "en-NZ", "Female"),
    EdgeVoice("en-NG-AbeoNeural", "en-NG", "Male"),
    EdgeVoice("en-NG-EzinneNeural", "en-NG", "Female"),
    EdgeVoice("en-PH-JamesNeural", "en-PH", "Male"),
    EdgeVoice("en-PH-RosaNeural", "en-PH", "Female"),
    EdgeVoice("en-US-AvaNeural", "en-US", "Female"),
    EdgeVoice("en-US-AndrewNeural", "en-US", "Male"),
    EdgeVoice("en-US-EmmaNeural", "en-US", "Female"),
    EdgeVoice("en-US-BrianNeural", "en-US", "Male"),
    EdgeVoice("en-SG-LunaNeural", "en-SG", "Female"),
    EdgeVoice("en-SG-WayneNeural", "en-SG", "Male"),
    EdgeVoice("en-ZA-LeahNeural", "en-ZA", "Female"),
    EdgeVoice("en-ZA-LukeNeural", "en-ZA", "Male"),
    EdgeVoice("en-TZ-ElimuNeural", "en-TZ", "Male"),
    EdgeVoice("en-TZ-ImaniNeural", "en-TZ", "Female"),
    EdgeVoice("en-GB-LibbyNeural", "en-GB", "Female"),
    EdgeVoice("en-GB-MaisieNeural", "en-GB", "Female"),
    EdgeVoice("en-GB-RyanNeural", "en-GB", "Male"),
    EdgeVoice("en-GB-SoniaNeural", "en-GB", "Female"),
    EdgeVoice("en-GB-ThomasNeural", "en-GB", "Male"),
    EdgeVoice("en-US-AnaNeural", "en-US", "Female"),
    EdgeVoice("en-US-AndrewMultilingualNeural", "en-US", "Male"),
    EdgeVoice("en-US-AriaNeural", "en-US", "Female"),
    EdgeVoice("en-US-AvaMultilingualNeural", "en-US", "Female"),
    EdgeVoice("en-US-BrianMultilingualNeural", "en-US", "Male"),
    EdgeVoice("en-US-ChristopherNeural", "en-US", "Male"),
    EdgeVoice("en-US-EmmaMultilingualNeural", "en-US", "Female"),
    EdgeVoice("en-US-EricNeural", "en-US", "Male"),
    EdgeVoice("en-US-GuyNeural", "en-US", "Male"),
    EdgeVoice("en-US-JennyNeural", "en-US", "Female"),
    EdgeVoice("en-US-MichelleNeural", "en-US", "Female"),
    EdgeVoice("en-US-RogerNeural", "en-US", "Male"),
    EdgeVoice("en-US-SteffanNeural", "en-US", "Male"),
)

EDGE_TTS_VOICE_BY_NAME = {voice.short_name: voice for voice in EDGE_TTS_VOICES}


def edge_voices_for_language(language: Literal["zh", "en"]) -> tuple[EdgeVoice, ...]:
    """Return all catalog voices for one supported narration language."""

    return tuple(voice for voice in EDGE_TTS_VOICES if voice.locale.startswith(f"{language}-"))
