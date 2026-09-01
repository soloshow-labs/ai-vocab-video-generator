"""MoviePy-based composition with explicit clip cleanup."""

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ai_vocab_video_generator.domain import (
    BackgroundMusicSettings,
    MaterialAsset,
    MaterialKind,
    MaterialStyle,
    ProgressBarStyle,
    RenderSettings,
)
from ai_vocab_video_generator.errors import RenderingError
from ai_vocab_video_generator.private_fs import ensure_private_directory, mark_private_file
from ai_vocab_video_generator.rendering.layout import draw_progress, resolve_position
from ai_vocab_video_generator.rendering.media import apply_material_mask, fit_material_frame


@dataclass(frozen=True)
class MaterialVideoOverlay:
    asset: MaterialAsset
    style: MaterialStyle
    start_offset_seconds: float

    def __post_init__(self) -> None:
        if self.asset.kind is not MaterialKind.VIDEO:
            raise ValueError("A material video overlay requires a video asset.")
        if self.start_offset_seconds < 0:
            raise ValueError("Material video start offset must not be negative.")


@dataclass(frozen=True)
class VideoSegment:
    image_path: Path
    audio_paths: tuple[Path, ...] = field(default_factory=tuple)
    duration: float | None = None
    progress_style: ProgressBarStyle | None = None
    foreground_path: Path | None = None
    material_video: MaterialVideoOverlay | None = None

    def __post_init__(self) -> None:
        if self.duration is not None and self.duration <= 0:
            raise ValueError("Segment duration must be positive.")


_MUSIC_TRANSITION_SECONDS = 0.05
_MAX_TIMELINE_SECONDS = 3600.0
_MAX_AUDIO_BYTES = 32 * 1024 * 1024
_MAX_NARRATION_DURATION_SECONDS = 300.0
_MAX_MUSIC_DURATION_SECONDS = 3600.0


def _validate_audio_size(path: Path) -> None:
    if not path.is_file() or not 0 < path.stat().st_size <= _MAX_AUDIO_BYTES:
        raise ValueError("An audio input is empty or exceeds the size limit.")


def _validate_audio_duration(clip: Any, *, maximum: float) -> None:
    duration = float(clip.duration)
    if duration <= 0 or duration > maximum:
        raise ValueError("An audio input has an unsupported duration.")


def music_gain_at(
    time: float,
    intervals: Sequence[tuple[float, float]],
    base_gain: float,
    ducking: float,
) -> float:
    """Return the music gain with a 50 ms ramp around narration."""
    ducked_gain = base_gain * (1.0 - ducking)
    gain = base_gain
    for start, end in intervals:
        if start <= time <= end:
            return ducked_gain
        if start - _MUSIC_TRANSITION_SECONDS < time < start:
            progress = (time - start + _MUSIC_TRANSITION_SECONDS) / _MUSIC_TRANSITION_SECONDS
            gain = min(gain, base_gain + (ducked_gain - base_gain) * progress)
        elif end < time < end + _MUSIC_TRANSITION_SECONDS:
            progress = (time - end) / _MUSIC_TRANSITION_SECONDS
            gain = min(gain, ducked_gain + (base_gain - ducked_gain) * progress)
    return gain


class VideoComposer:
    def compose(
        self,
        segments: Sequence[VideoSegment],
        destination: Path,
        *,
        render: RenderSettings,
        music: BackgroundMusicSettings,
    ) -> Path:
        if not segments:
            raise ValueError("At least one video segment is required.")
        explicit_duration = sum(segment.duration or 0.0 for segment in segments)
        if explicit_duration > _MAX_TIMELINE_SECONDS:
            raise ValueError("The video timeline exceeds the one-hour limit.")
        ensure_private_directory(destination.parent)
        try:
            self._compose(segments, destination, render, music)
        except (OSError, RuntimeError, ValueError, IndexError) as exc:
            message = "Unable to compose the video. Check FFmpeg and media files."
            raise RenderingError(message) from exc
        return mark_private_file(destination)

    def _compose(
        self,
        segments: Sequence[VideoSegment],
        destination: Path,
        render: RenderSettings,
        music: BackgroundMusicSettings,
    ) -> None:
        import numpy as np
        from moviepy.audio.fx import all as afx  # type: ignore[import-untyped]
        from moviepy.editor import (  # type: ignore[import-untyped]
            AudioFileClip,
            CompositeAudioClip,
            ImageClip,
            VideoClip,
            VideoFileClip,
            concatenate_audioclips,
            concatenate_videoclips,
        )
        from PIL import Image

        with ExitStack() as stack:
            video_clips: list[Any] = []
            narration_intervals: list[tuple[float, float]] = []
            timeline_duration = 0.0
            for segment in segments:
                audio_clips: list[Any] = []
                for path in segment.audio_paths:
                    _validate_audio_size(path)
                    audio_clip = AudioFileClip(str(path))
                    stack.callback(audio_clip.close)
                    _validate_audio_duration(audio_clip, maximum=_MAX_NARRATION_DURATION_SECONDS)
                    audio_clips.append(audio_clip)
                audio = None
                if audio_clips:
                    audio = concatenate_audioclips(audio_clips)
                    stack.callback(audio.close)
                duration = segment.duration or (float(audio.duration) if audio is not None else 1.0)
                foreground_audio = audio
                if audio is not None and float(audio.duration) > duration:
                    foreground_audio = audio.subclip(0, duration)
                    stack.callback(foreground_audio.close)
                if foreground_audio is not None:
                    narration_intervals.append(
                        (
                            timeline_duration,
                            timeline_duration + float(foreground_audio.duration),
                        )
                    )
                timeline_duration += duration
                if timeline_duration > _MAX_TIMELINE_SECONDS:
                    raise ValueError("The video timeline exceeds the one-hour limit.")
                dynamic = (
                    segment.progress_style is not None
                    or segment.foreground_path is not None
                    or segment.material_video is not None
                )
                if not dynamic:
                    with Image.open(segment.image_path) as source:
                        converted = source.convert("RGB")
                        try:
                            base_frame = np.asarray(converted).copy()
                        finally:
                            converted.close()
                    clip = ImageClip(base_frame).set_duration(duration)
                else:
                    with Image.open(segment.image_path) as source:
                        converted = source.convert("RGB")
                        try:
                            base_frame = np.asarray(converted).copy()
                        finally:
                            converted.close()
                    foreground_frame = None
                    if segment.foreground_path is not None:
                        with Image.open(segment.foreground_path) as source:
                            converted = source.convert("RGBA")
                            try:
                                foreground_frame = np.asarray(converted).copy()
                            finally:
                                converted.close()
                    material_clip = None
                    material_duration = None
                    if segment.material_video is not None:
                        material_clip = VideoFileClip(
                            str(segment.material_video.asset.path), audio=False
                        )
                        stack.callback(material_clip.close)
                        material_duration = float(material_clip.duration)
                        if material_duration <= 0:
                            raise ValueError("Material video duration must be positive.")

                    def make_frame(
                        time: float,
                        *,
                        frame: Any = base_frame,
                        foreground: Any = foreground_frame,
                        overlay: MaterialVideoOverlay | None = segment.material_video,
                        source_clip: Any = material_clip,
                        source_duration: float | None = material_duration,
                        progress_style: ProgressBarStyle | None = segment.progress_style,
                        clip_duration: float = duration,
                    ) -> Any:
                        base_image = Image.fromarray(frame.copy())
                        try:
                            canvas = base_image.convert("RGBA")
                        finally:
                            base_image.close()
                        try:
                            if overlay is not None:
                                assert source_clip is not None
                                assert source_duration is not None
                                source_time = (
                                    overlay.start_offset_seconds + time
                                ) % source_duration
                                source_image = Image.fromarray(source_clip.get_frame(source_time))
                                try:
                                    fitted = fit_material_frame(
                                        source_image,
                                        (overlay.style.width, overlay.style.height),
                                        overlay.style.fit_mode,
                                    )
                                finally:
                                    source_image.close()
                                try:
                                    material = apply_material_mask(fitted, overlay.style.shape)
                                finally:
                                    fitted.close()
                                try:
                                    position = resolve_position(
                                        canvas.size, material.size, overlay.style.offsets
                                    )
                                    canvas.paste(material, position, material)
                                finally:
                                    material.close()
                            if foreground is not None:
                                foreground_image = Image.fromarray(foreground.copy(), mode="RGBA")
                                try:
                                    canvas.alpha_composite(foreground_image)
                                finally:
                                    foreground_image.close()
                            if progress_style is not None:
                                draw_progress(canvas, progress_style, time / clip_duration)
                            result = canvas.convert("RGB")
                            try:
                                return np.asarray(result).copy()
                            finally:
                                result.close()
                        finally:
                            canvas.close()

                    clip = VideoClip(make_frame=make_frame, duration=duration)
                stack.callback(clip.close)
                if foreground_audio is not None:
                    clip = clip.set_audio(foreground_audio)
                    stack.callback(clip.close)
                video_clips.append(clip)

            video = concatenate_videoclips(video_clips, method="compose")
            stack.callback(video.close)
            output_video = video
            if music.enabled:
                assert music.path is not None
                _validate_audio_size(music.path)
                music_clip = AudioFileClip(str(music.path))
                stack.callback(music_clip.close)
                _validate_audio_duration(music_clip, maximum=_MAX_MUSIC_DURATION_SECONDS)
                looped_music = afx.audio_loop(music_clip, duration=timeline_duration)
                stack.callback(looped_music.close)
                intervals = tuple(narration_intervals)
                base_gain = music.volume_percent / 100.0
                ducking = music.ducking_percent / 100.0

                def apply_music_gain(get_frame: Any, time: Any) -> Any:
                    frame = get_frame(time)
                    if np.isscalar(time):
                        return frame * music_gain_at(
                            float(cast(float, time)), intervals, base_gain, ducking
                        )
                    gains = np.asarray(
                        [
                            music_gain_at(float(value), intervals, base_gain, ducking)
                            for value in np.asarray(time)
                        ]
                    )
                    if np.asarray(frame).ndim > 1:
                        gains = gains[:, None]
                    return frame * gains

                ducked_music = looped_music.fl(apply_music_gain).set_duration(timeline_duration)
                stack.callback(ducked_music.close)
                audio_tracks = [ducked_music]
                if video.audio is not None:
                    audio_tracks.insert(0, video.audio)
                mixed_audio = CompositeAudioClip(audio_tracks)
                stack.callback(mixed_audio.close)
                output_video = video.set_audio(mixed_audio)
                stack.callback(output_video.close)
            output_video.write_videofile(
                str(destination),
                fps=render.fps,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=str(destination.with_suffix(".audio.m4a")),
                remove_temp=True,
                logger=None,
                threads=1,
            )
