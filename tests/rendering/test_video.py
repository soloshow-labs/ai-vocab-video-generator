import math
import os
import stat
import struct
import warnings
import wave
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ai_vocab_video_generator.domain import (
    AnchorOffsets,
    BackgroundMusicSettings,
    MaterialAsset,
    MaterialFitMode,
    MaterialKind,
    MaterialShape,
    MaterialStyle,
    ProgressBarStyle,
    RenderSettings,
)
from ai_vocab_video_generator.rendering import video as video_module
from ai_vocab_video_generator.rendering.video import (
    MaterialVideoOverlay,
    VideoComposer,
    VideoSegment,
)


def _write_color_video(
    destination: Path,
    *,
    duration: float,
    switch_at: float,
    with_audio: bool = False,
) -> None:
    moviepy = pytest.importorskip("moviepy.editor")
    np = pytest.importorskip("numpy")

    def make_frame(time: float) -> Any:
        color = (255, 0, 0) if time < switch_at else (0, 0, 255)
        return np.full((90, 80, 3), color, dtype=np.uint8)

    video = moviepy.VideoClip(make_frame=make_frame, duration=duration)
    audio = None
    clip = video
    if with_audio:
        audio = moviepy.AudioClip(lambda _time: 0.1, duration=duration, fps=44100)
        clip = video.set_audio(audio)
    try:
        clip.write_videofile(
            str(destination),
            fps=20,
            codec="libx264",
            audio_codec="aac",
            audio=with_audio,
            logger=None,
            threads=1,
        )
    finally:
        if clip is not video:
            clip.close()
        if audio is not None:
            audio.close()
        video.close()


def _write_sine_wav(
    destination: Path,
    *,
    duration: float,
    frequency: float,
    amplitude: float,
    sample_rate: int = 44_100,
) -> None:
    sample_count = round(duration * sample_rate)
    frames = b"".join(
        struct.pack(
            "<h",
            round(amplitude * 32_767 * math.sin(2.0 * math.pi * frequency * index / sample_rate)),
        )
        for index in range(sample_count)
    )
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)


def _tone_rms(samples: Any, *, frequency: float, sample_rate: int = 44_100) -> float:
    np = pytest.importorskip("numpy")
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    times = np.arange(len(mono)) / sample_rate
    sine = np.sin(2.0 * np.pi * frequency * times)
    cosine = np.cos(2.0 * np.pi * frequency * times)
    sine_amplitude = 2.0 * float(np.dot(mono, sine)) / len(mono)
    cosine_amplitude = 2.0 * float(np.dot(mono, cosine)) / len(mono)
    return math.sqrt((sine_amplitude**2 + cosine_amplitude**2) / 2.0)


def _soundarray(audio: Any, *, sample_rate: int = 44_100) -> Any:
    np = pytest.importorskip("numpy")
    return np.vstack(list(audio.iter_chunks(fps=sample_rate, chunksize=4096, logger=None)))


def _material_overlay(source: Path) -> MaterialVideoOverlay:
    style = MaterialStyle(
        width=80,
        height=90,
        shape=MaterialShape.RECTANGLE,
        fit_mode=MaterialFitMode.STRETCH,
        offsets=AnchorOffsets(top=0, left=0),
    )
    return MaterialVideoOverlay(
        asset=MaterialAsset(path=source, kind=MaterialKind.VIDEO),
        style=style,
        start_offset_seconds=0.0,
    )


def test_music_gain_ducks_during_narration_with_a_fifty_millisecond_transition() -> None:
    intervals = [(1.0, 2.0)]

    assert video_module.music_gain_at(0.5, intervals, 0.12, 0.65) == pytest.approx(0.12)
    assert video_module.music_gain_at(1.5, intervals, 0.12, 0.65) == pytest.approx(0.042)
    assert video_module.music_gain_at(0.975, intervals, 0.12, 0.65) == pytest.approx(0.081)
    assert video_module.music_gain_at(2.1, intervals, 0.12, 0.65) == pytest.approx(0.12)


def test_composer_encodes_at_the_request_fps(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    moviepy = pytest.importorskip("moviepy.editor")
    image = tmp_path / "frame.png"
    Image.new("RGB", (160, 90), "red").save(image)
    destination = tmp_path / "fifteen-fps.mp4"

    VideoComposer().compose(
        [VideoSegment(image_path=image, duration=0.4)],
        destination,
        render=RenderSettings(fps=15),
        music=BackgroundMusicSettings(),
    )

    clip = moviepy.VideoFileClip(str(destination), audio=False)
    try:
        assert round(clip.fps) == 15
    finally:
        clip.close()


def test_static_image_render_avoids_the_deprecated_imageio_read_path(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    image = tmp_path / "frame.png"
    Image.new("RGB", (160, 90), "red").save(image)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        VideoComposer().compose(
            [VideoSegment(image_path=image, duration=0.2)],
            tmp_path / "static.mp4",
            render=RenderSettings(fps=15),
            music=BackgroundMusicSettings(),
        )

    assert not [
        warning for warning in captured if "Starting with ImageIO v3" in str(warning.message)
    ]


def test_composer_ducks_music_only_while_foreground_audio_is_active(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    moviepy = pytest.importorskip("moviepy.editor")
    image = tmp_path / "frame.png"
    music_path = tmp_path / "music.wav"
    foreground_path = tmp_path / "foreground.wav"
    Image.new("RGB", (160, 90), "red").save(image)
    _write_sine_wav(music_path, duration=2.0, frequency=220.0, amplitude=0.5)
    _write_sine_wav(foreground_path, duration=1.0, frequency=880.0, amplitude=0.15)
    destination = tmp_path / "ducked.mp4"

    VideoComposer().compose(
        [
            VideoSegment(image_path=image, audio_paths=(foreground_path,), duration=1.0),
            VideoSegment(image_path=image, duration=1.0),
        ],
        destination,
        render=RenderSettings(fps=15),
        music=BackgroundMusicSettings(
            enabled=True,
            path=music_path,
            volume_percent=30,
            ducking_percent=50,
        ),
    )

    clip = moviepy.VideoFileClip(str(destination))
    try:
        assert clip.audio is not None
        samples = _soundarray(clip.audio)
    finally:
        clip.close()
    ducked = _tone_rms(samples[11_025:33_075], frequency=220.0)
    release = _tone_rms(samples[44_100:46_305], frequency=220.0)
    full = _tone_rms(samples[55_125:77_175], frequency=220.0)
    assert 0.025 < ducked < 0.05
    assert 0.06 < full < 0.09
    assert 1.7 < full / ducked < 2.3
    assert ducked * 1.25 < release < full * 0.9


def test_foreground_and_ducking_end_at_the_video_segment_boundary(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    moviepy = pytest.importorskip("moviepy.editor")
    image = tmp_path / "frame.png"
    music_path = tmp_path / "music.wav"
    foreground_path = tmp_path / "overlong-foreground.wav"
    Image.new("RGB", (160, 90), "red").save(image)
    _write_sine_wav(music_path, duration=2.0, frequency=220.0, amplitude=0.5)
    _write_sine_wav(foreground_path, duration=2.0, frequency=880.0, amplitude=0.15)
    destination = tmp_path / "trimmed-foreground.mp4"

    VideoComposer().compose(
        [
            VideoSegment(image_path=image, audio_paths=(foreground_path,), duration=1.0),
            VideoSegment(image_path=image, duration=1.0),
        ],
        destination,
        render=RenderSettings(fps=15),
        music=BackgroundMusicSettings(enabled=True, path=music_path),
    )

    clip = moviepy.VideoFileClip(str(destination))
    try:
        assert clip.audio is not None
        samples = _soundarray(clip.audio)
    finally:
        clip.close()
    narration_during_first = _tone_rms(samples[11_025:33_075], frequency=880.0)
    narration_during_second = _tone_rms(samples[55_125:77_175], frequency=880.0)
    music_during_first = _tone_rms(samples[11_025:33_075], frequency=220.0)
    music_during_second = _tone_rms(samples[55_125:77_175], frequency=220.0)
    assert narration_during_first > 0.05
    assert narration_during_second < narration_during_first * 0.05
    assert music_during_second > music_during_first * 2.0


def test_composer_loops_short_music_through_the_final_duration(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    moviepy = pytest.importorskip("moviepy.editor")
    image = tmp_path / "frame.png"
    music_path = tmp_path / "short-music.wav"
    Image.new("RGB", (160, 90), "blue").save(image)
    _write_sine_wav(music_path, duration=0.5, frequency=220.0, amplitude=0.5)
    destination = tmp_path / "looped-music.mp4"

    VideoComposer().compose(
        [
            VideoSegment(image_path=image, duration=1.0),
            VideoSegment(image_path=image, duration=1.0),
        ],
        destination,
        render=RenderSettings(fps=15),
        music=BackgroundMusicSettings(enabled=True, path=music_path),
    )

    clip = moviepy.VideoFileClip(str(destination))
    try:
        assert clip.audio is not None
        samples = _soundarray(clip.audio)
    finally:
        clip.close()
    after_two_source_lengths = _tone_rms(samples[55_125:77_175], frequency=220.0)
    assert after_two_source_lengths > 0.02


def test_composer_does_not_add_music_when_it_is_disabled(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    moviepy = pytest.importorskip("moviepy.editor")
    image = tmp_path / "frame.png"
    music_path = tmp_path / "disabled-music.wav"
    Image.new("RGB", (160, 90), "green").save(image)
    _write_sine_wav(music_path, duration=1.0, frequency=220.0, amplitude=0.5)
    destination = tmp_path / "silent.mp4"

    VideoComposer().compose(
        [VideoSegment(image_path=image, duration=1.0)],
        destination,
        render=RenderSettings(fps=15),
        music=BackgroundMusicSettings(enabled=False, path=music_path),
    )

    clip = moviepy.VideoFileClip(str(destination))
    try:
        assert clip.audio is None
    finally:
        clip.close()


def test_composer_creates_a_two_segment_mp4(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (160, 90), "red").save(first)
    Image.new("RGB", (160, 90), "blue").save(second)
    destination = tmp_path / "result.mp4"

    result = VideoComposer().compose(
        [
            VideoSegment(image_path=first, duration=0.2),
            VideoSegment(image_path=second, duration=0.2),
        ],
        destination,
        render=RenderSettings(fps=12),
        music=BackgroundMusicSettings(),
    )

    assert result == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0


def test_composer_animates_muted_material_beneath_foreground(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    moviepy = pytest.importorskip("moviepy.editor")
    source = tmp_path / "material.mp4"
    _write_color_video(source, duration=0.4, switch_at=0.2, with_audio=True)
    base = tmp_path / "base.png"
    foreground = tmp_path / "foreground.png"
    Image.new("RGB", (160, 90), "white").save(base)
    marker = Image.new("RGBA", (160, 90))
    for x in range(20):
        for y in range(20):
            marker.putpixel((x, y), (0, 0, 0, 255))
    marker.save(foreground)
    marker.close()
    destination = tmp_path / "animated.mp4"

    VideoComposer().compose(
        [
            VideoSegment(
                image_path=base,
                foreground_path=foreground,
                material_video=_material_overlay(source),
                duration=0.4,
            )
        ],
        destination,
        render=RenderSettings(fps=20),
        music=BackgroundMusicSettings(),
    )

    clip = moviepy.VideoFileClip(str(destination))
    try:
        early = clip.get_frame(0.05)
        late = clip.get_frame(0.30)
        assert early[45, 40].tolist() != late[45, 40].tolist()
        assert early[10, 10].tolist() == late[10, 10].tolist()
        assert clip.audio is None
    finally:
        clip.close()


def test_composer_loops_short_material_for_a_long_segment(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    moviepy = pytest.importorskip("moviepy.editor")
    source = tmp_path / "short.mp4"
    _write_color_video(source, duration=0.2, switch_at=0.1)
    base = tmp_path / "base.png"
    Image.new("RGB", (160, 90), "white").save(base)
    destination = tmp_path / "looped.mp4"

    VideoComposer().compose(
        [
            VideoSegment(
                image_path=base,
                material_video=_material_overlay(source),
                duration=0.45,
            )
        ],
        destination,
        render=RenderSettings(fps=20),
        music=BackgroundMusicSettings(),
    )

    clip = moviepy.VideoFileClip(str(destination), audio=False)
    try:
        first_pass = clip.get_frame(0.05)
        second_pass = clip.get_frame(0.25)
    finally:
        clip.close()
    assert first_pass[45, 40].tolist() == second_pass[45, 40].tolist()


def test_composer_rejects_empty_segments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="segment"):
        VideoComposer().compose(
            [],
            tmp_path / "empty.mp4",
            render=RenderSettings(),
            music=BackgroundMusicSettings(),
        )


def test_composer_rejects_an_excessive_total_timeline_before_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    composer = VideoComposer()
    monkeypatch.setattr(
        composer,
        "_compose",
        lambda *_args, **_kwargs: pytest.fail("rendering must not start"),
    )

    with pytest.raises(ValueError, match="timeline"):
        composer.compose(
            [VideoSegment(image_path=Path("frame.png"), duration=1801.0)] * 2,
            tmp_path / "too-long.mp4",
            render=RenderSettings(),
            music=BackgroundMusicSettings(),
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_composer_marks_output_owner_only(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    image = tmp_path / "frame.png"
    Image.new("RGB", (160, 90), "red").save(image)
    destination = tmp_path / "private" / "video.mp4"

    VideoComposer().compose(
        [VideoSegment(image_path=image, duration=0.2)],
        destination,
        render=RenderSettings(fps=12),
        music=BackgroundMusicSettings(),
    )

    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_question_segment_accepts_time_based_progress() -> None:
    style = ProgressBarStyle(
        width=120,
        height=10,
        offsets=AnchorOffsets(top=70),
    )

    segment = VideoSegment(
        image_path=Path("question.png"),
        duration=2.0,
        progress_style=style,
    )

    assert segment.progress_style == style


def test_composer_animates_progress_from_start_to_end(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    moviepy = pytest.importorskip("moviepy.editor")
    image = tmp_path / "question.png"
    Image.new("RGB", (160, 90), "white").save(image)
    destination = tmp_path / "progress.mp4"
    style = ProgressBarStyle(
        width=120,
        height=10,
        start_color="#FF0000",
        end_color="#00FF00",
        offsets=AnchorOffsets(top=70),
    )

    VideoComposer().compose(
        [VideoSegment(image_path=image, duration=0.5, progress_style=style)],
        destination,
        render=RenderSettings(fps=24),
        music=BackgroundMusicSettings(),
    )

    clip = moviepy.VideoFileClip(str(destination))
    try:
        start = clip.get_frame(0.0)
        end = clip.get_frame(0.45)
    finally:
        clip.close()
    assert start[75, 40].tolist() != end[75, 40].tolist()
    assert start[10, 10].tolist() == end[10, 10].tolist()
