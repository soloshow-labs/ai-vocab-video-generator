import pytest
from PIL import Image, ImageColor

from ai_vocab_video_generator.domain import AnchorOffsets, ProgressBarStyle
from ai_vocab_video_generator.rendering.layout import draw_progress, resolve_position


@pytest.mark.parametrize(
    ("offsets", "want"),
    [
        (AnchorOffsets(), (450, 910)),
        (AnchorOffsets(left=20, top=30), (20, 30)),
        (AnchorOffsets(right=40, bottom=50), (860, 1770)),
        (AnchorOffsets(left=20, right=40, top=30, bottom=50), (440, 900)),
    ],
)
def test_resolve_position_honors_every_offset(
    offsets: AnchorOffsets, want: tuple[int, int]
) -> None:
    assert resolve_position((1080, 1920), (180, 100), offsets) == want


def test_progress_gradient_changes_with_elapsed_fraction() -> None:
    style = ProgressBarStyle()
    empty = draw_progress(Image.new("RGB", (1080, 1920), "white"), style, 0.0)
    half = draw_progress(Image.new("RGB", (1080, 1920), "white"), style, 0.5)

    assert empty.getpixel((300, 1430)) != half.getpixel((300, 1430))
    assert half.getpixel((162, 1430)) == ImageColor.getrgb("#FFA500")


def test_disabled_progress_does_not_modify_the_image() -> None:
    source = Image.new("RGB", (1080, 1920), "white")
    result = draw_progress(source.copy(), ProgressBarStyle(enabled=False), 0.75)

    assert result.tobytes() == source.tobytes()
