from PIL import Image

from ai_vocab_video_generator.domain import MaterialFitMode, MaterialShape
from ai_vocab_video_generator.rendering.media import (
    apply_material_mask,
    fit_material_frame,
)


def _split_source() -> Image.Image:
    source = Image.new("RGB", (200, 100), "red")
    for x in range(100, 200):
        for y in range(100):
            source.putpixel((x, y), (0, 0, 255))
    return source


def test_fit_material_frame_uses_requested_pixel_semantics() -> None:
    source = _split_source()
    try:
        contained = fit_material_frame(source, (100, 100), MaterialFitMode.CONTAIN)
        covered = fit_material_frame(source, (100, 100), MaterialFitMode.COVER)
        stretched = fit_material_frame(source, (100, 100), MaterialFitMode.STRETCH)
        try:
            assert contained.getpixel((50, 5))[3] == 0
            assert contained.getpixel((25, 50))[:3] == (255, 0, 0)
            assert covered.getpixel((0, 50))[:3] == (255, 0, 0)
            assert covered.getpixel((99, 50))[:3] == (0, 0, 255)
            assert stretched.size == (100, 100)
        finally:
            contained.close()
            covered.close()
            stretched.close()
    finally:
        source.close()


def test_circle_mask_makes_fitted_corners_transparent() -> None:
    source = _split_source()
    try:
        fitted = fit_material_frame(source, (100, 100), MaterialFitMode.COVER)
        try:
            masked = apply_material_mask(fitted, MaterialShape.CIRCLE)
            try:
                assert masked.getpixel((0, 0))[3] == 0
                assert masked.getpixel((50, 50))[3] == 255
            finally:
                masked.close()
        finally:
            fitted.close()
    finally:
        source.close()
