"""Shared image-frame fitting and masking for material media."""

from PIL import Image, ImageChops, ImageDraw, ImageOps

from ai_vocab_video_generator.domain import MaterialFitMode, MaterialShape


def fit_material_frame(
    source: Image.Image,
    size: tuple[int, int],
    mode: MaterialFitMode,
) -> Image.Image:
    """Fit a source image into a target-sized RGBA frame."""
    rgba = source.convert("RGBA")
    try:
        if mode is MaterialFitMode.CONTAIN:
            contained = ImageOps.contain(rgba, size, method=Image.Resampling.LANCZOS)
            try:
                frame = Image.new("RGBA", size)
                frame.alpha_composite(
                    contained,
                    ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
                )
                return frame
            finally:
                contained.close()
        if mode is MaterialFitMode.COVER:
            return ImageOps.fit(rgba, size, method=Image.Resampling.LANCZOS)
        return rgba.resize(size, Image.Resampling.LANCZOS)
    finally:
        rgba.close()


def apply_material_mask(source: Image.Image, shape: MaterialShape) -> Image.Image:
    """Return an RGBA material frame with the requested shape mask applied."""
    result = source.convert("RGBA")
    if shape is MaterialShape.RECTANGLE:
        return result
    mask = Image.new("L", result.size, 0)
    try:
        ImageDraw.Draw(mask).ellipse((0, 0, result.width - 1, result.height - 1), fill=255)
        alpha = result.getchannel("A")
        try:
            result.putalpha(ImageChops.multiply(alpha, mask))
        finally:
            alpha.close()
        return result
    finally:
        mask.close()
