"""Shared deterministic layout and progress drawing primitives."""

from PIL import Image, ImageColor, ImageDraw

from ai_vocab_video_generator.domain import AnchorOffsets, ProgressBarStyle


def resolve_position(
    canvas_size: tuple[int, int],
    element_size: tuple[int, int],
    offsets: AnchorOffsets,
) -> tuple[int, int]:
    return offsets.resolve(canvas_size, element_size)


def draw_progress(
    image: Image.Image,
    style: ProgressBarStyle,
    fraction: float,
) -> Image.Image:
    if not style.enabled:
        return image
    position = resolve_position(image.size, (style.width, style.height), style.offsets)
    left, top = position
    right = left + style.width
    bottom = top + style.height
    draw = ImageDraw.Draw(image)
    radius = max(0, min(style.height // 2, style.width // 2))
    draw.rounded_rectangle((left, top, right - 1, bottom - 1), radius=radius, fill="#CBD5E1")

    bounded = min(1.0, max(0.0, fraction))
    filled = round(style.width * bounded)
    if filled <= 0:
        return image
    start = ImageColor.getrgb(style.start_color)
    end = ImageColor.getrgb(style.end_color)
    gradient = Image.new("RGB", (filled, style.height))
    gradient_draw = ImageDraw.Draw(gradient)
    denominator = max(1, style.width - 1)
    for x in range(filled):
        ratio = x / denominator
        color = tuple(round(a + (b - a) * ratio) for a, b in zip(start, end, strict=True))
        gradient_draw.line((x, 0, x, style.height - 1), fill=color)

    mask = Image.new("L", (style.width, style.height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, style.width - 1, style.height - 1), radius=radius, fill=255
    )
    image.paste(gradient, (left, top), mask.crop((0, 0, filled, style.height)))
    gradient.close()
    mask.close()
    return image
