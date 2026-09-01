import os
import stat
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from ai_vocab_video_generator.domain import (
    AnchorOffsets,
    CanvasSettings,
    GenerationRequest,
    MaterialFitMode,
    MaterialShape,
    TextElementStyle,
    VideoAspect,
    WordEntry,
)
from ai_vocab_video_generator.errors import RenderingError
from ai_vocab_video_generator.rendering.cards import CardLayers, CardRenderer


def _request(background: Path, material: Path | None = None) -> GenerationRequest:
    return GenerationRequest(
        entries=[WordEntry(english="serendipity", phonetic="/test/", chinese="意外发现")],
        background_image=background,
        local_materials=[material] if material else [],
    )


def test_answer_layers_keep_text_above_an_inserted_material(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (320, 240), "green").save(background)
    request = _request(background)
    request.canvas = CanvasSettings(aspect=VideoAspect.LANDSCAPE, width=320, height=240)
    request.material.enabled = False
    request.question.enabled = False
    request.english_text.font_size = 40
    request.english_text.offsets = AnchorOffsets(top=80)
    request.phonetic_text.enabled = False
    request.chinese_text.enabled = False
    base = tmp_path / "base.png"
    foreground = tmp_path / "foreground.png"

    layers = CardRenderer().render_answer_layers(
        request.entries[0], background, request, base, foreground
    )

    assert layers == CardLayers(base_path=base, foreground_path=foreground)
    with (
        Image.open(layers.base_path) as base_image,
        Image.open(layers.foreground_path) as foreground_image,
    ):
        assert base_image.mode == "RGB"
        assert base_image.getpixel((160, 120)) == (0, 128, 0)
        assert foreground_image.mode == "RGBA"
        assert foreground_image.getpixel((0, 0))[3] == 0
        alpha = foreground_image.getchannel("A")
        try:
            text_bounds = alpha.getbbox()
        finally:
            alpha.close()
        assert text_bounds is not None
        text_pixel = next(
            (x, y)
            for y in range(text_bounds[1], text_bounds[3])
            for x in range(text_bounds[0], text_bounds[2])
            if foreground_image.getpixel((x, y))[3] == 255
        )

        composited = base_image.convert("RGBA")
        material = Image.new("RGBA", composited.size, "blue")
        try:
            composited.alpha_composite(material)
            composited.alpha_composite(foreground_image)
            assert composited.getpixel(text_pixel) != (0, 0, 255, 255)
        finally:
            material.close()
            composited.close()


def test_answer_card_uses_custom_canvas_without_darkening_background(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (240, 120), "#336699").save(background)
    request = GenerationRequest.with_aspect_defaults(
        VideoAspect.LANDSCAPE,
        entries=[WordEntry(english="serendipity", phonetic="/test/", chinese="意外发现")],
        background_image=background,
    )
    request.material.enabled = False
    destination = tmp_path / "answer.png"

    CardRenderer().render_answer(request.entries[0], background, None, request, destination)

    with Image.open(destination) as image:
        assert image.size == (1920, 1080)
        assert image.getpixel((0, 0)) == (51, 102, 153)


def test_card_renderer_rejects_oversized_background_before_decoding(
    monkeypatch, tmp_path: Path
) -> None:
    class OversizedBackground:
        size = (10_000, 6_000)

        def __enter__(self) -> "OversizedBackground":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def convert(self, _mode: str) -> Image.Image:
            raise AssertionError("oversized background must not be decoded")

    monkeypatch.setattr(
        "ai_vocab_video_generator.rendering.cards.Image.open",
        lambda _path: OversizedBackground(),
    )
    background = tmp_path / "oversized.png"
    request = _request(background)

    with pytest.raises(RenderingError, match="background image"):
        CardRenderer().render_answer_layers(
            request.entries[0],
            background,
            request,
            tmp_path / "base.png",
            tmp_path / "foreground.png",
        )


def test_circle_material_masks_corners_and_keeps_center(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    material = tmp_path / "material.png"
    Image.new("RGB", (1080, 1920), "red").save(background)
    Image.new("RGB", (400, 400), "blue").save(material)
    request = _request(background, material)
    request.material.shape = MaterialShape.CIRCLE
    destination = tmp_path / "circle.png"

    CardRenderer().render_answer(request.entries[0], background, material, request, destination)

    with Image.open(destination) as image:
        assert image.getpixel((216, 384)) == (255, 0, 0)
        assert image.getpixel((540, 708)) == (0, 0, 255)


def test_material_fit_mode_keeps_contain_letterbox_transparent(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    material = tmp_path / "material.png"
    Image.new("RGB", (320, 480), "green").save(background)
    source = Image.new("RGB", (200, 100), "red")
    for x in range(100, 200):
        for y in range(100):
            source.putpixel((x, y), (0, 0, 255))
    source.save(material)
    source.close()
    request = _request(background, material)
    request.canvas = CanvasSettings(aspect=VideoAspect.PORTRAIT, width=320, height=480)
    request.material.width = 100
    request.material.height = 100
    request.material.fit_mode = MaterialFitMode.CONTAIN
    request.material.shape = MaterialShape.RECTANGLE
    request.material.offsets = AnchorOffsets(top=20)
    request.question.enabled = False
    request.english_text.enabled = False
    request.phonetic_text.enabled = False
    request.chinese_text.enabled = False
    destination = tmp_path / "contained.png"

    CardRenderer().render_answer(request.entries[0], background, material, request, destination)

    with Image.open(destination) as image:
        assert image.getpixel((160, 25)) == (0, 128, 0)
        assert image.getpixel((135, 70)) == (255, 0, 0)


def test_disabled_text_element_changes_no_pixels_in_its_region(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (1080, 1920), "white").save(background)
    request = _request(background)
    request.material.enabled = False
    request.question.enabled = False
    request.phonetic_text.enabled = False
    request.chinese_text.enabled = False
    enabled = tmp_path / "enabled.png"
    disabled = tmp_path / "disabled.png"

    CardRenderer().render_answer(request.entries[0], background, None, request, enabled)
    request.english_text.enabled = False
    CardRenderer().render_answer(request.entries[0], background, None, request, disabled)

    assert enabled.read_bytes() != disabled.read_bytes()
    with Image.open(disabled) as image:
        assert image.getpixel((540, 1100)) == (255, 255, 255)


def test_english_keeps_the_user_selected_font_size(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (1080, 1920), "white").save(background)
    request = _request(background)
    renderer = CardRenderer()

    assert renderer.effective_font_size("x" * 25, request.english_text) == 100
    assert renderer.effective_font_size("x" * 26, request.english_text) == 100


def test_base_stroke_value_keeps_the_visible_four_pixel_outer_outline(
    tmp_path: Path,
) -> None:
    background = tmp_path / "background.png"
    background_color = (152, 124, 249)
    Image.new("RGB", (320, 240), background_color).save(background)
    request = _request(background)
    request.canvas = CanvasSettings(aspect=VideoAspect.LANDSCAPE, width=320, height=240)
    request.material.enabled = False
    request.question.enabled = False
    request.english_text.font_size = 80
    request.english_text.weight = 0
    request.english_text.stroke_width = 1.5
    request.english_text.offsets = AnchorOffsets(top=60)
    request.phonetic_text.enabled = False
    request.chinese_text.enabled = False
    request.entries = [WordEntry(english="I")]
    destination = tmp_path / "outlined.png"

    CardRenderer().render_answer(request.entries[0], background, None, request, destination)

    with Image.open(destination) as image:
        pixels = image.convert("RGB")
        outer = [
            (x, y)
            for y in range(pixels.height)
            for x in range(pixels.width)
            if pixels.getpixel((x, y)) != background_color
        ]
        fill = [
            (x, y)
            for y in range(pixels.height)
            for x in range(pixels.width)
            if pixels.getpixel((x, y)) == (0, 0, 0)
        ]

    outer_box = (
        min(x for x, _ in outer),
        min(y for _, y in outer),
        max(x for x, _ in outer),
        max(y for _, y in outer),
    )
    fill_box = (
        min(x for x, _ in fill),
        min(y for _, y in fill),
        max(x for x, _ in fill),
        max(y for _, y in fill),
    )
    margins = (
        fill_box[0] - outer_box[0],
        fill_box[1] - outer_box[1],
        outer_box[2] - fill_box[2],
        outer_box[3] - fill_box[3],
    )
    assert min(margins) >= 4


def test_font_weight_expands_the_visible_fill_inside_the_outline(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (320, 240), (152, 124, 249)).save(background)

    def render_fill_box(weight: float, name: str) -> tuple[int, int, int, int]:
        request = _request(background)
        request.canvas = CanvasSettings(aspect=VideoAspect.LANDSCAPE, width=320, height=240)
        request.material.enabled = False
        request.question.enabled = False
        request.english_text.font_size = 80
        request.english_text.weight = weight
        request.english_text.stroke_width = 1.5
        request.english_text.offsets = AnchorOffsets(top=60)
        request.phonetic_text.enabled = False
        request.chinese_text.enabled = False
        request.entries = [WordEntry(english="I")]
        destination = tmp_path / name
        CardRenderer().render_answer(request.entries[0], background, None, request, destination)
        with Image.open(destination) as image:
            pixels = image.convert("RGB")
            black = [
                (x, y)
                for y in range(image.height)
                for x in range(image.width)
                if pixels.getpixel((x, y)) == (0, 0, 0)
            ]
        return (
            min(x for x, _ in black),
            min(y for _, y in black),
            max(x for x, _ in black),
            max(y for _, y in black),
        )

    regular = render_fill_box(0, "regular.png")
    weighted = render_fill_box(1, "weighted.png")

    assert weighted[0] < regular[0]
    assert weighted[1] < regular[1]
    assert weighted[2] > regular[2]
    assert weighted[3] > regular[3]


def test_long_english_sentence_asks_the_user_to_reduce_font_size(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (1080, 1920), "white").save(background)
    sentence = "Where is the nearest subway station?"
    request = _request(background)
    request.entries = [WordEntry(english=sentence)]
    request.material.enabled = False
    request.question.enabled = False
    request.english_text.fill_color = "#FFD54F"
    request.phonetic_text.enabled = False
    request.chinese_text.enabled = False
    destination = tmp_path / "long-sentence.png"

    with pytest.raises(RenderingError, match="reduce its font size"):
        CardRenderer().render_answer(request.entries[0], background, None, request, destination)

    assert not destination.exists()


def test_system_font_fallback_supports_ipa_before_rendering() -> None:
    text = "/ˈæp.əl/"

    font = CardRenderer._font(TextElementStyle(), 80, text)

    def signature(character: str) -> tuple[tuple[int, int], bytes]:
        mask = font.getmask(character, mode="L")
        return mask.size, bytes(mask)

    missing_glyphs = {signature("\uffff"), signature("\U0010ffff")}
    assert all(
        character.isspace() or signature(character) not in missing_glyphs for character in text
    )


def test_question_card_uses_question_style_and_material(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    material = tmp_path / "material.png"
    Image.new("RGB", (1080, 1920), "white").save(background)
    Image.new("RGB", (400, 400), "blue").save(material)
    request = _request(background, material)
    request.question.enabled = True
    destination = tmp_path / "question.png"

    CardRenderer().render_question(
        request.question_text, background, material, request, destination
    )

    with Image.open(destination) as image:
        assert image.size == (1080, 1920)
        assert image.getpixel((540, 708)) == (0, 0, 255)


def test_missing_custom_font_is_a_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="font"):
        TextElementStyle(font_path=tmp_path / "missing.ttf")


def test_custom_even_canvas_dimensions_are_rendered(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (320, 480), "white").save(background)
    request = _request(background)
    request.canvas = CanvasSettings(aspect=VideoAspect.PORTRAIT, width=320, height=480)
    request.material.enabled = False
    request.question.enabled = False
    request.english_text.enabled = False
    request.phonetic_text.enabled = False
    request.chinese_text.enabled = False
    request.question.enabled = False
    request.english_text.font_size = 30
    request.english_text.offsets = AnchorOffsets(top=30)
    request.phonetic_text.enabled = False
    request.chinese_text.enabled = False
    destination = tmp_path / "custom.png"

    CardRenderer().render_answer(request.entries[0], background, None, request, destination)

    with Image.open(destination) as image:
        assert image.size == (320, 480)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_rendered_card_is_owner_only(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (320, 480), "white").save(background)
    request = _request(background)
    request.canvas = CanvasSettings(aspect=VideoAspect.PORTRAIT, width=320, height=480)
    request.material.enabled = False
    request.question.enabled = False
    request.english_text.enabled = False
    request.phonetic_text.enabled = False
    request.chinese_text.enabled = False
    destination = tmp_path / "private" / "card.png"

    CardRenderer().render_answer(request.entries[0], background, None, request, destination)

    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
