"""Layered deterministic Pillow vocabulary-card rendering."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ai_vocab_video_generator.domain import (
    GenerationRequest,
    TextElementStyle,
    WordEntry,
)
from ai_vocab_video_generator.errors import RenderingError
from ai_vocab_video_generator.private_fs import ensure_private_directory, mark_private_file
from ai_vocab_video_generator.rendering.layout import resolve_position
from ai_vocab_video_generator.rendering.media import apply_material_mask, fit_material_frame

_BASE_OUTLINE_RADIUS = 4
_MAX_BACKGROUND_PIXELS = 50_000_000


@dataclass(frozen=True, slots=True)
class CardLayers:
    base_path: Path
    foreground_path: Path


class CardRenderer:
    """Render question and answer cards without bundled media."""

    def render_answer(
        self,
        entry: WordEntry,
        background: Path | None,
        material: Path | None,
        request: GenerationRequest,
        destination: Path,
    ) -> Path:
        base_path, foreground_path = self._temporary_layer_paths(destination)
        try:
            layers = self.render_answer_layers(
                entry,
                background,
                request,
                base_path,
                foreground_path,
            )
            return self._compose_still(layers, material, request, destination)
        finally:
            base_path.unlink(missing_ok=True)
            foreground_path.unlink(missing_ok=True)

    def render_question(
        self,
        question: str,
        background: Path | None,
        material: Path | None,
        request: GenerationRequest,
        destination: Path,
    ) -> Path:
        base_path, foreground_path = self._temporary_layer_paths(destination)
        try:
            layers = self.render_question_layers(
                question,
                background,
                request,
                base_path,
                foreground_path,
            )
            return self._compose_still(layers, material, request, destination)
        finally:
            base_path.unlink(missing_ok=True)
            foreground_path.unlink(missing_ok=True)

    def render_answer_layers(
        self,
        entry: WordEntry,
        background: Path | None,
        request: GenerationRequest,
        base_destination: Path,
        foreground_destination: Path,
    ) -> CardLayers:
        canvas = self._canvas(background, request)
        foreground = Image.new("RGBA", canvas.size)
        try:
            if request.question.enabled and request.question_text:
                self._draw_text(foreground, request.question_text, request.question)
            if request.english_text.enabled and entry.english:
                size = self.effective_font_size(entry.english, request.english_text)
                self._draw_text(foreground, entry.english, request.english_text, font_size=size)
            if request.phonetic_text.enabled and entry.phonetic:
                self._draw_text(foreground, entry.phonetic, request.phonetic_text)
            if request.chinese_text.enabled and entry.chinese:
                self._draw_text(foreground, entry.chinese, request.chinese_text)
            self._save(canvas, base_destination)
            self._save(foreground, foreground_destination)
            return CardLayers(base_destination, foreground_destination)
        finally:
            foreground.close()
            canvas.close()

    def render_question_layers(
        self,
        question: str,
        background: Path | None,
        request: GenerationRequest,
        base_destination: Path,
        foreground_destination: Path,
    ) -> CardLayers:
        canvas = self._canvas(background, request)
        foreground = Image.new("RGBA", canvas.size)
        try:
            if request.question.enabled and question:
                self._draw_text(foreground, question, request.question)
            self._save(canvas, base_destination)
            self._save(foreground, foreground_destination)
            return CardLayers(base_destination, foreground_destination)
        finally:
            foreground.close()
            canvas.close()

    def _compose_still(
        self,
        layers: CardLayers,
        material: Path | None,
        request: GenerationRequest,
        destination: Path,
    ) -> Path:
        try:
            with (
                Image.open(layers.base_path) as base,
                Image.open(layers.foreground_path) as foreground,
            ):
                canvas = base.convert("RGB")
                try:
                    self._place_material(canvas, material, request)
                    composed = canvas.convert("RGBA")
                    try:
                        foreground_layer = foreground.convert("RGBA")
                        try:
                            composed.alpha_composite(foreground_layer)
                        finally:
                            foreground_layer.close()
                        result = composed.convert("RGB")
                        try:
                            return self._save(result, destination)
                        finally:
                            result.close()
                    finally:
                        composed.close()
                finally:
                    canvas.close()
        except (OSError, ValueError) as exc:
            raise RenderingError("Unable to read rendered card layers.") from exc

    @staticmethod
    def _temporary_layer_paths(destination: Path) -> tuple[Path, Path]:
        return (
            destination.with_name(f".{destination.name}.base.png"),
            destination.with_name(f".{destination.name}.foreground.png"),
        )

    @staticmethod
    def effective_font_size(text: str, style: TextElementStyle) -> int:
        return style.font_size

    @staticmethod
    def _canvas(background: Path | None, request: GenerationRequest) -> Image.Image:
        size = (request.canvas.width, request.canvas.height)
        if background is None:
            return Image.new("RGB", size, "#F1F5F9")
        try:
            with Image.open(background) as source:
                width, height = source.size
                if width <= 0 or height <= 0 or width * height > _MAX_BACKGROUND_PIXELS:
                    raise ValueError("Background image dimensions exceed the safety limit.")
                return ImageOps.fit(source.convert("RGB"), size, method=Image.Resampling.LANCZOS)
        except (OSError, ValueError) as exc:
            raise RenderingError("Unable to read the selected background image.") from exc

    @staticmethod
    def _place_material(
        canvas: Image.Image,
        material: Path | None,
        request: GenerationRequest,
    ) -> None:
        style = request.material
        if not style.enabled or material is None:
            return
        try:
            with Image.open(material) as source:
                fitted = fit_material_frame(
                    source,
                    (style.width, style.height),
                    style.fit_mode,
                )
        except (OSError, ValueError) as exc:
            raise RenderingError("Unable to read a selected material image.") from exc
        try:
            layer = apply_material_mask(fitted, style.shape)
        finally:
            fitted.close()
        try:
            position = resolve_position(canvas.size, layer.size, style.offsets)
            canvas.paste(layer, position, layer)
        finally:
            layer.close()

    def _draw_text(
        self,
        canvas: Image.Image,
        text: str,
        style: TextElementStyle,
        *,
        font_size: int | None = None,
    ) -> None:
        font = self._font(style, font_size or style.font_size, text)
        draw = ImageDraw.Draw(canvas)
        # Keep a four-pixel base outline so low user-selected widths remain legible.
        stroke_width = _BASE_OUTLINE_RADIUS + int(style.stroke_width)
        box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        width = round(box[2] - box[0])
        height = round(box[3] - box[1])
        try:
            x, y = resolve_position(canvas.size, (width, height), style.offsets)
        except ValueError as exc:
            raise RenderingError(
                "Text does not fit inside the selected layout; reduce its font size "
                "or adjust its position."
            ) from exc
        origin = (x - box[0], y - box[1])
        draw.text(
            origin,
            text,
            font=font,
            fill=style.fill_color,
            stroke_width=stroke_width,
            stroke_fill=style.stroke_color,
        )
        weight = round(style.weight)
        if weight > 0:
            for dx in range(-weight, weight + 1):
                for dy in range(-weight, weight + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text(
                        (origin[0] + dx, origin[1] + dy),
                        text,
                        font=font,
                        fill=style.fill_color,
                    )
        draw.text(
            origin,
            text,
            font=font,
            fill=style.fill_color,
        )

    @staticmethod
    def _font(
        style: TextElementStyle,
        size: int,
        text: str = "",
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if style.verified_font_bytes is not None:
            try:
                return ImageFont.truetype(BytesIO(style.verified_font_bytes), size=size)
            except OSError as exc:
                raise RenderingError("Unable to load the verified custom font.") from exc
        if style.font_path is not None:
            try:
                return ImageFont.truetype(str(style.font_path), size=size)
            except OSError as exc:
                raise RenderingError("Unable to load the selected custom font.") from exc
        candidates = (
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/arialuni.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        )
        first_available: ImageFont.FreeTypeFont | None = None
        for candidate in candidates:
            if candidate.is_file():
                try:
                    font = ImageFont.truetype(str(candidate), size=size)
                except OSError:
                    continue
                if first_available is None:
                    first_available = font
                if CardRenderer._font_supports_text(font, text):
                    return font
        if first_available is not None:
            return first_available
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _font_supports_text(font: ImageFont.FreeTypeFont, text: str) -> bool:
        def signature(character: str) -> tuple[tuple[int, int], bytes]:
            mask = font.getmask(character, mode="L")
            return mask.size, bytes(mask)

        missing_glyphs = {signature("\uffff"), signature("\U0010ffff")}
        return all(
            signature(character) not in missing_glyphs
            for character in set(text)
            if not character.isspace()
        )

    @staticmethod
    def _save(canvas: Image.Image, destination: Path) -> Path:
        ensure_private_directory(destination.parent)
        try:
            canvas.save(destination, format="PNG", optimize=False)
            mark_private_file(destination)
        except OSError as exc:
            raise RenderingError("Unable to save the rendered card.") from exc
        return destination
