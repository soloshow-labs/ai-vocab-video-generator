"""Deterministic local and remote still-image provider adapters."""

import json
import random
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from moviepy.editor import VideoFileClip  # type: ignore[import-untyped]
from PIL import Image, ImageDraw
from pydantic import SecretStr

from ai_vocab_video_generator.domain import MaterialAsset, MaterialKind, SelectionMode, VideoAspect
from ai_vocab_video_generator.errors import ProviderError, UploadSizeError
from ai_vocab_video_generator.media_limits import MAX_LOCAL_IMAGE_BYTES, MAX_LOCAL_VIDEO_BYTES
from ai_vocab_video_generator.private_fs import (
    copy_private_file,
    ensure_private_directory,
    mark_private_file,
    write_private_bytes,
)
from ai_vocab_video_generator.providers.base import ImageSelectionContext

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".webm"})
_MAX_REMOTE_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_REMOTE_IMAGE_PIXELS = 50_000_000
_MAX_VIDEO_DURATION_SECONDS = 300.0
_MAX_VIDEO_DIMENSION = 3840
_MAX_VIDEO_FPS = 60.0
_MAX_PROVIDER_JSON_BYTES = 1024 * 1024
_MAX_REMOTE_REDIRECTS = 3
_PEXELS_IMAGE_HOSTS = frozenset({"images.pexels.com"})
_PIXABAY_IMAGE_HOSTS = frozenset({"cdn.pixabay.com", "pixabay.com"})


def _default_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
    )


def _default_context() -> ImageSelectionContext:
    return ImageSelectionContext(
        entry_index=0,
        pool_size=1,
        mode=SelectionMode.SEQUENTIAL,
        seed=0,
    )


def _candidate_index(length: int, context: ImageSelectionContext) -> int:
    if length <= 0:
        raise ValueError("Candidate collection must not be empty.")
    if context.mode is SelectionMode.SEQUENTIAL:
        return context.entry_index % length
    return random.Random(context.seed + context.entry_index).randrange(length)


def seeded_video_start_offset(
    duration_seconds: float,
    *,
    seed: int,
    entry_index: int,
) -> float:
    """Return the pipeline's stable per-entry video starting position."""
    if duration_seconds <= 0:
        raise ValueError("Material video duration must be positive.")
    return random.Random(seed + entry_index).random() * duration_seconds


def extract_seeded_video_frame(
    source: Path,
    destination: Path,
    *,
    seed: int,
    entry_index: int,
) -> Path:
    """Decode the same video frame that begins a pipeline material overlay."""
    clip: VideoFileClip | None = None
    try:
        clip = VideoFileClip(str(source), audio=False)
        offset = seeded_video_start_offset(float(clip.duration), seed=seed, entry_index=entry_index)
        frame = clip.get_frame(offset)
        ensure_private_directory(destination.parent)
        image = Image.fromarray(frame)
        try:
            image.save(destination, format="PNG")
            mark_private_file(destination)
        finally:
            image.close()
    except (OSError, ValueError, IndexError) as exc:
        raise ProviderError(
            "The selected material video cannot be decoded.",
            diagnostic=type(exc).__name__,
        ) from exc
    finally:
        if clip is not None:
            clip.close()
    return destination


def probe_material(path: Path) -> MaterialAsset:
    """Verify an allowlisted material can be decoded and identify its media kind."""
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        image: Image.Image | None = None
        try:
            size = path.stat().st_size
            if size > MAX_LOCAL_IMAGE_BYTES:
                raise UploadSizeError(size, MAX_LOCAL_IMAGE_BYTES)
            image = Image.open(path)
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > _MAX_REMOTE_IMAGE_PIXELS:
                raise ValueError
            image.load()
        except UploadSizeError:
            raise
        except (OSError, ValueError) as exc:
            raise ProviderError(
                "The selected material image cannot be decoded.", diagnostic=type(exc).__name__
            ) from exc
        finally:
            if image is not None:
                image.close()
        return MaterialAsset(path=path, kind=MaterialKind.IMAGE)
    if suffix in _VIDEO_SUFFIXES:
        clip: VideoFileClip | None = None
        try:
            size = path.stat().st_size
            if size > MAX_LOCAL_VIDEO_BYTES:
                raise UploadSizeError(size, MAX_LOCAL_VIDEO_BYTES)
            clip = VideoFileClip(str(path), audio=False)
            width, height = (int(value) for value in clip.size)
            duration = float(clip.duration)
            fps = float(clip.fps)
            if (
                width <= 0
                or height <= 0
                or width > _MAX_VIDEO_DIMENSION
                or height > _MAX_VIDEO_DIMENSION
                or duration <= 0
                or duration > _MAX_VIDEO_DURATION_SECONDS
                or fps <= 0
                or fps > _MAX_VIDEO_FPS
            ):
                raise ValueError
            clip.get_frame(0.0)
        except UploadSizeError:
            raise
        except (OSError, ValueError, IndexError) as exc:
            raise ProviderError(
                "The selected material video cannot be decoded.", diagnostic=type(exc).__name__
            ) from exc
        finally:
            if clip is not None:
                clip.close()
        return MaterialAsset(path=path, kind=MaterialKind.VIDEO)
    raise ProviderError("The selected material file type is not supported.")


def _safe_provider_id(candidate: object, *, active_secret: str) -> str | None:
    if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
        source_id = str(candidate)
        if source_id != active_secret:
            return source_id
    return None


@dataclass(frozen=True, slots=True)
class RemoteImageCandidate:
    source_id: str | None
    image_url: str = field(repr=False)
    thumbnail_url: str = field(repr=False)


class _RemoteImageProvider:
    _client: httpx.Client
    _warnings: list[str]
    _hosts: frozenset[str]
    _name: str

    def search(
        self, query: str, aspect: VideoAspect, *, pool_size: int
    ) -> tuple[RemoteImageCandidate, ...]:
        raise NotImplementedError

    def download_candidate(
        self, candidate: RemoteImageCandidate, destination_stem: Path, *, thumbnail: bool = False
    ) -> MaterialAsset:
        try:
            result = _download(
                self._client,
                candidate.thumbnail_url if thumbnail else candidate.image_url,
                destination_stem,
                source_id=candidate.source_id,
                allowed_hosts=self._hosts,
            )
        except (httpx.HTTPError, httpx.InvalidURL, OSError, ValueError) as exc:
            error = ProviderError(
                f"No {self._name} image could be fetched.", diagnostic=type(exc).__name__
            )
        else:
            return result
        raise error

    def fetch(
        self,
        query: str,
        destination_stem: Path,
        aspect: VideoAspect,
        context: ImageSelectionContext | None = None,
    ) -> MaterialAsset:
        selection = context or _default_context()
        candidates = self.search(query, aspect, pool_size=selection.pool_size)
        if not candidates:
            self._warnings.append(f"No {self._name} result; used a neutral tile.")
            return _neutral_tile(destination_stem)
        return self.download_candidate(
            candidates[_candidate_index(len(candidates), selection)], destination_stem
        )

    def _candidate(
        self, source_id: str | None, image_url: str, thumbnail_url: str
    ) -> RemoteImageCandidate:
        _validate_remote_image_url(image_url, self._hosts)
        _validate_remote_image_url(thumbnail_url, self._hosts)
        return RemoteImageCandidate(source_id, image_url, thumbnail_url)


class PexelsImageProvider(_RemoteImageProvider):
    _hosts = _PEXELS_IMAGE_HOSTS
    _name = "Pexels"

    def __init__(self, api_key: SecretStr, *, client: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or _default_client()
        self._warnings: list[str] = []

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def check_connection(self) -> None:
        """Validate the key with a metadata-only search request."""
        try:
            payload = _request_json(
                self._client,
                "https://api.pexels.com/v1/search",
                params={"query": "test", "per_page": 1},
                headers={"Authorization": self._api_key.get_secret_value()},
            )
            if not isinstance(payload.get("photos"), list):
                raise ValueError("Pexels response did not contain a photo list.")
        except (httpx.HTTPError, httpx.InvalidURL, KeyError, TypeError, ValueError) as exc:
            error = _image_connection_error(exc)
        else:
            return
        raise error

    def search(
        self,
        query: str,
        aspect: VideoAspect,
        *,
        pool_size: int,
    ) -> tuple[RemoteImageCandidate, ...]:
        if not 1 <= pool_size <= 20:
            raise ValueError("Candidate count must be between 1 and 20.")
        orientation = "portrait" if aspect is VideoAspect.PORTRAIT else "landscape"
        try:
            payload = _request_json(
                self._client,
                "https://api.pexels.com/v1/search",
                params={
                    "query": query,
                    "orientation": orientation,
                    "per_page": pool_size,
                },
                headers={"Authorization": self._api_key.get_secret_value()},
            )
            photos: list[dict[str, Any]] = payload.get("photos", [])
            if not isinstance(photos, list) or any(not isinstance(photo, dict) for photo in photos):
                raise ValueError("Pexels response did not contain a valid photo list.")
            result = tuple(
                self._candidate(
                    _safe_provider_id(
                        photo.get("id"), active_secret=self._api_key.get_secret_value()
                    ),
                    str(photo["src"][orientation]),
                    str(photo["src"].get("medium", photo["src"][orientation])),
                )
                for photo in photos[:pool_size]
            )
        except (
            httpx.HTTPError,
            httpx.InvalidURL,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            error = ProviderError(
                "No Pexels image could be fetched.",
                diagnostic=type(exc).__name__,
            )
        else:
            return result
        raise error


class PixabayImageProvider(_RemoteImageProvider):
    _hosts = _PIXABAY_IMAGE_HOSTS
    _name = "Pixabay"

    def __init__(self, api_key: SecretStr, *, client: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or _default_client()
        self._warnings: list[str] = []

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def check_connection(self) -> None:
        """Validate the key with a metadata-only search request."""
        try:
            payload = _request_json(
                self._client,
                "https://pixabay.com/api/",
                params={
                    "key": self._api_key.get_secret_value(),
                    "q": "test",
                    "image_type": "photo",
                    "per_page": 3,
                    "safesearch": "true",
                },
            )
            if not isinstance(payload.get("hits"), list):
                raise ValueError("Pixabay response did not contain an image list.")
        except (httpx.HTTPError, httpx.InvalidURL, KeyError, TypeError, ValueError) as exc:
            error = _image_connection_error(exc)
        else:
            return
        raise error

    def search(
        self,
        query: str,
        aspect: VideoAspect,
        *,
        pool_size: int,
    ) -> tuple[RemoteImageCandidate, ...]:
        if not 1 <= pool_size <= 20:
            raise ValueError("Candidate count must be between 1 and 20.")
        orientation = "vertical" if aspect is VideoAspect.PORTRAIT else "horizontal"
        try:
            payload = _request_json(
                self._client,
                "https://pixabay.com/api/",
                params={
                    "key": self._api_key.get_secret_value(),
                    "q": query,
                    "orientation": orientation,
                    "image_type": "photo",
                    "per_page": max(3, pool_size),
                    "safesearch": "true",
                },
            )
            hits: list[dict[str, Any]] = payload.get("hits", [])
            if not isinstance(hits, list) or any(not isinstance(hit, dict) for hit in hits):
                raise ValueError("Pixabay response did not contain a valid image list.")
            result = tuple(
                self._candidate(
                    _safe_provider_id(
                        hit.get("id"), active_secret=self._api_key.get_secret_value()
                    ),
                    str(hit["largeImageURL"]),
                    str(hit.get("webformatURL", hit["largeImageURL"])),
                )
                for hit in hits[:pool_size]
            )
        except (
            httpx.HTTPError,
            httpx.InvalidURL,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            error = ProviderError(
                "No Pixabay image could be fetched.",
                diagnostic=type(exc).__name__,
            )
        else:
            return result
        raise error


class LocalImageProvider:
    def __init__(self, sources: Path | Sequence[Path]) -> None:
        self._sources: tuple[Path, ...]
        if isinstance(sources, Path):
            self._sources = (sources,)
        else:
            self._sources = tuple(sources)

    @property
    def warnings(self) -> tuple[str, ...]:
        return ()

    def fetch(
        self,
        query: str,
        destination_stem: Path,
        aspect: VideoAspect,
        context: ImageSelectionContext | None = None,
    ) -> MaterialAsset:
        del query, aspect
        if not self._sources:
            raise ProviderError("Select at least one local material image.")
        selection = context or _default_context()
        source = self._sources[_candidate_index(len(self._sources), selection)]
        if not source.is_file():
            raise ProviderError("A selected local material image does not exist.")
        asset = probe_material(source)
        destination = destination_stem.with_suffix(source.suffix)
        copy_private_file(source, destination)
        return MaterialAsset(path=destination, kind=asset.kind)


def _image_connection_error(exc: Exception) -> ProviderError:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {401, 403}:
        message = "The image provider rejected the API key. Check the configured credential."
    elif isinstance(exc, httpx.TimeoutException):
        message = "The image provider connection test timed out. Try again and check the network."
    elif isinstance(exc, httpx.ConnectError):
        message = "The image provider is unavailable. Check the network and try again."
    else:
        message = "The image provider returned an invalid response."
    return ProviderError(message, diagnostic=type(exc).__name__)


def _download(
    client: httpx.Client,
    url: str,
    destination_stem: Path,
    *,
    source_id: str | None,
    allowed_hosts: frozenset[str],
) -> MaterialAsset:
    current_url = url
    chunks: list[bytes] = []
    for redirect_count in range(_MAX_REMOTE_REDIRECTS + 1):
        _validate_remote_image_url(current_url, allowed_hosts)
        with client.stream("GET", current_url, follow_redirects=False) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if location is None or redirect_count >= _MAX_REMOTE_REDIRECTS:
                    raise ValueError("Remote material redirect is invalid.")
                current_url = urljoin(str(response.url), location)
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
            if not content_type.startswith("image/"):
                raise ValueError("Remote material response must be an image.")
            byte_count = 0
            for chunk in response.iter_bytes():
                byte_count += len(chunk)
                if byte_count > _MAX_REMOTE_IMAGE_BYTES:
                    raise ValueError("Remote material image exceeds the size limit.")
                chunks.append(chunk)
            break
    else:
        raise ValueError("Remote material redirect is invalid.")
    contents = b"".join(chunks)
    if not contents:
        raise ValueError("Remote material response was empty.")
    try:
        with Image.open(BytesIO(contents)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > _MAX_REMOTE_IMAGE_PIXELS:
                raise ValueError("Remote material image dimensions are not supported.")
            image.load()
    except OSError as exc:
        raise ValueError("Remote material response could not be decoded.") from exc
    destination = destination_stem.with_suffix(".jpg")
    ensure_private_directory(destination.parent)
    staged = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    try:
        write_private_bytes(staged, contents)
        staged.replace(destination)
        mark_private_file(destination)
    finally:
        staged.unlink(missing_ok=True)
    return MaterialAsset(path=destination, kind=MaterialKind.IMAGE, source_id=source_id)


def _request_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    with client.stream(
        "GET", url, params=params, headers=headers, follow_redirects=False
    ) as response:
        response.raise_for_status()
        declared = response.headers.get("content-length")
        if declared is not None and int(declared) > _MAX_PROVIDER_JSON_BYTES:
            raise ValueError("Provider metadata exceeds the size limit.")
        chunks: list[bytes] = []
        byte_count = 0
        for chunk in response.iter_bytes():
            byte_count += len(chunk)
            if byte_count > _MAX_PROVIDER_JSON_BYTES:
                raise ValueError("Provider metadata exceeds the size limit.")
            chunks.append(chunk)
    payload = json.loads(b"".join(chunks))
    if not isinstance(payload, dict):
        raise ValueError("Provider metadata must be an object.")
    return payload


def _validate_remote_image_url(url: str, allowed_hosts: frozenset[str]) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
    ):
        raise ValueError("Remote material URL is not an approved HTTPS image host.")


def _neutral_tile(destination_stem: Path) -> MaterialAsset:
    destination = destination_stem.with_suffix(".jpg")
    ensure_private_directory(destination.parent)
    image = Image.new("RGB", (512, 512), "#E2E8F0")
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), "Material")
    x = max(12, (512 - (box[2] - box[0])) // 2)
    y = (512 - (box[3] - box[1])) // 2
    draw.text((x, y), "Material", fill="#475569")
    image.save(destination, format="JPEG", quality=90)
    image.close()
    mark_private_file(destination)
    return MaterialAsset(path=destination, kind=MaterialKind.IMAGE)
