import traceback
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pytest
from moviepy.editor import ColorClip
from PIL import Image
from pydantic import SecretStr

from ai_vocab_video_generator.domain import MaterialKind, SelectionMode, VideoAspect
from ai_vocab_video_generator.errors import ProviderError
from ai_vocab_video_generator.providers.base import ImageSelectionContext
from ai_vocab_video_generator.providers.images import (
    LocalImageProvider,
    PexelsImageProvider,
    PixabayImageProvider,
)


def _image(path: Path, color: str) -> Path:
    Image.new("RGB", (12, 12), color).save(path)
    return path


def _image_bytes(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 12), color).save(output, format="JPEG")
    return output.getvalue()


def _video(path: Path) -> Path:
    clip = ColorClip((16, 16), color=(255, 0, 0), duration=0.2)
    try:
        clip.write_videofile(str(path), fps=10, audio=False, logger=None)
    finally:
        clip.close()
    return path


@pytest.mark.parametrize("provider_type", [PexelsImageProvider, PixabayImageProvider])
def test_candidate_search_is_metadata_only_and_downloads_exact_selection(
    tmp_path: Path, provider_type: type
) -> None:
    requests: list[httpx.Request] = []
    pexels = provider_type is PexelsImageProvider
    host = "images.pexels.com" if pexels else "cdn.pixabay.com"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path in {"/v1/search", "/api/"}:
            photos = [
                {
                    "id": i,
                    "src": {
                        "portrait": f"https://{host}/{i}.jpg",
                        "medium": f"https://{host}/{i}-thumb.jpg",
                    },
                }
                for i in range(1, 5)
            ]
            hits = [
                {
                    "id": i,
                    "largeImageURL": f"https://{host}/{i}.jpg",
                    "webformatURL": f"https://{host}/{i}-thumb.jpg",
                }
                for i in range(1, 5)
            ]
            return httpx.Response(200, json={"photos": photos} if pexels else {"hits": hits})
        return httpx.Response(200, content=_image_bytes(), headers={"content-type": "image/jpeg"})

    provider = provider_type(
        SecretStr("candidate-secret"), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    search = getattr(provider, "search", None)
    assert callable(search), "Remote providers must expose all candidates without downloading them"
    candidates = search("apple", VideoAspect.PORTRAIT, pool_size=3)
    assert len(candidates) == 3
    assert len(requests) == 1
    thumbnail = provider.download_candidate(candidates[1], tmp_path / "thumb", thumbnail=True)
    selected = provider.download_candidate(candidates[1], tmp_path / "selected")
    assert thumbnail.path.is_file()
    assert selected.source_id == "2"
    assert [request.url.path for request in requests[1:]] == ["/2-thumb.jpg", "/2.jpg"]
    assert all("authorization" not in request.headers for request in requests[1:])
    assert all("candidate-secret" not in str(request.url) for request in requests[1:])


@pytest.mark.parametrize("provider_type", [PexelsImageProvider, PixabayImageProvider])
@pytest.mark.parametrize("items", [None, {}, [None], ["unexpected"]])
def test_candidate_search_reports_malformed_lists_as_provider_errors(
    provider_type: type, items
) -> None:
    payload = {"photos" if provider_type is PexelsImageProvider else "hits": items}
    provider = provider_type(
        SecretStr("secret"),
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
        ),
    )
    with pytest.raises(ProviderError):
        provider.search("apple", VideoAspect.PORTRAIT, pool_size=8)


def test_candidate_search_rejects_untrusted_thumbnail_hosts() -> None:
    provider = PexelsImageProvider(
        SecretStr("secret"),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "photos": [
                            {
                                "id": 1,
                                "src": {
                                    "portrait": "https://images.pexels.com/1.jpg",
                                    "medium": "http://127.0.0.1/private",
                                },
                            }
                        ]
                    },
                )
            )
        ),
    )
    search = getattr(provider, "search", None)
    assert callable(search), "Candidate URLs must use the existing safe download policy"
    with pytest.raises(ProviderError):
        search("apple", VideoAspect.PORTRAIT, pool_size=8)


def test_pexels_connection_check_validates_search_without_downloading_an_image() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"photos": []})

    provider = PexelsImageProvider(
        SecretStr("pexels-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    provider.check_connection()

    assert len(requests) == 1
    assert requests[0].url == "https://api.pexels.com/v1/search?query=test&per_page=1"
    assert requests[0].headers["authorization"] == "pexels-secret"


def test_pixabay_connection_check_validates_search_without_downloading_an_image() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"hits": []})

    provider = PixabayImageProvider(
        SecretStr("pixabay-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    provider.check_connection()

    assert len(requests) == 1
    assert requests[0].url.params["key"] == "pixabay-secret"
    assert requests[0].url.params["q"] == "test"
    assert requests[0].url.params["per_page"] == "3"


@pytest.mark.parametrize("provider_type", [PexelsImageProvider, PixabayImageProvider])
def test_remote_connection_check_reports_a_rejected_key(provider_type: type) -> None:
    provider = provider_type(
        SecretStr("rejected-secret"),
        client=httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(401))),
    )

    with pytest.raises(ProviderError, match="rejected the API key"):
        provider.check_connection()


def test_pexels_fetches_portrait_image_without_putting_key_in_url(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.pexels.com":
            return httpx.Response(
                200,
                json={
                    "photos": [
                        {
                            "id": 123,
                            "src": {"portrait": "https://images.pexels.com/apple.jpg"},
                        }
                    ]
                },
            )
        return httpx.Response(200, content=_image_bytes(), headers={"content-type": "image/jpeg"})

    provider = PexelsImageProvider(
        SecretStr("pexels-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    destination = tmp_path / "apple"

    result = provider.fetch(
        "apple",
        destination,
        VideoAspect.PORTRAIT,
        ImageSelectionContext(
            entry_index=0,
            pool_size=5,
            mode=SelectionMode.SEQUENTIAL,
            seed=7,
        ),
    )

    assert result.kind is MaterialKind.IMAGE
    assert result.path == tmp_path / "apple.jpg"
    assert result.source_id == "123"
    with Image.open(result.path) as image:
        assert image.size == (12, 12)
    assert "pexels-secret" not in str(requests[0].url)
    assert requests[0].headers["authorization"] == "pexels-secret"
    assert requests[0].url.params["per_page"] == "5"


def test_pixabay_fetches_large_image(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "pixabay.com":
            assert request.url.params["key"] == "pixabay-secret"
            return httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "id": 456,
                            "largeImageURL": "https://cdn.pixabay.com/banana.jpg",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            content=_image_bytes("yellow"),
            headers={"content-type": "image/jpeg"},
        )

    provider = PixabayImageProvider(
        SecretStr("pixabay-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.fetch(
        "banana",
        tmp_path / "banana",
        VideoAspect.LANDSCAPE,
        ImageSelectionContext(
            entry_index=0,
            pool_size=3,
            mode=SelectionMode.SEQUENTIAL,
            seed=11,
        ),
    )

    assert result.kind is MaterialKind.IMAGE
    assert result.path == tmp_path / "banana.jpg"
    assert result.source_id == "456"
    with Image.open(result.path) as image:
        assert image.size == (12, 12)


@pytest.mark.parametrize(
    ("provider_type", "api_host", "credential", "payload"),
    [
        (
            PexelsImageProvider,
            "api.pexels.com",
            "pexels-reflected-secret",
            {
                "photos": [
                    {
                        "id": "pexels-reflected-secret",
                        "src": {"portrait": "https://images.pexels.com/reflected.jpg"},
                    }
                ]
            },
        ),
        (
            PixabayImageProvider,
            "pixabay.com",
            "pixabay-reflected-secret",
            {
                "hits": [
                    {
                        "id": "pixabay-reflected-secret",
                        "largeImageURL": "https://cdn.pixabay.com/reflected.jpg",
                    }
                ]
            },
        ),
        (
            PexelsImageProvider,
            "api.pexels.com",
            "12345678",
            {
                "photos": [
                    {
                        "id": 12345678,
                        "src": {"portrait": "https://images.pexels.com/numeric.jpg"},
                    }
                ]
            },
        ),
    ],
)
def test_remote_provider_does_not_expose_a_reflected_key_as_source_id(
    tmp_path: Path,
    provider_type: type,
    api_host: str,
    credential: str,
    payload: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == api_host:
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200,
            content=_image_bytes(),
            headers={"content-type": "image/jpeg"},
        )

    provider = provider_type(
        SecretStr(credential),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.fetch(
        "apple",
        tmp_path / "reflected",
        VideoAspect.PORTRAIT,
        ImageSelectionContext(
            entry_index=0,
            pool_size=3,
            mode=SelectionMode.SEQUENTIAL,
            seed=5,
        ),
    )

    assert result.source_id is None
    assert credential not in repr(result)


@pytest.mark.parametrize(
    ("pool_size", "mode", "entry_index", "expected_ids"),
    [
        (1, SelectionMode.SEQUENTIAL, 4, {"1"}),
        (2, SelectionMode.SEQUENTIAL, 5, {"1", "2"}),
        (2, SelectionMode.RANDOM, 9, {"1", "2"}),
    ],
)
def test_pixabay_slices_returned_hits_to_configured_pool(
    tmp_path: Path,
    pool_size: int,
    mode: SelectionMode,
    entry_index: int,
    expected_ids: set[str],
) -> None:
    requested_per_page: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "pixabay.com":
            requested_per_page.append(request.url.params["per_page"])
            return httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "id": index,
                            "largeImageURL": f"https://cdn.pixabay.com/{index}.jpg",
                        }
                        for index in range(1, 6)
                    ]
                },
            )
        return httpx.Response(
            200,
            content=_image_bytes(),
            headers={"content-type": "image/jpeg"},
        )

    provider = PixabayImageProvider(
        SecretStr("pixabay-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = provider.fetch(
        "apple",
        tmp_path / "apple",
        VideoAspect.PORTRAIT,
        ImageSelectionContext(
            entry_index=entry_index,
            pool_size=pool_size,
            mode=mode,
            seed=17,
        ),
    )

    assert requested_per_page == ["3"]
    assert result.source_id in expected_ids


def test_remote_download_rejects_oversized_stream_before_publication(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.pexels.com":
            return httpx.Response(
                200,
                json={
                    "photos": [
                        {
                            "id": 1,
                            "src": {"portrait": "https://images.pexels.com/large.jpg"},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            content=b"x" * (10 * 1024 * 1024 + 1),
            headers={"content-type": "image/jpeg"},
        )

    provider = PexelsImageProvider(
        SecretStr("pexels-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="No Pexels image"):
        provider.fetch("apple", tmp_path / "material", VideoAspect.PORTRAIT)

    assert not (tmp_path / "material.jpg").exists()


def test_remote_download_rejects_https_redirect_to_http(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "pixabay.com":
            return httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "id": 1,
                            "largeImageURL": "https://cdn.pixabay.com/material.jpg",
                        }
                    ]
                },
            )
        if request.url.scheme == "https":
            return httpx.Response(
                302,
                headers={"location": "http://cdn.pixabay.com/material.jpg"},
            )
        return httpx.Response(
            200,
            content=_image_bytes(),
            headers={"content-type": "image/jpeg"},
        )

    provider = PixabayImageProvider(
        SecretStr("pixabay-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
    )

    with pytest.raises(ProviderError, match="No Pixabay image"):
        provider.fetch("apple", tmp_path / "material", VideoAspect.PORTRAIT)

    assert not (tmp_path / "material.jpg").exists()


def test_remote_download_rejects_private_redirect_before_issuing_the_request(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "pixabay.com":
            return httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "id": 1,
                            "largeImageURL": "https://cdn.pixabay.com/material.jpg",
                        }
                    ]
                },
            )
        if request.url.host == "cdn.pixabay.com":
            return httpx.Response(
                302,
                headers={"location": "https://127.0.0.1/private.jpg"},
            )
        return httpx.Response(
            200,
            content=_image_bytes(),
            headers={"content-type": "image/jpeg"},
        )

    provider = PixabayImageProvider(
        SecretStr("pixabay-secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
    )

    with pytest.raises(ProviderError, match="No Pixabay image"):
        provider.fetch("apple", tmp_path / "material", VideoAspect.PORTRAIT)

    assert "127.0.0.1" not in [request.url.host for request in requests]
    assert not (tmp_path / "material.jpg").exists()


def test_remote_provider_rejects_oversized_metadata_response(tmp_path: Path) -> None:
    oversized = b'{"photos":[],' + b'"padding":"' + b"x" * (1024 * 1024) + b'"}'
    provider = PexelsImageProvider(
        SecretStr("pexels-secret"),
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=oversized))
        ),
    )

    with pytest.raises(ProviderError, match="No Pexels image"):
        provider.fetch("apple", tmp_path / "material", VideoAspect.PORTRAIT)

    assert not (tmp_path / "material.jpg").exists()


def test_local_provider_returns_validated_image_and_video_assets(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "apple.png", "red")
    video_path = _video(tmp_path / "apple.mp4")

    image_asset = LocalImageProvider(image_path).fetch(
        "apple", tmp_path / "out", VideoAspect.PORTRAIT
    )
    video_asset = LocalImageProvider(video_path).fetch(
        "apple", tmp_path / "video", VideoAspect.PORTRAIT
    )

    assert image_asset.kind is MaterialKind.IMAGE
    assert image_asset.path.suffix == ".png"
    assert video_asset.kind is MaterialKind.VIDEO
    assert video_asset.path.suffix == ".mp4"


@pytest.mark.parametrize(
    ("filename", "contents"),
    [("unsupported.gif", b"GIF89a"), ("broken.png", b"not an image")],
)
def test_local_provider_rejects_unsupported_or_undecodable_material(
    tmp_path: Path,
    filename: str,
    contents: bytes,
) -> None:
    source = tmp_path / filename
    source.write_bytes(contents)

    with pytest.raises(ProviderError) as error:
        LocalImageProvider(source).fetch("apple", tmp_path / "out", VideoAspect.PORTRAIT)

    assert str(source) not in error.value.safe_message


def test_local_provider_rejects_an_image_over_the_decoded_pixel_budget(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pixel-bomb.png"
    with Image.new("1", (10_000, 6_000), 1) as image:
        image.save(source)

    with pytest.raises(ProviderError, match="cannot be decoded"):
        LocalImageProvider(source).fetch("apple", tmp_path / "out", VideoAspect.PORTRAIT)


def test_local_provider_rejects_an_overlong_video(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "overlong.mp4"
    source.write_bytes(b"bounded-video-fixture")

    class OverlongClip:
        duration = 301.0
        fps = 24.0
        size = (1920, 1080)

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_frame(self, _time: float) -> np.ndarray:
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    monkeypatch.setattr("ai_vocab_video_generator.providers.images.VideoFileClip", OverlongClip)

    with pytest.raises(ProviderError, match="cannot be decoded"):
        LocalImageProvider(source).fetch("apple", tmp_path / "out", VideoAspect.PORTRAIT)


def test_local_random_selection_is_stable_per_entry(tmp_path: Path) -> None:
    sources = [
        _image(tmp_path / "red.png", "red"),
        _image(tmp_path / "green.png", "green"),
        _image(tmp_path / "blue.png", "blue"),
    ]
    provider = LocalImageProvider(sources)
    context = ImageSelectionContext(
        entry_index=4,
        pool_size=3,
        mode=SelectionMode.RANDOM,
        seed=71,
    )

    first = provider.fetch("apple", tmp_path / "first", VideoAspect.PORTRAIT, context)
    second = provider.fetch("apple", tmp_path / "second", VideoAspect.PORTRAIT, context)

    assert first.path.read_bytes() == second.path.read_bytes()


def test_local_sequential_selection_cycles_without_off_by_one(tmp_path: Path) -> None:
    sources = [_image(tmp_path / "red.png", "red"), _image(tmp_path / "blue.png", "blue")]
    provider = LocalImageProvider(sources)

    selected = [
        provider.fetch(
            "word",
            tmp_path / f"selected-{index}",
            VideoAspect.PORTRAIT,
            ImageSelectionContext(
                entry_index=index,
                pool_size=2,
                mode=SelectionMode.SEQUENTIAL,
                seed=5,
            ),
        ).path.read_bytes()
        for index in range(3)
    ]

    assert selected == [sources[0].read_bytes(), sources[1].read_bytes(), sources[0].read_bytes()]


def test_empty_remote_result_generates_a_neutral_fallback(tmp_path: Path) -> None:
    query = "personal-secret-query"
    provider = PexelsImageProvider(
        SecretStr("pexels-secret"),
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"photos": []}))
        ),
    )
    destination = tmp_path / "fallback"

    result = provider.fetch(
        query,
        destination,
        VideoAspect.PORTRAIT,
        ImageSelectionContext(
            entry_index=0,
            pool_size=5,
            mode=SelectionMode.RANDOM,
            seed=3,
        ),
    )

    assert result.kind is MaterialKind.IMAGE
    assert result.path == tmp_path / "fallback.jpg"
    assert result.source_id is None
    with Image.open(result.path) as image:
        assert image.size == (512, 512)
    assert provider.warnings == ("No Pexels result; used a neutral tile.",)
    assert query not in provider.warnings[0]


@pytest.mark.parametrize(
    ("provider_name", "credential", "provider_factory"),
    [
        (
            "Pexels",
            "pexels-secret",
            lambda client: PexelsImageProvider(SecretStr("pexels-secret"), client=client),
        ),
        (
            "Pixabay",
            "pixabay-secret",
            lambda client: PixabayImageProvider(SecretStr("pixabay-secret"), client=client),
        ),
    ],
)
def test_remote_provider_error_redacts_query(
    tmp_path: Path,
    provider_name: str,
    credential: str,
    provider_factory: object,
) -> None:
    query = "personal-secret-query"
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(500, text="upstream failed"))
    )
    provider = provider_factory(client)  # type: ignore[operator]

    with pytest.raises(ProviderError) as error:
        provider.fetch(query, tmp_path / "material", VideoAspect.PORTRAIT)  # type: ignore[union-attr]

    assert error.value.safe_message == f"No {provider_name} image could be fetched."
    assert error.value.diagnostic == "HTTPStatusError"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    formatted = "".join(traceback.format_exception(error.value))
    assert query not in error.value.safe_message
    assert query not in formatted
    assert credential not in formatted


@pytest.mark.parametrize(
    ("credential", "provider_factory"),
    [
        (
            "pexels-secret",
            lambda client: PexelsImageProvider(SecretStr("pexels-secret"), client=client),
        ),
        (
            "pixabay-secret",
            lambda client: PixabayImageProvider(SecretStr("pixabay-secret"), client=client),
        ),
    ],
)
def test_remote_provider_connection_error_detaches_raw_http_exception(
    credential: str,
    provider_factory: object,
) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(500, text="upstream failed"))
    )
    provider = provider_factory(client)  # type: ignore[operator]

    with pytest.raises(ProviderError) as error:
        provider.check_connection()  # type: ignore[union-attr]

    assert error.value.safe_message == "The image provider returned an invalid response."
    assert error.value.diagnostic == "HTTPStatusError"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert credential not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    ("provider_name", "provider_factory", "image_url"),
    [
        (
            "Pexels",
            lambda client: PexelsImageProvider(SecretStr("pexels-secret"), client=client),
            "https://images.pexels.com/\x00sensitive-path-marker?token=signed-secret",
        ),
        (
            "Pixabay",
            lambda client: PixabayImageProvider(SecretStr("pixabay-secret"), client=client),
            "https://cdn.pixabay.com/\x00sensitive-path-marker?token=signed-secret",
        ),
    ],
)
def test_remote_provider_invalid_download_url_is_safely_detached(
    tmp_path: Path,
    provider_name: str,
    provider_factory: object,
    image_url: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.pexels.com":
            return httpx.Response(
                200,
                json={"photos": [{"id": 1, "src": {"portrait": image_url}}]},
            )
        if request.url.host == "pixabay.com":
            return httpx.Response(
                200,
                json={"hits": [{"id": 1, "largeImageURL": image_url}]},
            )
        return httpx.Response(500)

    provider = provider_factory(  # type: ignore[operator]
        httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(ProviderError) as error:
        provider.fetch("word", tmp_path / "material", VideoAspect.PORTRAIT)  # type: ignore[union-attr]

    assert error.value.safe_message == f"No {provider_name} image could be fetched."
    assert error.value.diagnostic == "InvalidURL"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    formatted = "".join(traceback.format_exception(error.value))
    assert "signed-secret" not in formatted
    assert "sensitive-path-marker" not in formatted
