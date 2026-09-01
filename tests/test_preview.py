import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from ai_vocab_video_generator.domain import (
    MaterialAsset,
    MaterialKind,
    MaterialStyle,
    VideoAspect,
    WordEntry,
)
from ai_vocab_video_generator.preview import CandidateGallery, WordMaterialState, remote_search_key
from ai_vocab_video_generator.providers.images import RemoteImageCandidate


def _key(**changes):
    values = {
        "entry": WordEntry(english="apple"),
        "aspect": VideoAspect.PORTRAIT,
        "material": MaterialStyle(),
        "provider_name": "pexels",
        "credential_digest": "credential-a",
    }
    values.update(changes)
    return remote_search_key(**values)


def test_material_overview_status_precedence(tmp_path: Path):
    state = WordMaterialState(identity="apple")
    assert state.review_status() == "pending"
    state.gallery = CandidateGallery((), ())
    assert state.review_status() == "empty"
    assert state.review_status(has_saved_pin=True) == "saved"
    image = tmp_path / "selected.png"
    image.write_bytes(b"image")
    candidate = RemoteImageCandidate("one", "image", "thumb")
    asset = MaterialAsset(path=image, kind=MaterialKind.IMAGE)
    state.select(candidate, asset, manual=False)
    assert state.review_status() == "auto"
    state.select(candidate, asset, manual=True)
    assert state.review_status() == "manual"
    state.set_upload(Upload())
    assert state.review_status() == "upload"
    state.set_query("apple fruit")
    assert state.review_status() == "upload"
    state.set_upload(None)
    assert state.review_status() == "manual"
    image.unlink()
    assert state.review_status() == "pending"


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 500),
        ("height", 600),
        ("offsets", {"top": 20}),
        ("shape", "rectangle"),
        ("fit_mode", "contain"),
        ("enabled", False),
        ("source", "local"),
    ],
)
def test_layout_and_visibility_changes_keep_search_identity(field, value):
    assert _key(material=MaterialStyle(**{field: value})) == _key()


@pytest.mark.parametrize(
    "changes",
    [
        {"entry": WordEntry(english="pear")},
        {"aspect": VideoAspect.LANDSCAPE},
        {"material": MaterialStyle(pool_size=9)},
        {"material": MaterialStyle(selection_mode="random")},
        {"provider_name": "pixabay"},
        {"credential_digest": "credential-b"},
    ],
)
def test_search_changes_have_independent_candidate_identity(changes):
    assert _key(**changes) != _key()


@dataclass
class Upload:
    name: str = "local.png"
    file_id: str = "test-upload"

    def getvalue(self):
        return b"local"


def test_word_state_switches_sources_without_losing_remote_fallback(tmp_path: Path):
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    candidate = RemoteImageCandidate(
        "1", "https://images.pexels.com/1", "https://images.pexels.com/t1"
    )
    asset = MaterialAsset(path=path, kind=MaterialKind.IMAGE)
    state = WordMaterialState("apple")
    state.sync_search(_key())
    state.gallery = CandidateGallery((candidate,), (path,))
    state.select(candidate, asset, manual=True)
    chosen = state.selection
    state.set_upload(Upload())
    state.sync_search(_key(material=MaterialStyle(width=500)))
    assert state.upload is not None
    assert state.selection == chosen
    assert state.gallery is not None
    state.set_upload(None)
    assert state.selection == chosen
    state.set_upload(Upload())
    state.select(candidate, asset, manual=False)
    assert state.upload is None
    assert state.selection.asset == asset


def test_search_change_clears_remote_choice_but_keeps_local_upload(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    state = WordMaterialState("apple")
    candidate = RemoteImageCandidate(
        "1", "https://images.pexels.com/1", "https://images.pexels.com/t1"
    )
    state.sync_search(_key())
    state.select(candidate, MaterialAsset(path=path, kind=MaterialKind.IMAGE), manual=True)
    state.gallery = CandidateGallery((candidate,), (path,))
    upload = Upload()
    state.set_upload(upload)
    state.sync_search(_key(provider_name="pixabay"))
    assert state.selection is None
    assert state.gallery is None
    assert state.upload is upload


def test_missing_selected_file_cannot_remain_marked_selected(tmp_path):
    path = tmp_path / "missing.png"
    state = WordMaterialState("apple")
    state.sync_search(_key())
    candidate = RemoteImageCandidate(
        "1", "https://images.pexels.com/1", "https://images.pexels.com/t1"
    )
    state.select(candidate, MaterialAsset(path=path, kind=MaterialKind.IMAGE), manual=True)
    state.sync_search(_key())
    assert state.selection is None


def test_candidate_urls_and_raw_credentials_are_not_in_state_representations():
    raw = "private-test-credential"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    key = _key(credential_digest=digest)
    candidate = RemoteImageCandidate(
        "1",
        f"https://images.pexels.com/1?secret={raw}",
        f"https://images.pexels.com/t1?secret={raw}",
    )
    state = WordMaterialState("apple")
    state.sync_search(key)
    state.gallery = CandidateGallery((candidate,), (None,))
    assert raw not in repr(state)
    assert "https://" not in repr(state)
    assert len(key) == 64
