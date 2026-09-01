"""Session-only image choices and independent search/render cache identities."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from ai_vocab_video_generator.domain import MaterialAsset, MaterialStyle, VideoAspect, WordEntry
from ai_vocab_video_generator.providers.images import RemoteImageCandidate


class MaterialUpload(Protocol):
    name: str
    file_id: str

    def getvalue(self) -> bytes: ...


@dataclass(frozen=True)
class CandidateGallery:
    candidates: tuple[RemoteImageCandidate, ...]
    thumbnails: tuple[Path | None, ...]


@dataclass(frozen=True)
class MaterialSelection:
    candidate: RemoteImageCandidate
    asset: MaterialAsset
    manual: bool


@dataclass
class WordMaterialState:
    identity: str
    search_query: str = ""
    search_key: str = ""
    gallery: CandidateGallery | None = None
    selection: MaterialSelection | None = None
    upload: MaterialUpload | None = field(default=None, repr=False)

    def set_query(self, value: str) -> None:
        query = " ".join(value.split())
        if query != self.search_query:
            self.search_query = query
            # Browsing new candidates must not replace a committed image or upload.
            self.gallery = None

    def sync_search(self, key: str) -> None:
        if key != self.search_key:
            self.search_key = key
            self.gallery = None
            self.selection = None
        if self.selection is not None and not self.selection.asset.path.is_file():
            self.selection = None

    def select(
        self, candidate: RemoteImageCandidate, asset: MaterialAsset, *, manual: bool
    ) -> None:
        self.selection = MaterialSelection(candidate, asset, manual)
        self.upload = None

    def set_upload(self, upload: MaterialUpload | None) -> None:
        self.upload = upload

    def review_status(
        self, *, has_saved_pin: bool = False
    ) -> Literal["upload", "manual", "auto", "saved", "empty", "pending"]:
        """Describe the effective source, not just the most recent search result."""
        if self.upload is not None:
            return "upload"
        if self.selection is not None and self.selection.asset.path.is_file():
            return "manual" if self.selection.manual else "auto"
        if has_saved_pin:
            return "saved"
        if self.gallery is not None and not self.gallery.candidates:
            return "empty"
        return "pending"


def remote_search_key(
    entry: WordEntry,
    aspect: VideoAspect,
    material: MaterialStyle,
    provider_name: str,
    credential_digest: str,
) -> str:
    """Layout changes affect rendering, not the chosen source image."""
    payload = {
        "query": " ".join(entry.english.split()).casefold(),
        "aspect": aspect.value,
        "pool_size": material.pool_size,
        "selection_mode": material.selection_mode.value,
        "provider": provider_name.strip().casefold(),
        "credential_identity": credential_digest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
