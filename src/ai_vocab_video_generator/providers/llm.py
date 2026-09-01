"""Vocabulary generation through OpenAI-compatible chat APIs."""

import json
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import SecretStr, TypeAdapter, ValidationError

from ai_vocab_video_generator.domain import (
    MAX_TOPIC_LENGTH,
    MAX_VOCABULARY_ENTRIES,
    WordEntry,
    contains_sensitive_text,
    validate_llm_base_url,
)
from ai_vocab_video_generator.errors import ProviderError

_ENTRY_LIST = TypeAdapter(list[WordEntry])
_MAX_LLM_RESPONSE_BYTES = 1024 * 1024
_VOCABULARY_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "vocabulary_entries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "english": {"type": "string"},
                            "phonetic": {"type": "string"},
                            "chinese": {"type": "string"},
                        },
                        "required": ["english", "phonetic", "chinese"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entries"],
            "additionalProperties": False,
        },
    },
}


class OpenAICompatibleVocabularyProvider:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str,
        model: str,
        strict_json_schema: bool = False,
        reasoning_effort: str | None = None,
        thinking_mode: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._strict_json_schema = strict_json_schema
        self._reasoning_effort = reasoning_effort
        self._thinking_mode = thinking_mode
        self._active_secret = api_key.get_secret_value()
        safe_base_url = validate_llm_base_url(base_url)
        parsed = urlsplit(safe_base_url)
        self._endpoint = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f"{parsed.path.rstrip('/')}/chat/completions",
                parsed.query,
                "",
            )
        )
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=False,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def check_connection(self) -> None:
        """Validate credentials, endpoint, model, and structured vocabulary output."""
        self.generate("connection test", 1)

    def generate(self, topic: str, count: int) -> list[WordEntry]:
        normalized_topic = topic.strip()
        if not normalized_topic or len(normalized_topic) > MAX_TOPIC_LENGTH:
            raise ProviderError(
                f"The vocabulary topic must contain 1 to {MAX_TOPIC_LENGTH} characters."
            )
        bounded_count = max(1, min(count, MAX_VOCABULARY_ENTRIES))
        entries = self._request_entries(
            [
                {
                    "role": "system",
                    "content": (
                        "Return JSON only. Create English-learning vocabulary. Each "
                        "entry must normally be a single English word, not a phrase or "
                        "sentence. Use a short fixed expression only when the topic cannot "
                        "be represented naturally as a single word. Return it as "
                        '{"entries":[{"english":"...","phonetic":"/.../",'
                        '"chinese":"..."}]}. Omit IPA syllable-break periods. Do not add '
                        "commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Topic: {normalized_topic}\nNumber of entries: {bounded_count}",
                },
            ]
        )
        if not entries:
            raise ProviderError("The vocabulary provider returned no entries.")
        return entries[:bounded_count]

    def complete_phonetics(self, entries: Sequence[WordEntry]) -> list[WordEntry]:
        if not entries:
            return []
        if len(entries) > MAX_VOCABULARY_ENTRIES:
            raise ProviderError(
                f"Complete phonetics for at most {MAX_VOCABULARY_ENTRIES} entries at a time."
            )
        payload = [
            {"index": index, "english": entry.english, "chinese": entry.chinese}
            for index, entry in enumerate(entries)
        ]
        completed = self._request_entries(
            [
                {
                    "role": "system",
                    "content": (
                        "Return JSON only as "
                        '{"entries":[{"english":"...","phonetic":"/.../",'
                        '"chinese":"..."}]}. Preserve every English and Chinese value and '
                        "their order exactly. Add only standard English phonetics."
                        " Omit IPA syllable-break periods."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        if len(completed) != len(entries):
            raise ProviderError("The vocabulary provider changed the number of entries.")
        for original, result in zip(entries, completed, strict=True):
            if original.english != result.english or original.chinese != result.chinese:
                raise ProviderError("The vocabulary provider changed entered vocabulary.")
            if not result.phonetic:
                raise ProviderError("The vocabulary provider returned an empty phonetic value.")
        return completed

    def _request_entries(self, messages: list[Any]) -> list[WordEntry]:
        request_body: dict[str, Any] = {
            "model": self._model,
            "response_format": (
                _VOCABULARY_RESPONSE_FORMAT if self._strict_json_schema else {"type": "json_object"}
            ),
            "messages": messages,
        }
        if self._strict_json_schema:
            request_body["temperature"] = 0
        if self._reasoning_effort is not None:
            request_body["reasoning_effort"] = self._reasoning_effort
        if self._thinking_mode is not None:
            request_body["thinking"] = {"type": self._thinking_mode}
        try:
            with self._client.stream(
                "POST",
                self._endpoint,
                headers={"Authorization": f"Bearer {self._active_secret}"},
                json=request_body,
                follow_redirects=False,
            ) as response:
                response.raise_for_status()
                declared = response.headers.get("content-length")
                if declared is not None and int(declared) > _MAX_LLM_RESPONSE_BYTES:
                    raise ValueError("LLM response exceeds the size limit.")
                chunks: list[bytes] = []
                byte_count = 0
                for chunk in response.iter_bytes():
                    byte_count += len(chunk)
                    if byte_count > _MAX_LLM_RESPONSE_BYTES:
                        raise ValueError("LLM response exceeds the size limit.")
                    chunks.append(chunk)
            payload = json.loads(b"".join(chunks))
            content = payload["choices"][0]["message"].get("content") or ""
            entries = parse_vocabulary_payload(content)
            if any(
                contains_sensitive_text(
                    value,
                    active_secrets=(self._active_secret,),
                )
                for entry in entries
                for value in (entry.english, entry.phonetic, entry.chinese)
            ):
                raise ProviderError(
                    "The vocabulary provider returned unsafe content.",
                    diagnostic="SensitiveProviderContent",
                )
            return entries
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "The vocabulary provider request timed out. "
                "Try again and check provider availability.",
                diagnostic=type(exc).__name__,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(
                "The vocabulary provider is unavailable. "
                "Check that the service is running and reachable.",
                diagnostic=type(exc).__name__,
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 403}:
                message = (
                    "The vocabulary provider rejected the API key. Check the configured credential."
                )
            elif exc.response.status_code == 404:
                message = (
                    "The configured vocabulary model or API endpoint was not found. "
                    "Check the model name and provider URL."
                )
            elif exc.response.status_code == 429:
                message = (
                    "The vocabulary provider quota is exhausted or requests are too frequent. "
                    "Check the account balance and rate limits, then try again."
                )
            elif exc.response.status_code in {500, 502, 503, 504}:
                message = (
                    "The vocabulary provider is unavailable. "
                    "Check that the service is running and reachable."
                )
            else:
                message = "The vocabulary provider returned an invalid response."
            raise ProviderError(message, diagnostic=type(exc).__name__) from exc
        except (
            httpx.HTTPError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            raise ProviderError(
                "The vocabulary provider returned an invalid response.",
                diagnostic=type(exc).__name__,
            ) from exc


def parse_vocabulary_payload(content: str) -> list[WordEntry]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    payload = json.loads(cleaned)
    raw_entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_entries, list) or len(raw_entries) > MAX_VOCABULARY_ENTRIES:
        raise ValueError(
            f"Vocabulary response must contain at most {MAX_VOCABULARY_ENTRIES} entries."
        )
    return _ENTRY_LIST.validate_python(raw_entries)
