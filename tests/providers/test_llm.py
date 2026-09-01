import json

import httpx
import pytest
from pydantic import SecretStr

from ai_vocab_video_generator.domain import MAX_TOPIC_LENGTH, WordEntry
from ai_vocab_video_generator.errors import ProviderError
from ai_vocab_video_generator.providers.llm import OpenAICompatibleVocabularyProvider


def test_connection_check_uses_the_real_structured_vocabulary_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"entries":[{"english":"test",'
                                '"phonetic":"/test/","chinese":"测试"}]}'
                            )
                        }
                    }
                ]
            },
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    provider.check_connection()

    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["model"] == "test-model"
    assert body["response_format"] == {"type": "json_object"}
    assert "connection test" in body["messages"][1]["content"]
    assert "Number of entries: 1" in body["messages"][1]["content"]


def test_openai_compatible_provider_parses_entries_and_bounds_count() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = json.dumps(
            {
                "entries": [
                    {"english": "apple", "phonetic": "/ˈæp.əl/", "chinese": "苹果"},
                    {"english": "banana", "phonetic": "/bəˈnɑː.nə/", "chinese": "香蕉"},
                ]
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=http_client,
    )

    entries = provider.generate("fruit", 1)

    assert [entry.english for entry in entries] == ["apple"]
    assert requests[0].url == "https://llm.example/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-secret"
    assert b"fruit" in requests[0].content
    body = json.loads(requests[0].content)
    assert "single English word" in body["messages"][0]["content"]


def test_provider_rejects_oversized_topic_before_network_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="240"):
        provider.generate("x" * (MAX_TOPIC_LENGTH + 1), 1)

    assert requests == []


def test_provider_rejects_oversized_phonetic_completion_before_network_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="at most 50"):
        provider.complete_phonetics([WordEntry(english="word") for _ in range(51)])

    assert requests == []


def test_provider_uses_strict_schema_and_disables_reasoning_when_requested() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"entries":[{"english":"apple",'
                                '"phonetic":"/apple/","chinese":"苹果"}]}'
                            )
                        }
                    }
                ]
            },
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("local-session-key"),
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:9b",
        strict_json_schema=True,
        reasoning_effort="none",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    provider.generate("fruit", 1)

    body = json.loads(requests[0].content)
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    item_schema = response_format["json_schema"]["schema"]["properties"]["entries"]["items"]
    assert item_schema["required"] == ["english", "phonetic", "chinese"]
    assert item_schema["additionalProperties"] is False


def test_provider_keeps_generic_openai_compatible_request_by_default() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entries":[{"english":"apple"}]}'}}]},
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    provider.generate("fruit", 1)

    body = json.loads(requests[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in body
    assert "temperature" not in body


def test_provider_can_disable_thinking_for_supported_models() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entries":[{"english":"apple"}]}'}}]},
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k2.6",
        thinking_mode="disabled",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    provider.generate("fruit", 1)

    body = json.loads(requests[0].content)
    assert body["thinking"] == {"type": "disabled"}


def test_provider_appends_chat_path_before_a_noncredential_query() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entries":[{"english":"apple"}]}'}}]},
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1?api-version=2026-08-01",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    provider.generate("fruit", 1)

    assert requests[0].url == ("https://llm.example/v1/chat/completions?api-version=2026-08-01")


def test_openai_compatible_provider_accepts_markdown_fenced_json() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n[{"english":"cat","chinese":"猫"}]\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.generate("pets", 3)[0].english == "cat"


def test_complete_phonetics_preserves_user_words_and_translations() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "entries": [
                                        {
                                            "english": "apple",
                                            "phonetic": "/ˈæp.əl/",
                                            "chinese": "苹果",
                                        }
                                    ]
                                }
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.complete_phonetics([WordEntry(chinese="苹果", english="apple")]) == [
        WordEntry(chinese="苹果", english="apple", phonetic="/ˈæp.əl/")
    ]


def test_complete_phonetics_rejects_changed_user_content() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"entries":[{"english":"pear","chinese":"梨"}]}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="changed"):
        provider.complete_phonetics([WordEntry(chinese="苹果", english="apple")])


@pytest.mark.parametrize(
    "base_url",
    [
        "http://llm.example/v1",
        "https://user:password@llm.example/v1",
        "https://llm.example/v1?api_key=private-value",
    ],
)
def test_provider_rejects_unsafe_endpoint_before_constructing_a_request(base_url: str) -> None:
    requests: list[httpx.Request] = []
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(500)
        )
    )

    with pytest.raises(ValueError, match=r"LLM base URL|HTTPS"):
        OpenAICompatibleVocabularyProvider(
            api_key=SecretStr("test-secret"),
            base_url=base_url,
            model="test-model",
            http_client=client,
        )

    assert requests == []


def test_provider_allows_plain_http_only_for_exact_loopback() -> None:
    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("local-session-key"),
        base_url="http://127.0.0.1:11434/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
    )

    assert provider is not None


def test_provider_rejects_a_reflected_active_credential() -> None:
    secret = "provider-active-secret-value"

    def handler(_: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {"entries": [{"english": secret, "phonetic": "/x/", "chinese": "测试"}]}
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ]
            },
        )

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr(secret),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate("safe topic", 1)

    assert secret not in str(caught.value)
    assert secret not in caught.value.safe_message


def test_provider_wraps_a_malicious_http_error_without_exposing_its_body() -> None:
    reflected = "server-reflected-private-value"
    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(401, text=f"authorization failed: {reflected}")
            )
        ),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate("safe topic", 1)

    assert caught.value.safe_message == (
        "The vocabulary provider rejected the API key. Check the configured credential."
    )
    assert caught.value.diagnostic == "HTTPStatusError"
    assert reflected not in str(caught.value)


def test_provider_wraps_request_timeout_without_exposing_transport_details() -> None:
    private_detail = "timeout while using private test route"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(private_detail, request=request)

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate("safe topic", 1)

    assert caught.value.safe_message == (
        "The vocabulary provider request timed out. Try again and check provider availability."
    )
    assert caught.value.diagnostic == "ReadTimeout"
    assert private_detail not in str(caught.value)


def test_provider_reports_an_unreachable_service_without_exposing_transport_details() -> None:
    private_detail = "connection failed through private test route"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(private_detail, request=request)

    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate("safe topic", 1)

    assert caught.value.safe_message == (
        "The vocabulary provider is unavailable. Check that the service is running and reachable."
    )
    assert caught.value.diagnostic == "ConnectError"
    assert private_detail not in str(caught.value)


def test_provider_reports_a_gateway_failure_as_unavailable_without_exposing_response_body() -> None:
    reflected = "gateway exposed private network detail"
    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(502, text=reflected))
        ),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate("safe topic", 1)

    assert caught.value.safe_message == (
        "The vocabulary provider is unavailable. Check that the service is running and reachable."
    )
    assert caught.value.diagnostic == "HTTPStatusError"
    assert reflected not in str(caught.value)


def test_provider_reports_a_missing_model_or_endpoint_without_exposing_response_body() -> None:
    reflected = "missing private-model-name"
    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(404, text=f"model not found: {reflected}")
            )
        ),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate("safe topic", 1)

    assert caught.value.safe_message == (
        "The configured vocabulary model or API endpoint was not found. "
        "Check the model name and provider URL."
    )
    assert caught.value.diagnostic == "HTTPStatusError"
    assert reflected not in str(caught.value)


def test_provider_reports_exhausted_quota_without_exposing_response_body() -> None:
    reflected = "private account quota detail"
    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(429, text=reflected))
        ),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate("safe topic", 1)

    assert caught.value.safe_message == (
        "The vocabulary provider quota is exhausted or requests are too frequent. "
        "Check the account balance and rate limits, then try again."
    )
    assert caught.value.diagnostic == "HTTPStatusError"
    assert reflected not in str(caught.value)


def test_provider_rejects_an_oversized_response_body() -> None:
    oversized = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"entries":[{"english":"apple"}]}',
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "padding": "x" * (1024 * 1024),
        }
    ).encode()
    provider = OpenAICompatibleVocabularyProvider(
        api_key=SecretStr("test-secret"),
        base_url="https://llm.example/v1",
        model="test-model",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=oversized))
        ),
    )

    with pytest.raises(ProviderError, match="invalid response"):
        provider.generate("safe topic", 1)


def test_parser_rejects_more_than_the_supported_entry_budget() -> None:
    content = json.dumps({"entries": [{"english": f"word-{index}"} for index in range(51)]})

    with pytest.raises(ValueError, match="50"):
        from ai_vocab_video_generator.providers.llm import parse_vocabulary_payload

        parse_vocabulary_payload(content)
