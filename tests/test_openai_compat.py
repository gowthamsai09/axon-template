from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from digest.summarizers.base import (
    AuthError,
    BadResponseError,
    QuotaError,
    RateLimitError,
    TransientError,
)
from digest.summarizers.deepseek import deprecation_warning, make_deepseek
from digest.summarizers.huggingface import make_huggingface
from digest.summarizers.openai_compat import OpenAICompatClient

KEY = "sk-test-key-123456"


def ok_body(text: str = '{"a": 1}', **extra: Any) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}, **extra}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }


class Server:
    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if self.responses else httpx.Response(200, json=ok_body())
        if isinstance(item, Exception):
            raise item
        return item

    def payload(self, i: int = 0) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[i].content)
        return data


def make(
    server: Server, sleeps: list[float] | None = None, *, json_mode: bool = True
) -> OpenAICompatClient:
    log = sleeps if sleeps is not None else []
    return OpenAICompatClient(
        provider="deepseek",
        base_url="https://api.example.com/v1/",
        api_key=KEY,
        model="m",
        http=httpx.Client(transport=httpx.MockTransport(server)),
        json_mode=json_mode,
        sleep=log.append,
        rng=lambda: 0.5,
    )


def test_success_request_shape_and_usage() -> None:
    server = Server()
    result = make(server).complete("SYS", "USER")
    assert result.text == '{"a": 1}'
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.calls) == (
        12,
        5,
        1,
    )
    req = server.requests[0]
    assert str(req.url) == "https://api.example.com/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = server.payload()
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["response_format"] == {"type": "json_object"} and body["stream"] is False


def test_repr_never_shows_the_key() -> None:
    assert KEY not in repr(make(Server()))


def test_auth_and_quota_errors_do_not_retry_and_do_not_leak() -> None:
    for status, exc in ((401, AuthError), (403, AuthError), (402, QuotaError)):
        server = Server(httpx.Response(status, json={"error": {"message": f"bad {KEY}"}}))
        with pytest.raises(exc) as info:
            make(server).complete("s", "u")
        assert len(server.requests) == 1 and KEY not in str(info.value)


def test_rate_limit_backs_off_with_jitter_then_raises() -> None:
    sleeps: list[float] = []
    server = Server(*[httpx.Response(429)] * 3)
    with pytest.raises(RateLimitError):
        make(server, sleeps).complete("s", "u")
    assert len(server.requests) == 3 and len(sleeps) == 2
    assert sleeps == [2.0, 4.0]  # 2**attempt * (0.5 + rng 0.5)


def test_retry_after_is_honoured_and_capped() -> None:
    sleeps: list[float] = []
    server = Server(
        httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, json=ok_body())
    )
    make(server, sleeps).complete("s", "u")
    assert sleeps == [7.0]
    sleeps2: list[float] = []
    server2 = Server(
        httpx.Response(429, headers={"retry-after": "9999"}), httpx.Response(200, json=ok_body())
    )
    make(server2, sleeps2).complete("s", "u")
    assert sleeps2 == [30.0]


def test_5xx_retried_then_transient_and_recovery() -> None:
    server = Server(httpx.Response(503), httpx.Response(500), httpx.Response(502))
    with pytest.raises(TransientError, match="HTTP 502"):
        make(server).complete("s", "u")
    server = Server(httpx.Response(503), httpx.Response(200, json=ok_body("fine")))
    assert make(server).complete("s", "u").text == "fine"


def test_network_errors_are_transient_without_details() -> None:
    server = Server(*[httpx.ConnectTimeout(f"timeout talking with {KEY}")] * 3)
    with pytest.raises(TransientError) as info:
        make(server).complete("s", "u")
    assert "ConnectTimeout" in str(info.value) and KEY not in str(info.value)


def test_json_mode_rejected_by_provider_is_retried_without_it() -> None:
    server = Server(httpx.Response(400, json={"error": {"message": "response_format unsupported"}}))
    client = make(server)
    assert client.complete("s", "u").text == '{"a": 1}'
    assert "response_format" in server.payload(0) and "response_format" not in server.payload(1)
    client.complete("s", "u")  # remembered for the rest of the run
    assert "response_format" not in server.payload(2)


def test_other_4xx_is_a_bad_response_with_redacted_detail() -> None:
    server = Server(httpx.Response(404, json={"error": {"message": f"model not found {KEY}"}}))
    with pytest.raises(BadResponseError) as info:
        make(server, json_mode=False).complete("s", "u")
    assert "model not found ***" in str(info.value) and KEY not in str(info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=ok_body("")),
        httpx.Response(200, json=ok_body("cut off", finish_reason="length")),
    ],
)
def test_unusable_answers(response: httpx.Response) -> None:
    with pytest.raises(BadResponseError):
        make(Server(response)).complete("s", "u")


def test_deepseek_adapter_v4_disables_thinking() -> None:
    server = Server()
    http = httpx.Client(transport=httpx.MockTransport(server))
    make_deepseek(KEY, "deepseek-v4-flash", http).complete("s", "u")
    assert str(server.requests[0].url) == "https://api.deepseek.com/chat/completions"
    assert server.payload()["thinking"] == {"type": "disabled"}
    make_deepseek(KEY, "some-other-model", http).complete("s", "u")
    assert "thinking" not in server.payload(1)


def test_deepseek_legacy_model_warning() -> None:
    assert "deepseek-flash" in (deprecation_warning("deepseek-chat") or "")
    assert deprecation_warning("deepseek-reasoner") is not None
    assert deprecation_warning("deepseek-v4-pro") is None


def test_huggingface_adapter() -> None:
    server = Server()
    http = httpx.Client(transport=httpx.MockTransport(server))
    client = make_huggingface(KEY, "openai/gpt-oss-120b:cerebras", http)
    client.complete("s", "u")
    assert str(server.requests[0].url) == "https://router.huggingface.co/v1/chat/completions"
    assert server.payload()["model"] == "openai/gpt-oss-120b:cerebras"
    assert client.provider == "huggingface"
