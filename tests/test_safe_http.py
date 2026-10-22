from __future__ import annotations

import httpx
import pytest

from digest.safe_http import (
    FetchError,
    SafeFetcher,
    UnsafeUrlError,
    check_public_url,
    is_public_ip,
)

PUBLIC = lambda host: ["93.184.216.34"]  # noqa: E731


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("93.184.216.34", True),
        ("8.8.8.8", True),
        ("10.0.0.5", False),
        ("192.168.1.1", False),
        ("127.0.0.1", False),
        ("169.254.169.254", False),  # cloud metadata
        ("100.64.0.1", False),
        ("::1", False),
        ("fd00::1", False),
        ("::ffff:10.0.0.1", False),
        ("2606:4700:4700::1111", True),
    ],
)
def test_is_public_ip(ip: str, expected: bool) -> None:
    assert is_public_ip(ip) is expected


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x",
        "file:///etc/passwd",
        "https://user:pw@example.com/",
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
    ],
)
def test_blocked_urls(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        check_public_url(url, PUBLIC)


def test_dns_to_private_blocked_and_public_ok() -> None:
    with pytest.raises(UnsafeUrlError):
        check_public_url("https://internal.example/", lambda h: ["10.1.2.3", "93.184.216.34"])
    check_public_url("https://example.com/", PUBLIC)


def _fetcher(handler: httpx.MockTransport, **kw: int) -> SafeFetcher:
    return SafeFetcher(httpx.Client(transport=handler), resolver=PUBLIC, **kw)


def test_fetch_ok() -> None:
    f = _fetcher(httpx.MockTransport(lambda r: httpx.Response(200, text="hello")))
    assert f.get("https://example.com/a").body == b"hello"


def test_redirect_to_metadata_is_blocked() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/x"})
        return httpx.Response(200, text="secret")

    f = SafeFetcher(
        httpx.Client(transport=httpx.MockTransport(handler)),
        resolver=lambda h: ["169.254.169.254"] if h == "169.254.169.254" else ["93.184.216.34"],
    )
    with pytest.raises(FetchError, match="blocked"):
        f.get("https://example.com/start")


def test_redirect_loop_size_and_status() -> None:
    loop = httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": str(r.url)}))
    with pytest.raises(FetchError, match="too many redirects"):
        _fetcher(loop, max_redirects=3).get("https://example.com/")
    big = httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 5000))
    with pytest.raises(FetchError, match="too large"):
        _fetcher(big, max_bytes=1000).get("https://example.com/")
    nf = httpx.MockTransport(lambda r: httpx.Response(404))
    with pytest.raises(FetchError) as info:
        _fetcher(nf).get("https://example.com/")
    assert info.value.status == 404
