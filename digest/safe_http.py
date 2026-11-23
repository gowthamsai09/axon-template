"""SSRF-guarded HTTP fetching (spec 10.1): http(s) only, public addresses only, every
redirect hop re-checked, size and time caps. Used for extra_rss feeds, page extraction and
custom summarizer endpoints. Residual risk: DNS is resolved by the guard and again by the
HTTP stack (rebinding window); the Actions runner has no internal network of value."""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

USER_AGENT = "AxonDigest/1.0 (+https://github.com/gowthamsai09/Axon)"
Resolver = Callable[[str], list[str]]
# A host written as numbers that is not a canonical IP address (2130706433, 0x7f.1, 017700000001,
# 127.1, fullwidth digits). Whatever a resolver would do with it, it is refused (Appendix C 84).
_NUMERIC_HOST = re.compile(r"(?:0[xX][0-9a-fA-F]*|\d+)(?:\.(?:0[xX][0-9a-fA-F]*|\d+))*\.?")
# letters, digits, dot, hyphen, underscore (also non-ASCII letters)
_HOST_CHARS = re.compile(r"[\w.-]+")


class UnsafeUrlError(ValueError):
    """The URL points somewhere the pipeline must never connect to."""


class FetchError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def system_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UnsafeUrlError(f"cannot resolve {host!r}") from exc
    return sorted({str(info[4][0]) for info in infos})


def is_public_ip(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def check_public_url(url: str, resolver: Resolver = system_resolver) -> None:
    try:
        parts = urlsplit(url)
    except ValueError:
        raise UnsafeUrlError("not a valid URL") from None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise UnsafeUrlError("only http(s) URLs are allowed")
    if "@" in parts.netloc or parts.username or parts.password:
        raise UnsafeUrlError("credentials in URLs are not allowed")
    try:
        _ = parts.port  # raises ValueError for 99999, abc, -1
    except ValueError:
        raise UnsafeUrlError("invalid port") from None
    host = parts.hostname
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        if _NUMERIC_HOST.fullmatch(host):
            raise UnsafeUrlError("non-canonical IP address form") from None
        if _HOST_CHARS.fullmatch(host) is None:
            raise UnsafeUrlError("invalid host name") from None
        addresses = resolver(host.rstrip("."))
    if not addresses:
        raise UnsafeUrlError(f"{host!r} did not resolve")
    for address in addresses:
        if not is_public_ip(address):
            raise UnsafeUrlError(f"{host!r} resolves to a non-public address")


@dataclass(frozen=True)
class FetchResult:
    url: str
    content_type: str
    body: bytes


class SafeFetcher:
    def __init__(
        self,
        client: httpx.Client,
        *,
        resolver: Resolver = system_resolver,
        max_bytes: int = 2_000_000,
        max_redirects: int = 5,
        max_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._max_seconds = max_seconds
        self._clock = clock
        self._resolver = resolver
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects

    def get(self, url: str) -> FetchResult:
        current = url
        started = self._clock()
        for _ in range(self._max_redirects + 1):
            if self._clock() - started > self._max_seconds:
                raise FetchError("response too slow")
            try:
                check_public_url(current, self._resolver)
            except UnsafeUrlError as exc:
                raise FetchError(f"blocked: {exc}") from exc
            try:
                with self._client.stream(
                    "GET", current, headers={"User-Agent": USER_AGENT}, follow_redirects=False
                ) as resp:
                    if resp.status_code in {301, 302, 303, 307, 308}:
                        location = resp.headers.get("location")
                        if not location:
                            raise FetchError("redirect without Location", resp.status_code)
                        try:
                            current = urljoin(current, location)
                        except ValueError:
                            raise FetchError("invalid redirect location") from None
                        continue
                    if resp.status_code != 200:
                        raise FetchError(f"HTTP {resp.status_code}", resp.status_code)
                    body = bytearray()
                    for chunk in resp.iter_bytes():
                        body.extend(chunk)
                        if len(body) > self._max_bytes:
                            raise FetchError("response too large")
                        if self._clock() - started > self._max_seconds:
                            raise FetchError("response too slow")
                    ctype = resp.headers.get("content-type", "")
                    return FetchResult(current, ctype, bytes(body))
            except (httpx.HTTPError, httpx.InvalidURL) as exc:
                raise FetchError(f"network error: {type(exc).__name__}") from exc
        raise FetchError("too many redirects")
