"""SSRF-guarded HTTP fetching (spec 10.1): http(s) only, public addresses only, every
redirect hop re-checked, size and time caps. Used for extra_rss feeds, page extraction and
custom summarizer endpoints. Residual risk: DNS is resolved by the guard and again by the
HTTP stack (rebinding window); the Actions runner has no internal network of value."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

USER_AGENT = "AxonDigest/1.0 (+https://github.com/gowthamsai09/Axon)"
Resolver = Callable[[str], list[str]]


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
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise UnsafeUrlError("only http(s) URLs are allowed")
    if parts.username or parts.password:
        raise UnsafeUrlError("credentials in URLs are not allowed")
    host = parts.hostname
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        addresses = resolver(host)
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
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects

    def get(self, url: str) -> FetchResult:
        current = url
        for _ in range(self._max_redirects + 1):
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
                        current = urljoin(current, location)
                        continue
                    if resp.status_code != 200:
                        raise FetchError(f"HTTP {resp.status_code}", resp.status_code)
                    body = bytearray()
                    for chunk in resp.iter_bytes():
                        body.extend(chunk)
                        if len(body) > self._max_bytes:
                            raise FetchError("response too large")
                    ctype = resp.headers.get("content-type", "")
                    return FetchResult(current, ctype, bytes(body))
            except httpx.HTTPError as exc:
                raise FetchError(f"network error: {type(exc).__name__}") from exc
        raise FetchError("too many redirects")
