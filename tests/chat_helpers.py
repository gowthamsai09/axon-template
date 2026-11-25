"""Test doubles for the chat channels: a scripted server and a host router (offline)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

Step = httpx.Response | Exception


class Script:
    """Scripted service: each request takes the next step; an empty script answers `default`."""

    def __init__(self, default: Callable[[], httpx.Response], *steps: Step) -> None:
        self.default = default
        self.steps: list[Step] = list(steps)
        self.requests: list[httpx.Request] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.steps.pop(0) if self.steps else self.default()
        if isinstance(step, Exception):
            raise step
        return step

    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(r.content) for r in self.requests]


CHAT_HOSTS = frozenset({"api.telegram.org", "hooks.slack.com"})


def routed(routes: dict[str, Script]) -> httpx.Client:
    """One client that sends each request to the script registered for its host.

    A chat host without a script fails the test; any other host (article pages, robots.txt)
    answers 404, like the World fake in test_cli."""

    def handle(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        script = routes.get(host)
        if script is not None:
            return script.handle(request)
        if host in CHAT_HOSTS:
            raise AssertionError(f"unexpected chat request to {host}")
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handle))


def refusing() -> httpx.Client:
    """A client that fails the test if anything is sent to a chat service."""
    return routed({})


def tg_ok() -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})


def tg_error(
    code: int, description: str = "Bad Request: x", **parameters: object
) -> httpx.Response:
    body: dict[str, object] = {"ok": False, "error_code": code, "description": description}
    if parameters:
        body["parameters"] = dict(parameters)
    return httpx.Response(code, json=body)


def slack_ok() -> httpx.Response:
    return httpx.Response(200, text="ok")
