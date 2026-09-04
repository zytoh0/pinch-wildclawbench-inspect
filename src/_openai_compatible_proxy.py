from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenAICompatibleProxy(BaseHTTPRequestHandler):
    """Small OpenAI-compatible proxy used to map benchmark-local model aliases."""

    target_base_url: ClassVar[str]
    alias_model: ClassVar[str]
    actual_model: ClassVar[str]
    api_key: ClassVar[str | None]
    timeout_seconds: ClassVar[float]
    extra_body: ClassVar[dict[str, object]]

    def _target_url(self) -> str:
        base = self.target_base_url.rstrip("/")
        path = self.path
        if base.endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        return base + path

    def _send(
        self, status: int, body: bytes, content_type: str = "application/json"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _relay(self, response: Any) -> None:
        """Stream the upstream response to the client as it arrives.

        Buffering the whole body would hold back every token until generation
        finishes; OpenClaw aborts a request after 120 s without any bytes, so
        long generations on a busy server would fail even though the server was
        still producing output.
        """
        self.send_response(response.status)
        self.send_header(
            "Content-Type", response.headers.get("Content-Type", "application/json")
        )
        for name in ("Cache-Control", "X-Request-Id"):
            if response.headers.get(name):
                self.send_header(name, response.headers[name])
        self.end_headers()
        while True:
            chunk = response.read1(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()

    def _forward(self, body: bytes | None = None) -> None:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in {"host", "content-length", "authorization"}
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if body is not None:
            headers["Content-Length"] = str(len(body))

        request = Request(
            self._target_url(),
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                self._relay(response)
        except HTTPError as exc:
            self._send(
                exc.code,
                exc.read(),
                exc.headers.get("Content-Type", "application/json"),
            )
        except URLError as exc:
            message = {"error": f"upstream request failed: {exc}"}
            self._send(502, json.dumps(message).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        content_type = self.headers.get("Content-Type", "")
        if body and "application/json" in content_type:
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                if payload.get("model") == self.alias_model:
                    payload["model"] = self.actual_model
                if self.extra_body and self.path.rstrip("/").endswith(
                    "/chat/completions"
                ):
                    payload = merge_extra_body(payload, self.extra_body)
                body = json.dumps(payload).encode("utf-8")
        self._forward(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def merge_extra_body(
    payload: dict[str, object], extra: dict[str, object]
) -> dict[str, object]:
    """Add server-specific request fields the benchmark clients cannot set themselves.

    Nested dicts (e.g. ``chat_template_kwargs``) are merged one level deep so a
    client's own entries survive; scalar fields in ``extra`` win.
    """
    merged = dict(payload)
    for key, value in extra.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = {**current, **value}
        else:
            merged[key] = value
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible model alias proxy")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--alias-model", required=True)
    parser.add_argument("--actual-model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON object merged into every chat completion request body",
    )
    args = parser.parse_args()

    OpenAICompatibleProxy.target_base_url = args.target_base_url
    OpenAICompatibleProxy.alias_model = args.alias_model
    OpenAICompatibleProxy.actual_model = args.actual_model
    OpenAICompatibleProxy.api_key = (
        args.api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY") or None
    )
    OpenAICompatibleProxy.timeout_seconds = args.timeout_seconds
    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    if not isinstance(extra_body, dict):
        raise SystemExit("--extra-body must be a JSON object")
    OpenAICompatibleProxy.extra_body = extra_body

    server = ThreadingHTTPServer((args.listen_host, args.port), OpenAICompatibleProxy)
    server.serve_forever()


if __name__ == "__main__":
    main()
