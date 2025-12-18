"""A counting proxy in front of Ollama.

Frameworks hide how many model calls they make (an agent "thinking" is often
several). Instead of trusting each framework's own callbacks, every
implementation talks to Ollama through this proxy, which counts requests and
tokens the same way for all of them.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx


@dataclass
class Counts:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class CountingProxy:
    def __init__(self, upstream: str, port: int = 11435):
        self.upstream = upstream.rstrip("/")
        self.port = port
        self.counts = Counts()
        self.lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _forward(self, method: str) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
                is_llm = self.path.startswith(("/api/chat", "/api/generate", "/v1/chat/completions"))
                with httpx.Client(timeout=300) as client, client.stream(
                        method, proxy.upstream + self.path, content=body,
                        headers={"Content-Type": self.headers.get("Content-Type", "application/json")}) as r:
                    self.send_response(r.status_code)
                    self.send_header("Content-Type", r.headers.get("content-type", "application/json"))
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    tail = b""
                    for chunk in r.iter_bytes():
                        tail = (tail + chunk)[-8192:]   # usage stats live in the last line/object
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                if is_llm:
                    proxy._record(tail)

            def do_POST(self):
                self._forward("POST")

            def do_GET(self):
                self._forward("GET")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def _record(self, tail: bytes) -> None:
        inp = out = 0
        for line in reversed(tail.decode("utf-8", "ignore").strip().splitlines()):
            line = line.removeprefix("data: ").strip()
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "prompt_eval_count" in data or "eval_count" in data:          # ollama native
                inp, out = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
                break
            if data.get("usage"):                                            # openai-compatible
                inp, out = data["usage"].get("prompt_tokens", 0), data["usage"].get("completion_tokens", 0)
                break
        with self.lock:
            self.counts.calls += 1
            self.counts.input_tokens += inp
            self.counts.output_tokens += out

    def reset(self) -> Counts:
        with self.lock:
            snap, self.counts = self.counts, Counts()
        return snap

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "CountingProxy":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self
