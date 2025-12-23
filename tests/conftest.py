"""Offline by default. A fake embedder stands in for sentence-transformers,
and FakeOllama below answers /api/chat (plain and streamed) so every
implementation can run without a model. test_smoke_ollama.py is opt-in."""
import json
import os
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("STACK_INDEX_DIR", tempfile.mkdtemp(prefix="stack-test-"))

import fake_st  # noqa: E402

fake_st.install()

import pytest  # noqa: E402

DOCS = {
    "eloquent.md": "# Eloquent\n\n## Removing Global Scopes\n\n" + (
        "If you would like to remove a global scope for a given query, you may use the "
        "withoutGlobalScope method. The method accepts the class name of the global scope as its "
        "only argument, and withoutGlobalScopes removes all of them at once.\n"),
    "queues.md": "# Queues\n\n## Dispatching Jobs\n\n" + (
        "Once you have written your job class, you may dispatch it using the dispatch method on the "
        "job itself. The arguments passed to the dispatch method will be given to the job's "
        "constructor when the job runs on a worker.\n"),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeOllama:
    """Replies by rule: grader prompts get {"relevant": true}, everything
    else gets `answer`. Streams NDJSON when asked to, like Ollama does."""

    def __init__(self, answer: str = "Use withoutGlobalScope on the query [1]."):
        self.answer, self.requests = answer, []
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                system = " ".join(m.get("content", "") for m in body.get("messages", []) if m["role"] == "system")
                text = '{"relevant": true}' if '"relevant"' in system else fake.answer
                base = {"model": body.get("model"), "created_at": "2026-01-01T00:00:00Z"}
                if body.get("stream", True):
                    lines = [{**base, "message": {"role": "assistant", "content": text}, "done": False},
                             {**base, "message": {"role": "assistant", "content": ""}, "done": True,
                              "done_reason": "stop", "prompt_eval_count": 50, "eval_count": 7}]
                    payload = "".join(json.dumps(x) + "\n" for x in lines).encode()
                    ctype = "application/x-ndjson"
                else:
                    payload = json.dumps({**base, "message": {"role": "assistant", "content": text}, "done": True,
                                          "done_reason": "stop", "prompt_eval_count": 50, "eval_count": 7}).encode()
                    ctype = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.port = free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_ollama():
    f = FakeOllama()
    yield f
    f.close()


@pytest.fixture(scope="session")
def corpus_index():
    from stack import retrieval

    d = Path(tempfile.mkdtemp(prefix="stack-corpus-"))
    for name, text in DOCS.items():
        (d / name).write_text(text)
    assert retrieval.build_index(str(d)) == 2
    return d
