"""No framework: retrieval function + one HTTP call. The baseline every other
implementation has to justify itself against."""
from __future__ import annotations

import httpx

from .. import config
from ..common import SYSTEM, Result, finish
from ..retrieval import format_passages, search

NAME = "plain"


def answer(question: str) -> Result:
    passages = search(question)
    r = httpx.post(f"{config.OLLAMA_URL}/api/chat", timeout=120, json={
        "model": config.MODEL,
        "stream": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{format_passages(passages)}\n\nQuestion: {question}"},
        ],
    })
    r.raise_for_status()
    return finish(r.json()["message"]["content"], [p.cite() for p in passages])
