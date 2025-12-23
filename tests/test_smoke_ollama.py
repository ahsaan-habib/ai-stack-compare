"""plain and LangGraph against a real local model, over a two-section index
(fake embedder, so no model download). Opt-in:

    RUN_OLLAMA=1 pytest tests/test_smoke_ollama.py -s
"""
import importlib
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_OLLAMA") != "1", reason="set RUN_OLLAMA=1 to run")


@pytest.mark.parametrize("impl", ["plain", "langgraph_graph"])
def test_answers_and_refuses(impl, corpus_index):
    try:
        mod = importlib.import_module(f"stack.impl.{impl}")
    except ImportError as e:
        pytest.skip(str(e))
    ok = mod.answer("How do I remove a global scope for one query?")
    print(f"\n{impl}: {ok}")
    assert not ok.refused and "withoutGlobalScope" in ok.answer
    off = mod.answer("What is the capital of Australia?")
    print(f"{impl} off-topic: {off}")
    assert off.refused
