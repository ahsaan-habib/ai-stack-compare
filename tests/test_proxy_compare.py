import json
import sys

import httpx

from conftest import free_port
from stack import compare, config
from stack.proxy import CountingProxy


def test_proxy_passes_bodies_through_and_counts(fake_ollama):
    proxy = CountingProxy(fake_ollama.url, port=free_port()).start()
    r = httpx.post(f"{proxy.url}/api/chat", json={"model": "m", "stream": False, "messages": []})
    assert r.json()["message"]["content"].startswith("Use withoutGlobalScope")
    with httpx.stream("POST", f"{proxy.url}/api/chat", json={"model": "m", "messages": []}) as s:
        lines = [json.loads(l) for l in s.iter_lines() if l]
    assert lines[-1]["done"] is True
    c = proxy.reset()
    assert (c.calls, c.input_tokens, c.output_tokens) == (2, 100, 14)
    assert proxy.reset().calls == 0
    proxy.server.shutdown()


def test_compare_runs_every_installed_impl(fake_ollama, corpus_index, tmp_path, monkeypatch, capsys):
    q = tmp_path / "q.jsonl"
    q.write_text('{"id": "q1", "question": "How do I remove a global scope?", "answerable": true}\n')
    monkeypatch.setattr(config, "OLLAMA_URL", fake_ollama.url)
    monkeypatch.setenv("OLLAMA_URL", fake_ollama.url)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["compare", "--questions", str(q), "--only", "plain", "langchain_lcel",
                                      "langgraph_graph", "crewai_crew"])
    for name in ("stack.impl.langchain_lcel", "stack.impl.langgraph_graph", "stack.impl.crewai_crew"):
        sys.modules.pop(name, None)       # they bind the URL at import
    compare.main()
    out = capsys.readouterr().out
    assert "| plain |" in out and "ERROR" not in out
    rows = list((tmp_path / "results").glob("compare-*.csv"))[0].read_text().splitlines()
    assert rows[0].startswith("impl,id,answerable,refused")
    for impl in ("langchain", "langgraph"):
        if f"skip {impl}" not in out:
            assert f"| {impl} |" in out
    # langgraph = grade + generate: two calls, the proxy saw both
    if "skip langgraph" not in out:
        line = next(l for l in out.splitlines() if l.startswith("| langgraph |"))
        assert float(line.split("|")[4]) == 2
