"""Run every question through every implementation; compare what each stack
costs (code, calls, tokens, time) against what it buys (refusal behaviour).

    python -m stack.compare                      # all installed implementations
    python -m stack.compare --only plain langgraph
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

from . import config
from .proxy import CountingProxy

IMPLS = ["plain", "langchain_lcel", "langgraph_graph", "crewai_crew"]


def loc(module) -> int:
    """Non-blank, non-comment lines — rough, but the same rough for everyone."""
    lines = Path(module.__file__).read_text().splitlines()
    return sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--questions", default="questions.jsonl")
    args = ap.parse_args()

    proxy = CountingProxy(config.OLLAMA_URL).start()
    # implementations read OLLAMA_URL at import time, so point them at the proxy first
    config.OLLAMA_URL = os.environ["OLLAMA_URL"] = proxy.url

    questions = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    rows, summary = [], []
    for name in args.only or IMPLS:
        try:
            mod = importlib.import_module(f"stack.impl.{name}")
        except ImportError as e:
            print(f"skip {name}: {e} (pip install -e '.[all]')")
            continue
        proxy.reset()
        per = []
        for q in questions:
            t0 = time.perf_counter()
            try:
                res, err = mod.answer(q["question"]), ""
            except Exception as e:
                res, err = None, f"{type(e).__name__}: {e}"
            secs = time.perf_counter() - t0
            c = proxy.reset()
            row = {"impl": mod.NAME, "id": q["id"], "answerable": q["answerable"],
                   "refused": res.refused if res else None, "seconds": round(secs, 2),
                   "llm_calls": c.calls, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                   "answer": (res.answer if res else "")[:300], "error": err}
            rows.append(row)
            per.append(row)
            print(f"{mod.NAME:<10} {q['id']:<3} {secs:6.1f}s  calls {c.calls}  "
                  f"{'REFUSED' if row['refused'] else 'answered' if res else 'ERROR'}")
        ok = [r for r in per if not r["error"]]
        neg = [r for r in ok if not r["answerable"]]
        pos = [r for r in ok if r["answerable"]]
        summary.append({
            "impl": mod.NAME, "loc": loc(mod),
            "p50_s": round(statistics.median(r["seconds"] for r in ok), 1) if ok else None,
            "calls_per_q": round(statistics.mean(r["llm_calls"] for r in ok), 1) if ok else None,
            "tokens_per_q": round(statistics.mean(r["input_tokens"] + r["output_tokens"] for r in ok)) if ok else None,
            "refusal_correct": f"{sum(r['refused'] for r in neg)}/{len(neg)}",
            "false_refusals": f"{sum(r['refused'] for r in pos)}/{len(pos)}",
            "errors": len(per) - len(ok),
        })

    out = Path("results") / f"compare-{datetime.now():%Y%m%d-%H%M}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = list(summary[0])
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for s in summary:
        print("| " + " | ".join(str(s[c]) for c in cols) + " |")
    print(f"\nper-question rows -> {out}")


if __name__ == "__main__":
    main()
