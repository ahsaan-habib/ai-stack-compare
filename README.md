# ai-stack-compare

The same small task built four ways — no framework, LangChain, LangGraph,
CrewAI — on the same local model (`qwen3:4b` via Ollama) and the same
retrieval, then measured side by side.

The task: answer a Laravel docs question in a few sentences with inline
citations, and refuse when the docs don't cover it.

| Implementation | Layer it represents | Shape |
|---|---|---|
| `stack/impl/plain.py` | provider API, no framework | retrieve → one chat call |
| `stack/impl/langchain_lcel.py` | framework glue | LCEL: retriever \| prompt \| model \| parser |
| `stack/impl/langgraph_graph.py` | orchestration (graph) | retrieve → grade → rewrite once → generate or refuse |
| `stack/impl/crewai_crew.py` | orchestration (multi-agent) | researcher agent with a search tool → writer agent |

Everything that isn't the framework is shared: `stack/retrieval.py` (Chroma +
`bge-small`, one section per chunk), `stack/common.py` (prompt, refusal
sentinel, result type). So differences in the numbers are differences the
framework made.

## Measuring fairly

Frameworks hide how many model calls they make. Rather than trust each one's
callbacks, `stack/proxy.py` sits between every implementation and Ollama and
counts requests and tokens the same way for all of them.

```bash
make install corpus index
python -m stack.compare                  # all four
python -m stack.compare --only plain langgraph
```

Prints a table per implementation — lines of code, p50 seconds per question,
LLM calls per question, tokens per question, refusals on the unanswerable
questions, false refusals on the answerable ones, errors — and writes every
question's row to `results/compare-*.csv`.

`questions.jsonl` has 8 answerable and 3 deliberately unanswerable questions.
It's a smoke test for comparing stacks, not an eval set; for that see
[rag-eval-gate](https://github.com/ahsaan-habib/rag-eval-gate).

## The rule this is meant to make concrete

For every layer you add, name the constraint it solves that the layer below
can't. Here:

- **LangChain** buys composable pieces and a large integration catalogue. For
  a single retrieve-then-answer call it mostly moves the same code into
  different shapes.
- **LangGraph** earns its place when the flow branches — grade, retry, refuse
  without generating. That's a real behaviour the plain version doesn't have,
  paid for in extra model calls.
- **CrewAI** is built for several roles coordinating over a long task. On a
  one-shot question, watch the calls-per-question column.

The model, the retrieval and the prompt matter more than any of these. Start
with `plain.py` and climb only when a rung can't hold the task.
