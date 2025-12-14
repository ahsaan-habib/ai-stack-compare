"""LangGraph earns its place when the flow branches. Here: retrieve, have the
model grade whether the passages answer the question, rewrite the query once
if they don't, then answer — or refuse without calling the generator at all.

    retrieve -> grade --relevant--> generate -> END
                  |--not relevant, first try--> rewrite -> retrieve
                  `--not relevant, retried----> refuse -> END
"""
from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .. import config
from ..common import SYSTEM, Result, finish
from ..retrieval import Passage, format_passages, search

NAME = "langgraph"

llm = ChatOllama(model=config.MODEL, base_url=config.OLLAMA_URL, temperature=0)
grader = ChatOllama(model=config.MODEL, base_url=config.OLLAMA_URL, temperature=0, format="json")


class State(TypedDict, total=False):
    question: str
    query: str
    passages: list[Passage]
    relevant: bool
    rewrites: int
    result: Result


def retrieve(s: State) -> State:
    return {"passages": search(s.get("query") or s["question"])}


def grade(s: State) -> State:
    msg = grader.invoke([
        SystemMessage('Do these passages contain the answer to the question? '
                      'Reply as JSON: {"relevant": true|false}. /no_think'),
        HumanMessage(f"{format_passages(s['passages'])}\n\nQuestion: {s['question']}"),
    ])
    try:
        relevant = bool(json.loads(msg.content).get("relevant"))
    except (json.JSONDecodeError, AttributeError):
        relevant = True   # grader broke: let the generator's own refusal rule decide
    return {"relevant": relevant}


def rewrite(s: State) -> State:
    msg = llm.invoke([
        SystemMessage("Rewrite the question as a short search query using Laravel terminology. "
                      "Reply with the query only. /no_think"),
        HumanMessage(s["question"]),
    ])
    return {"query": msg.content.strip(), "rewrites": s.get("rewrites", 0) + 1}


def generate(s: State) -> State:
    msg = llm.invoke([SystemMessage(SYSTEM),
                      HumanMessage(f"{format_passages(s['passages'])}\n\nQuestion: {s['question']}")])
    return {"result": finish(msg.content, [p.cite() for p in s["passages"]])}


def refuse(s: State) -> State:
    return {"result": Result("The docs I have don't cover this.", [], refused=True)}


def route(s: State) -> str:
    if s["relevant"]:
        return "generate"
    return "rewrite" if s.get("rewrites", 0) < 1 else "refuse"


g = StateGraph(State)
for name, fn in [("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                 ("generate", generate), ("refuse", refuse)]:
    g.add_node(name, fn)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", route, {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
g.add_edge("rewrite", "retrieve")
g.add_edge("generate", END)
g.add_edge("refuse", END)
graph = g.compile()


def answer(question: str) -> Result:
    return graph.invoke({"question": question})["result"]
