"""CrewAI: a researcher agent with the search tool, a writer agent that turns
its notes into a cited answer. Two roles is the smallest crew that's still a
crew — and this task doesn't need one, which is the point of including it."""
from __future__ import annotations

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from .. import config
from ..common import NOT_IN_DOCS, SYSTEM, Result, finish
from ..retrieval import format_passages, search

NAME = "crewai"

_cited: list[str] = []


@tool("search_docs")
def search_docs(query: str) -> str:
    """Search the Laravel documentation. Returns numbered passages with their
    source. Covers official Laravel docs only."""
    passages = search(query)
    _cited.extend(p.cite() for p in passages if p.cite() not in _cited)
    return format_passages(passages)


llm = LLM(model=f"ollama/{config.MODEL}", base_url=config.OLLAMA_URL, temperature=0)

researcher = Agent(
    role="Documentation researcher",
    goal="Find the passages in the Laravel docs that answer the question",
    backstory="You search the docs and report exactly what they say, with sources. /no_think",
    tools=[search_docs], llm=llm, max_iter=3, allow_delegation=False, verbose=False,
)
writer = Agent(
    role="Answer writer",
    goal="Write a short answer grounded only in the researcher's passages",
    backstory=SYSTEM, llm=llm, allow_delegation=False, verbose=False,
)

research = Task(
    description="Search the docs for: {question}. Report the relevant passages verbatim with their [n] numbers and sources.",
    expected_output="Numbered passages with sources, or 'nothing relevant'.",
    agent=researcher,
)
write = Task(
    description=(f"Using only the research, answer: {{question}}. Cite like [1]. "
                 f"If the research does not answer it, reply exactly {NOT_IN_DOCS}."),
    expected_output="Two to five sentences with citations, or NOT_IN_DOCS.",
    agent=writer, context=[research],
)
crew = Crew(agents=[researcher, writer], tasks=[research, write], process=Process.sequential, verbose=False)


def answer(question: str) -> Result:
    _cited.clear()
    out = crew.kickoff(inputs={"question": question})
    return finish(out.raw, list(_cited))
