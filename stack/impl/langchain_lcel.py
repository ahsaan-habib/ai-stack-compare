"""LangChain as glue: a retriever, a prompt template, a chat model, a parser,
piped together with LCEL. Same retrieval, same prompt as plain.py."""
from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from .. import config
from ..common import SYSTEM, Result, finish
from ..retrieval import search

NAME = "langchain"


class SharedRetriever(BaseRetriever):
    """Wraps stack.retrieval so LangChain's own vector-store classes aren't in
    the comparison — only the chain around the model is."""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(page_content=p.text, metadata={"cite": p.cite()}) for p in search(query)]


def _format(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({d.metadata['cite']})\n{d.page_content}" for i, d in enumerate(docs, 1))


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "{context}\n\nQuestion: {question}"),
])
llm = ChatOllama(model=config.MODEL, base_url=config.OLLAMA_URL, temperature=0)

chain = (
    RunnableParallel(docs=SharedRetriever(), question=RunnablePassthrough())
    | RunnableParallel(
        text=RunnableLambda(lambda x: {"context": _format(x["docs"]), "question": x["question"]})
        | prompt | llm | StrOutputParser(),
        sources=RunnableLambda(lambda x: [d.metadata["cite"] for d in x["docs"]]),
    )
)


def answer(question: str) -> Result:
    out = chain.invoke(question)
    return finish(out["text"], out["sources"])
