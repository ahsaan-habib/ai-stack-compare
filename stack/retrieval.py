"""The shared retrieval layer. Every implementation calls this, so the only
thing that differs between them is the framework around the model."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from . import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Passage:
    id: str
    source: str
    heading: str
    text: str
    score: float

    def cite(self) -> str:
        return f"{self.source}#{self.heading}"


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=config.INDEX_DIR)
    return client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def _sections(path: Path) -> list[tuple[str, str]]:
    out, heading, buf = [], "", []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#{2,4}\s+(.*)", line)
        if m:
            if buf:
                out.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        elif not line.startswith("<a name="):
            buf.append(line)
    if buf:
        out.append((heading, "\n".join(buf).strip()))
    return [(h, t) for h, t in out if len(t.split()) > 20]


def build_index(corpus: str = "corpus") -> int:
    col = _collection()
    ids, docs, metas = [], [], []
    for path in sorted(Path(corpus).glob("*.md")):
        for i, (heading, text) in enumerate(_sections(path)):
            # sections are short in these docs; cap the long ones rather than chunking
            ids.append(f"{path.stem}-{i}")
            docs.append(text[:3000])
            metas.append({"source": path.name, "heading": heading})
    for s in range(0, len(ids), 128):
        col.upsert(ids=ids[s:s + 128], documents=docs[s:s + 128], metadatas=metas[s:s + 128],
                   embeddings=_embedder().encode(docs[s:s + 128], normalize_embeddings=True).tolist())
    return len(ids)


def search(query: str, k: int = config.TOP_K) -> list[Passage]:
    vec = _embedder().encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=[vec], n_results=k)
    return [Passage(id=i, source=m["source"], heading=m["heading"], text=d, score=1 - dist)
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i}] ({p.cite()})\n{p.text}" for i, p in enumerate(passages, 1))


if __name__ == "__main__":
    print(f"indexed {build_index()} sections")
