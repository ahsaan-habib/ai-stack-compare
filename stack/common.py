"""What every implementation must produce, and the one prompt they all use."""
from __future__ import annotations

from dataclasses import dataclass, field

NOT_IN_DOCS = "NOT_IN_DOCS"

# "/no_think" is qwen3's soft switch to skip the reasoning block. It goes in the
# prompt rather than an API flag because not every framework exposes the flag.
SYSTEM = f"""You answer questions about Laravel using ONLY the numbered passages.
Cite passages inline like [1] or [2]. Two to five sentences.
If the passages do not answer the question, reply with exactly {NOT_IN_DOCS}. /no_think"""


@dataclass
class Result:
    answer: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False


def finish(text: str, sources: list[str]) -> Result:
    text = text.replace("<think>", "").replace("</think>", "").strip()
    if NOT_IN_DOCS in text:
        return Result("The docs I have don't cover this.", [], refused=True)
    return Result(text, sources)
