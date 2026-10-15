# generates a grounded (no hallucinations), cited response based on the retrieved embeddings.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from src import config
from src.retrieve import Retrieved, retrieve

SYSTEM_PROMPT = """\
You answer questions about Google's 2026 Environmental Report using ONLY the numbered \
context passages provided by the user.

Rules:
- Use only facts stated in the passages. If they do not contain the answer, say you could \
not find it in the report. Never guess and never use outside knowledge.
- Cite every factual claim with the passage number in square brackets, e.g. [2] or [1][3].
- Copy numbers, units and years exactly as written.
- Passages marked (table) contain rows written as 'a | b | c'; the first row is the column headers.
- If passages conflict or the question is ambiguous, say so briefly.
- Be concise: 2-5 sentences, unless the question asks for a list or a comparison.
"""

NOT_FOUND = "I couldn't find anything relevant to that in the report."


@dataclass
class Answer:
    text: str
    sources: list[Retrieved]   # all passages shown to the model, in citation order ([1] = sources[0])
    cited: list[int]           # 1-based numbers the answer actually cites

    @property
    def cited_sources(self) -> list[tuple[int, Retrieved]]:
        return [(n, self.sources[n - 1]) for n in self.cited]


def build_prompt(question: str, sources: list[Retrieved]) -> str:
    blocks = [
        f"[{i}] Section: {s.section_path or 'n/a'} ({s.type})\n{s.text}"
        for i, s in enumerate(sources, start=1)
    ]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question.strip()}\n\nAnswer:"


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer_text: str, n_sources: int) -> list[int]:
    """Passage numbers cited in the answer ('[1]', '[2][3]', '[1, 4]'), sorted, out-of-range ignored."""
    found = {int(n) for group in _CITATION.findall(answer_text) for n in re.split(r"\s*,\s*", group)}
    return sorted(n for n in found if 1 <= n <= n_sources)


def answer_question(
    question: str,
    k: int = config.TOP_K,
    include_endnotes: bool = False,
    retriever: Callable[..., list[Retrieved]] = retrieve,
    generate_fn: Callable[[str, str], str] | None = None,
) -> Answer:
    sources = retriever(question, k=k, include_endnotes=include_endnotes)
    if not sources:
        return Answer(NOT_FOUND, [], [])
    if generate_fn is None:
        from src.gemini_client import generate_text as generate_fn
    text = generate_fn(SYSTEM_PROMPT, build_prompt(question, sources)) or NOT_FOUND
    return Answer(text, sources, extract_citations(text, len(sources)))
