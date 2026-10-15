# condsenses previous conversation turns into a single standalone question to enable follow-up questions.

from __future__ import annotations

import re
from typing import Callable

from src import config

SYSTEM_PROMPT = """\
You rewrite follow-up questions for a search system.
Given a short conversation and a follow-up question, rewrite the follow-up as ONE \
standalone question that can be understood without the conversation, keeping all \
names, years and numbers that matter. If the follow-up is already standalone, return it unchanged.
Output only the question: no preface, no quotes, no answer."""

_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def _build_prompt(history: list[tuple[str, str]], question: str) -> str:
    turns = []
    for q, a in history:
        a = _CITATION.sub("", a).strip()
        turns.append(f"User: {q.strip()}\nAssistant: {a[:500]}")
    return "Conversation:\n" + "\n\n".join(turns) + f"\n\nFollow-up question: {question.strip()}\n\nStandalone question:"


def condense_question(
    history: list[tuple[str, str]],
    question: str,
    generate_fn: Callable[[str, str], str] | None = None,
) -> str:
    if not history:
        return question
    if generate_fn is None:
        from src.gemini_client import generate_text as generate_fn
    try:
        out = generate_fn(SYSTEM_PROMPT, _build_prompt(history, question))
    except Exception:          # never let the helper call break the turn
        return question
    lines = (out or "").strip().splitlines()
    out = lines[0].strip().strip("\"'").strip() if lines else ""   # first line only, then drop wrapping quotes
    if not out or len(out) > config.MAX_QUESTION_CHARS * 2:
        return question
    return out
