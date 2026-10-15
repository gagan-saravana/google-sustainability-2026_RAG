"""Question-condensing tests with a fake generator (no API)."""
from src.condense import SYSTEM_PROMPT, condense_question

HISTORY = [("What was the 2025 PUE?", "The fleet-wide average PUE was 1.09 [2].")]


def test_first_turn_skips_the_model():
    def boom(*a):
        raise AssertionError("must not be called")

    assert condense_question([], "What is the PUE?", generate_fn=boom) == "What is the PUE?"


def test_followup_is_rewritten_and_prompt_has_history_without_citations():
    seen = {}

    def fake(system, user):
        seen["system"], seen["user"] = system, user
        return '"What was the 2025 water replenishment figure?"\nextra line'

    out = condense_question(HISTORY, "and for water?", generate_fn=fake)
    assert out == "What was the 2025 water replenishment figure?"
    assert seen["system"] == SYSTEM_PROMPT
    assert "User: What was the 2025 PUE?" in seen["user"] and "[2]" not in seen["user"]
    assert "Follow-up question: and for water?" in seen["user"]


def test_failure_or_empty_output_falls_back_to_original():
    def fail(*a):
        raise RuntimeError("429")

    assert condense_question(HISTORY, "and water?", generate_fn=fail) == "and water?"
    assert condense_question(HISTORY, "and water?", generate_fn=lambda s, u: "  ") == "and water?"
    assert condense_question(HISTORY, "and water?", generate_fn=lambda s, u: "x" * 5000) == "and water?"
