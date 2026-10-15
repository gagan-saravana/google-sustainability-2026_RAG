"""Retrieval + synthesis tests with fakes: no API key, Chroma or network needed."""
from src.retrieve import Retrieved, retrieve
from src.synthesize import NOT_FOUND, SYSTEM_PROMPT, answer_question, build_prompt, extract_citations


def _src(i, type_="text"):
    return Retrieved(id=f"id{i}", text=f"passage text {i}", section_path=f"Energy > Topic {i}", type=type_, similarity=0.9)


class FakeCollection:
    def __init__(self):
        self.kwargs = None

    def query(self, **kwargs):
        self.kwargs = kwargs
        return {
            "ids": [["a", "b"]],
            "documents": [["doc a", "doc b"]],
            "metadatas": [[{"section_path": "Energy", "type": "text"}, {"section_path": "Water", "type": "table"}]],
            "distances": [[0.1, 0.35]],
        }


def test_retrieve_filters_endnotes_and_converts_distance():
    col = FakeCollection()
    out = retrieve("q?", k=2, collection=col, embed_query_fn=lambda q: [0.0])
    assert col.kwargs["where"] == {"type": {"$ne": "endnote"}}
    assert col.kwargs["n_results"] == 2
    assert [r.similarity for r in out] == [0.9, 0.65]
    assert out[1].type == "table" and out[1].section_path == "Water"


def test_retrieve_can_include_endnotes():
    col = FakeCollection()
    retrieve("q?", collection=col, embed_query_fn=lambda q: [0.0], include_endnotes=True)
    assert col.kwargs["where"] is None


def test_build_prompt_numbers_sources_and_marks_tables():
    prompt = build_prompt("How much?", [_src(1), _src(2, "table")])
    assert "[1] Section: Energy > Topic 1 (text)" in prompt
    assert "[2] Section: Energy > Topic 2 (table)" in prompt
    assert prompt.rstrip().endswith("Answer:") and "Question: How much?" in prompt


def test_extract_citations_variants_and_range():
    assert extract_citations("A [1]. B [2][3]. C [1, 4]. D [9].", n_sources=4) == [1, 2, 3, 4]
    assert extract_citations("no citations", n_sources=3) == []


def test_answer_question_end_to_end_with_fakes():
    seen = {}

    def fake_generate(system, user):
        seen["system"], seen["user"] = system, user
        return "Google procured 12 GW [2]."

    res = answer_question("How much?", retriever=lambda q, **kw: [_src(1), _src(2)], generate_fn=fake_generate)
    assert seen["system"] == SYSTEM_PROMPT and "passage text 2" in seen["user"]
    assert res.cited == [2] and res.cited_sources[0][1].id == "id2"


def test_answer_question_without_sources_does_not_call_the_model():
    def boom(*a):
        raise AssertionError("generate should not be called")

    res = answer_question("?", retriever=lambda q, **kw: [], generate_fn=boom)
    assert res.text == NOT_FOUND and res.sources == []
