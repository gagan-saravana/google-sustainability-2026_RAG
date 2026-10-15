"""Chunker tests: pure Python, no API key or network needed. Run with `pytest`."""
from pathlib import Path

from src import config
from src.chunking import chunk_markdown, clean_inline, looks_like_toc, split_long_text, table_to_texts


def _chunk(tmp_path: Path, md: str):
    p = tmp_path / "doc.md"
    p.write_text(md, encoding="utf-8")
    return chunk_markdown(p)


def test_clean_inline_removes_markup():
    assert clean_inline("12 GW **clean**<sup>1</sup> and tCO<sub>2</sub>e") == "12 GW clean and tCO2e"


def test_breadcrumb_and_noise_removal(tmp_path):
    md = (
        "## **Energy**\n\nphoto: solar panels\n\n"
        "### **Nuclear**\n\nGoogle signed an agreement for new nuclear capacity to power data centers.\n"
    )
    chunks = _chunk(tmp_path, md)
    assert [c.section_path for c in chunks] == ["Energy > Nuclear"]
    assert "photo" not in chunks[0].text


def test_figure_heading_does_not_reset_breadcrumb(tmp_path):
    md = (
        "## Energy\n\n### Nuclear\n\n## **Figure 3.**\n\n"
        "Reactors will supply carbon-free electricity around the clock to our data centers.\n"
    )
    chunks = _chunk(tmp_path, md)
    assert chunks[0].section_path == "Energy > Nuclear"


def test_caption_attached_to_table_and_toc_dropped(tmp_path):
    md = (
        "## Water\n\n#### **Figure 9. Water use** — Legend: Year; Megaliters\n\n"
        "<table><tr><th>Year</th><th>ML</th></tr>\n<tr><td>2024</td><td>10</td></tr></table>\n\n"
        "<table><tr><td>35</td><td>Cooling with care</td></tr></table>\n"
    )
    chunks = _chunk(tmp_path, md)
    assert len(chunks) == 1 and chunks[0].type == "table"
    assert chunks[0].text.startswith("Figure 9. Water use")
    assert "2024 | 10" in chunks[0].text


def test_large_table_split_repeats_header():
    rows = [["Year", "Value"]] + [[str(2000 + i), "x" * 40] for i in range(30)]
    pieces = table_to_texts(rows, None, max_chars=300)
    assert len(pieces) > 1 and all(p.startswith("Year | Value") for p in pieces)


def test_long_paragraph_split_with_overlap():
    sentences = [f"Sentence number {i} says something useful about energy." for i in range(40)]
    windows = split_long_text(" ".join(sentences), max_chars=400, overlap_chars=120)
    assert len(windows) > 1 and all(len(w) <= 400 + 60 for w in windows)
    # the last sentence of window n reappears at the start of window n+1
    assert windows[0].split(". ")[-1].rstrip(".") in windows[1]


def test_endnotes_are_tagged(tmp_path):
    md = "## Endnotes\n\n1 PUE is a standard industry ratio that compares overhead energy with IT energy.\n"
    assert _chunk(tmp_path, md)[0].type == "endnote"


def test_toc_detector_ignores_data_tables():
    assert looks_like_toc([["35", "Water"], ["40", "Waste"]])
    assert not looks_like_toc([["2009", "1.19"], ["2010", "1.18"]])


def test_real_report_sanity():
    chunks = chunk_markdown(config.DATA_PATH)
    assert 300 < len(chunks) < 1500
    assert len({c.id for c in chunks}) == len(chunks)                 # ids unique
    assert all(c.text and "<sup>" not in c.text and "**" not in c.text for c in chunks)
    assert {c.type for c in chunks} <= {"text", "table", "figure", "endnote"}
