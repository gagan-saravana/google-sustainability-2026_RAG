# helper file for index.py ((i) chunks the LlamaCloud markdown)

from __future__ import annotations

import hashlib
import html
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from src import config

# LlamaCloud describes images as lines like "photo: solar panels on a wooden structure".
NOISE_LINE = re.compile(
    r"^\s*(logo|photo|icon|map|screenshot|image|drawing)\s*:", re.IGNORECASE
)
SUP = re.compile(r"<sup>.*?</sup>", re.IGNORECASE | re.DOTALL)   # footnote markers (all numeric in this file)
INLINE_TAG = re.compile(r"</?(?:sub|u|strong|b|em|i|span|mark)\b[^>]*>", re.IGNORECASE)
PAGE_NUMBER_ONLY = re.compile(r"^\d{1,3}$")   # stray page numbers on their own line
MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
BOLD = re.compile(r"\*\*(.+?)\*\*")


def clean_inline(text: str) -> str:
    """Normalise one line/cell of text.  tCO<sub>2</sub>e -> tCO2e,  'energy.<sup>1</sup>' -> 'energy.'"""
    text = SUP.sub("", text)
    text = INLINE_TAG.sub("", text)      # removing <sub> tags leaves the digit: CO2
    text = MD_LINK.sub(r"\1", text)
    text = BOLD.sub(r"\1", text).replace("**", "")
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()

# Data model

@dataclass
class Chunk:
    id: str             # stable hash, so re-indexing can skip chunks that are already stored
    text: str           # what the answer synthesizer will read
    section_path: str   # breadcrumb, e.g. "Energy > Energy for our data centers > Nuclear"
    type: str           # "text" | "table" | "figure" | "endnote"
    chunk_index: int    # position in the document (useful for debugging / ordering)

    @property
    def title(self) -> str:
        """Used as the `title:` field of the Gemini embedding document format."""
        return self.section_path[:200] or "none"

    @property
    def metadata(self) -> dict:
        """Chroma metadata values must be str / int / float / bool."""
        return {
            "section_path": self.section_path,
            "top_section": self.section_path.split(" > ")[0] if self.section_path else "",
            "type": self.type,
            "chunk_index": self.chunk_index,
            "n_chars": len(self.text),
        }


# HTML tables -> compact text rows
class _TableParser(HTMLParser):
    """Collect <tr>/<td>/<th> into rows of cell strings (rowspan/colspan are ignored)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._sup_depth = 0   # skip footnote numbers inside cells

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "sup":
            self._sup_depth += 1

    def handle_endtag(self, tag):
        if tag == "sup":
            self._sup_depth = max(0, self._sup_depth - 1)
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(clean_inline(" ".join("".join(self._cell).split())))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            while self._row and not self._row[-1]:      # trim trailing empty cells
                self._row.pop()
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None and self._sup_depth == 0:
            self._cell.append(data)


def parse_table(html_block: str) -> list[list[str]]:
    parser = _TableParser()
    parser.feed(html_block)
    return parser.rows


def looks_like_toc(rows: list[list[str]]) -> bool:
    """
    Section-divider pages carry a mini table of contents: rows like ['82', 'Methodology'].
    They point at page numbers instead of holding facts, so they only add retrieval noise.
    Data tables don't match: their first column is a year (>200) or a label, not a small integer.
    """
    return bool(rows) and all(
        len(r) >= 2 and r[0].isdigit() and int(r[0]) <= 200 and re.search(r"[A-Za-z]", r[1])
        for r in rows
    )


def table_to_texts(rows: list[list[str]], caption: str | None, max_chars: int) -> list[str]:
    """
    Render rows as 'cell | cell | cell' lines.  A table that fits in max_chars
    stays ONE chunk; a larger one is split by rows and the header row (plus the
    caption) is repeated in each piece, so every piece is self-explanatory.
    """
    if not rows:
        return []
    fmt = lambda r: " | ".join(r)  # noqa: E731
    head = (f"{caption}\n" if caption else "") + fmt(rows[0])
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = len(head)
    for row in rows[1:]:
        line = fmt(row)
        if cur and cur_len + 1 + len(line) > max_chars:
            pieces.append(head + "\n" + "\n".join(cur))
            cur, cur_len = [], len(head)
        cur.append(line)
        cur_len += 1 + len(line)
    pieces.append(head + ("\n" + "\n".join(cur) if cur else ""))
    return pieces


# Splitting an oversized paragraph
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])|\n")


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Sentence-aware windows. Each new window starts with the last few sentences
    of the previous one (up to overlap_chars) so a fact straddling a boundary
    still appears intact in at least one chunk.
    """
    sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    windows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        if cur and cur_len + len(sent) + 1 > max_chars:
            windows.append(" ".join(cur))
            carry: list[str] = []
            carried = 0
            for prev in reversed(cur):
                if carried + len(prev) > overlap_chars:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            cur, cur_len = carry, carried
        cur.append(sent)
        cur_len += len(sent) + 1
    if cur:
        windows.append(" ".join(cur))
    return windows


# Headings
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
BARE_FIGURE_LABEL = re.compile(r"^Figure\s+\d+\.?$", re.IGNORECASE)


def is_figure_heading(title: str) -> bool:
    """
    LlamaCloud turns chart captions into headings, e.g.
        '## Figure 3.'                                   (bare label)
        '#### Figure 4. Gemini efficiency ... — ...'     (caption)
        '### Terawatt-hours by Year — ... Legend: ...'   (caption, no 'Figure')
    These must NOT enter the breadcrumb: a level-2 'Figure 3.' would otherwise
    reset the section path and every later chunk would be filed under 'Figure 3.'.
    """
    return bool(
        re.match(r"^Figure\s+\d+", title, re.IGNORECASE)
        or re.search(r"\bLegend:|\bcategories:|\bseries:", title)
    )


# The chunker
class _Chunker:
    """Single pass over the markdown, emitting (section_path, type, text) drafts."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []   # (heading level, title)
        self.buffer: list[str] = []              # paragraphs waiting to be packed
        self.buffer_len = 0
        self.pending_caption: str | None = None  # figure caption waiting for its table
        self.skip = False                        # inside the table of contents
        self.drafts: list[tuple[str, str, str]] = []

    # -- helpers --------------------------------------------------------------
    @property
    def path(self) -> str:
        return " > ".join(title for _, title in self.stack)

    def _section_type(self) -> str:
        return "endnote" if any("endnote" in t.lower() for _, t in self.stack) else "text"

    def _emit(self, text: str, type_: str) -> None:
        text = text.strip()
        if type_ in ("text", "endnote") and len(text) < config.MIN_CHARS:
            return
        if text:
            self.drafts.append((self.path, type_, text))

    # -- flushing -------------------------------------------------------------
    def flush_text(self) -> None:
        if self.buffer:
            self._emit("\n\n".join(self.buffer), self._section_type())
        self.buffer, self.buffer_len = [], 0

    def flush_caption(self) -> None:
        """A caption that never met a data table is still searchable, so keep it as a 'figure' chunk."""
        if self.pending_caption:
            self._emit(self.pending_caption, "figure")
        self.pending_caption = None

    # -- block handlers ---------------------------------------------------------
    def heading(self, level: int, raw_title: str) -> None:
        title = clean_inline(raw_title)
        self.flush_text()
        self.flush_caption()
        if is_figure_heading(title):
            if not BARE_FIGURE_LABEL.match(title):
                self.pending_caption = title
            return                                  # never touches the breadcrumb
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))
        self.skip = title.lower() == "contents"    # drop the table of contents

    def paragraph(self, text: str) -> None:
        if self.skip or not text.strip() or PAGE_NUMBER_ONLY.match(text.strip()):
            return
        self.flush_caption()
        if len(text) > config.MAX_CHARS:            # too big on its own: split by sentence
            self.flush_text()
            for window in split_long_text(text, config.MAX_CHARS, config.OVERLAP_CHARS):
                self._emit(window, self._section_type())
            return
        if self.buffer and self.buffer_len + len(text) + 2 > config.TARGET_CHARS:
            self.flush_text()
        self.buffer.append(text)
        self.buffer_len += len(text) + 2

    def table(self, html_block: str) -> None:
        if self.skip:
            return
        rows = parse_table(html_block)
        caption, self.pending_caption = self.pending_caption, None
        if caption is None and looks_like_toc(rows):
            return
        self.flush_text()
        for piece in table_to_texts(rows, caption, config.TABLE_MAX_CHARS):
            self._emit(piece, "table")

    def finish(self) -> None:
        self.flush_text()
        self.flush_caption()


def chunk_markdown(path: Path | str = config.DATA_PATH) -> list[Chunk]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    ck = _Chunker()
    para: list[str] = []

    def end_paragraph() -> None:
        if para:
            ck.paragraph("\n".join(para))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if NOISE_LINE.match(line):
            i += 1
            continue
        if line.lstrip().lower().startswith("<table"):
            end_paragraph()
            block = [line]
            while "</table>" not in block[-1].lower() and i + 1 < len(lines):
                i += 1
                block.append(lines[i])
            ck.table("\n".join(block))
        elif (m := HEADING.match(line)):
            end_paragraph()
            ck.heading(len(m.group(1)), m.group(2))
        elif not line.strip():
            end_paragraph()
        else:
            cleaned = clean_inline(line)
            if cleaned:
                para.append(cleaned)
        i += 1
    end_paragraph()
    ck.finish()

    # Assign stable ids: hash of (path, type, text); a numeric suffix disambiguates exact repeats.
    chunks: list[Chunk] = []
    seen: Counter[str] = Counter()
    for idx, (section_path, type_, text) in enumerate(ck.drafts):
        digest = hashlib.sha1(f"{section_path}|{type_}|{text}".encode()).hexdigest()[:16]
        seen[digest] += 1
        cid = digest if seen[digest] == 1 else f"{digest}-{seen[digest]}"
        chunks.append(Chunk(cid, text, section_path, type_, idx))
    return chunks


# Inspection CLI:  python -m src.chunking
def print_stats(chunks: list[Chunk]) -> None:
    sizes = [len(c.text) for c in chunks]
    by_type = Counter(c.type for c in chunks)
    print(f"{len(chunks)} chunks | by type: {dict(by_type)}")
    print(
        f"chars: min={min(sizes)} median={int(statistics.median(sizes))} "
        f"mean={int(statistics.mean(sizes))} max={max(sizes)}  "
        f"(~{sum(sizes) // 4:,} tokens in total to embed)"
    )


def main() -> None:
    chunks = chunk_markdown()
    print_stats(chunks)
    with open(config.CHUNKS_DUMP, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.id, **c.metadata, "text": c.text}, ensure_ascii=False) + "\n")
    print(f"Full dump written to {config.CHUNKS_DUMP} (open it to eyeball the chunks).")
    for c in chunks[len(chunks) // 3 :: max(1, len(chunks) // 3)][:3]:
        print(f"\n--- [{c.type}] {c.section_path}\n{c.text[:350]}")


if __name__ == "__main__":
    main()
