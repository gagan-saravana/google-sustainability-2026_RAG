# this is only for testing question-answering in the terminal ! 

from __future__ import annotations

import argparse
import textwrap

from src import config
from src.synthesize import answer_question


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=config.TOP_K, help="passages to retrieve")
    ap.add_argument("--include-endnotes", action="store_true")
    ap.add_argument("--show-chunks", action="store_true", help="print every retrieved passage, not just cited ones")
    args = ap.parse_args()

    result = answer_question(args.question, k=args.k, include_endnotes=args.include_endnotes)

    print(f"\nAnswer:\n{textwrap.fill(result.text, 100, replace_whitespace=False)}\n")
    shown = list(enumerate(result.sources, 1)) if args.show_chunks else result.cited_sources
    print("Sources" + (" (all retrieved)" if args.show_chunks else " (cited)") + ":")
    for n, s in shown:
        mark = "*" if n in result.cited else " "
        print(f" {mark}[{n}] sim={s.similarity:.3f} [{s.type}] {s.section_path}")
        if args.show_chunks:
            print("      " + s.text[:200].replace("\n", " ") + ("..." if len(s.text) > 200 else ""))


if __name__ == "__main__":
    main()
