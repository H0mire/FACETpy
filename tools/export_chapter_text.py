#!/usr/bin/env python3
"""Export the thesis prose from chapter HTML files as plain text.

Only the chapter title, section headings, introductory paragraph, and content
inside ``div.prose`` are exported. Navigation, tables, figures, captions,
evidence boxes, registers, buttons, CSS, and JavaScript are ignored.
"""

from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML_FILES = (
    REPOSITORY_ROOT / "output" / "kapitel_4" / "kapitel_4.html",
    REPOSITORY_ROOT / "output" / "kapitel_5" / "kapitel_5.html",
)


def normalize_text(value: str) -> str:
    """Collapse HTML whitespace while keeping the visible text unchanged."""

    return re.sub(r"\s+", " ", value).strip()


class ChapterTextParser(HTMLParser):
    """Collect headings and thesis prose while ignoring presentation content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buffer: list[str] = []
        self._capture_tag: str | None = None
        self._in_header = False
        self._in_main = False
        self._in_nav = False
        self._div_depth = 0
        self._prose_depth: int | None = None
        self._section_depth = 0
        self._register_depth: int | None = None

    @property
    def _in_prose(self) -> bool:
        return self._prose_depth is not None

    @property
    def _in_register(self) -> bool:
        return self._register_depth is not None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)

        if tag == "header":
            self._in_header = True
        elif tag == "main":
            self._in_main = True
        elif tag == "nav":
            self._in_nav = True
        elif tag == "div":
            self._div_depth += 1
            classes = (attributes.get("class") or "").split()
            if "prose" in classes:
                self._prose_depth = self._div_depth
        elif tag == "section":
            self._section_depth += 1
            classes = (attributes.get("class") or "").split()
            if "register" in classes:
                self._register_depth = self._section_depth

        should_capture = (
            tag == "h1" and self._in_header and not self._in_nav
        ) or (
            tag == "p" and self._in_header and not self._in_nav
        ) or (
            tag in {"h2", "h3"} and self._in_main and not self._in_register
        ) or (tag == "p" and self._in_prose)

        if should_capture:
            self._capture_tag = tag
            self._buffer = []
        elif tag == "br" and self._capture_tag is not None:
            self._buffer.append(" ")

    def handle_data(self, data: str) -> None:
        if self._capture_tag is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._capture_tag:
            text = normalize_text("".join(self._buffer))
            if text:
                self.blocks.append(text)
            self._capture_tag = None
            self._buffer = []

        if tag == "header":
            self._in_header = False
        elif tag == "main":
            self._in_main = False
        elif tag == "nav":
            self._in_nav = False
        elif tag == "div":
            if self._prose_depth == self._div_depth:
                self._prose_depth = None
            self._div_depth -= 1
        elif tag == "section":
            if self._register_depth == self._section_depth:
                self._register_depth = None
            self._section_depth -= 1


def extract_chapter_text(html: str) -> str:
    """Return the selected chapter content as unformatted plain text."""

    parser = ChapterTextParser()
    parser.feed(html)
    parser.close()
    return "\n\n".join(parser.blocks) + "\n"


def output_path_for(html_path: Path, output_dir: Path | None) -> Path:
    destination = output_dir if output_dir is not None else html_path.parent
    return destination / f"{html_path.stem}.txt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract headings and prose from chapter HTML files while omitting "
            "tables, figures, captions, evidence boxes, and navigation."
        )
    )
    parser.add_argument(
        "html_files",
        nargs="*",
        type=Path,
        help="HTML files to export; defaults to the Chapter 4 and 5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional common output directory; defaults to each HTML directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    html_files = tuple(args.html_files) or DEFAULT_HTML_FILES
    output_dir = args.output_dir.resolve() if args.output_dir else None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for supplied_path in html_files:
        html_path = supplied_path.resolve()
        if not html_path.is_file():
            raise FileNotFoundError(f"HTML file not found: {html_path}")

        output_path = output_path_for(html_path, output_dir)
        plain_text = extract_chapter_text(html_path.read_text(encoding="utf-8"))
        output_path.write_text(plain_text, encoding="utf-8")
        print(f"{html_path} -> {output_path} ({len(plain_text.split())} words)")


if __name__ == "__main__":
    main()
