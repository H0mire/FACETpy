#!/usr/bin/env python3
"""Minimal syntax smoke test for documentation Python snippets.

This script extracts Python code blocks from selected .rst/.md docs files and
compiles each snippet to catch syntax errors early in CI.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import textwrap
from dataclasses import dataclass

DEFAULT_FILES = [
    "README.md",
    "QUICKSTART.md",
    "docs/source/getting_started/quickstart.rst",
    "docs/source/getting_started/tutorial.rst",
    "docs/source/getting_started/examples.rst",
    "docs/source/user_guide/helper_cookbook.rst",
    "docs/source/user_guide/pipelines.rst",
    "docs/source/user_guide/processors.rst",
]


@dataclass
class Snippet:
    path: pathlib.Path
    start_line: int
    code: str


def _extract_rst_snippets(path: pathlib.Path, text: str) -> list[Snippet]:
    snippets: list[Snippet] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\.\.\s+code-block::\s*python\s*$", line):
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1

            if i >= len(lines):
                break

            block_start = i
            block: list[str] = []
            while i < len(lines):
                current = lines[i]
                if current.strip() == "":
                    block.append("")
                    i += 1
                    continue

                if current.startswith("   "):
                    block.append(current[3:])
                    i += 1
                    continue

                break

            code = "\n".join(block).strip("\n")
            if code:
                snippets.append(Snippet(path=path, start_line=block_start + 1, code=code))
            continue

        i += 1

    return snippets


def _extract_md_snippets(path: pathlib.Path, text: str) -> list[Snippet]:
    snippets: list[Snippet] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("```python"):
            start_line = i + 2
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            code = "\n".join(block).strip("\n")
            if code:
                snippets.append(Snippet(path=path, start_line=start_line, code=code))
        i += 1

    return snippets


def extract_snippets(path: pathlib.Path) -> list[Snippet]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rst":
        return _extract_rst_snippets(path, text)
    if path.suffix == ".md":
        return _extract_md_snippets(path, text)
    return []


def compile_snippets(snippets: list[Snippet]) -> list[str]:
    errors: list[str] = []
    for idx, snippet in enumerate(snippets, start=1):
        name = f"{snippet.path}:{snippet.start_line}:snippet-{idx}"
        code = textwrap.dedent(snippet.code).strip("\n")
        if not code:
            continue
        try:
            compile(code, name, "exec")
        except SyntaxError as exc:
            errors.append(
                f"{snippet.path}:{snippet.start_line}: SyntaxError: {exc.msg} "
                f"(line {exc.lineno}, offset {exc.offset})"
            )
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Syntax smoke test for docs Python snippets")
    parser.add_argument("files", nargs="*", help="Files to scan (defaults to key docs pages)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    raw_files = args.files or DEFAULT_FILES
    files = [pathlib.Path(p) for p in raw_files]

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing file: {path}", file=sys.stderr)
        return 2

    all_snippets: list[Snippet] = []
    for path in files:
        all_snippets.extend(extract_snippets(path))

    if not all_snippets:
        print("No Python snippets found.")
        return 0

    errors = compile_snippets(all_snippets)
    if errors:
        print("Documentation snippet smoke test failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Compiled {len(all_snippets)} Python snippets from {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
