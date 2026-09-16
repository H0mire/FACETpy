"""Count pre-defined engineering indicators on two source trees.

The point of this tool is that the indicator set and its counting scope are
**fixed before either tree is measured** — they live in :data:`INDICATORS` and
:data:`SCOPE` below, not in a spreadsheet written after the numbers came out.
Without that, a refactoring comparison degenerates into picking whichever
metrics happen to favour the new code.

Scope (identical on both sides):

* **Counted:** the shipped library source only — ``src/FACET/**`` on the legacy
  side, ``src/facet/**`` on the current side.
* **Not counted:** tests, tooling, examples, notebooks, documentation, data
  files, and generated code. Test counts are reported separately and read from
  each tree's own test directory, because "how much is tested" is an indicator
  in its own right and must not inflate the source line count.

Every indicator is computed from the Python AST or from ``tokenize``, never from
a regular expression over source text, so a string containing the word ``class``
cannot be counted as a class.

Usage::

    .venv/bin/python tools/refactoring_comparison/engineering_indicators.py \\
        --legacy-root <path>/src/FACET --legacy-tests <path>/src/tests \\
        --current-root src/facet --current-tests tests \\
        --out output/refactoring_comparison
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

#: The counting scope, stated once and applied to both trees unchanged.
SCOPE = {
    "counted": "shipped library source (.py files under the given root, recursively)",
    "excluded": "tests, tooling, examples, notebooks, docs, data; __pycache__ and site-packages never enter a root",
    "tests_counted_separately": True,
    "line_definition": "code = a physical line carrying at least one token that is "
    "neither a comment, a docstring-only expression, nor whitespace",
    "rationale": "Indicators and scope are fixed before measurement so the comparison cannot be tuned to its outcome.",
}

#: Indicator id -> (human label, direction, what it is evidence for).
#:
#: ``direction`` says which way is better and is part of the definition: an
#: indicator whose preferred direction is decided after seeing the numbers is
#: not an indicator.
INDICATORS = {
    "modules": ("Module (.py-Dateien)", "neutral", "Größe und Gliederung der Codebasis"),
    "code_lines": ("Code-Zeilen", "neutral", "Umfang; allein kein Qualitätsmaß"),
    "comment_lines": ("Kommentarzeilen", "höher", "Erklärungsdichte"),
    "comment_ratio_pct": ("Kommentaranteil (%)", "höher", "Erklärungsdichte relativ zum Umfang"),
    "classes": ("Klassen", "neutral", "Gliederung"),
    "functions": ("Funktionen und Methoden", "neutral", "Gliederung"),
    "public_symbols": ("Öffentliche Symbole auf Modulebene", "neutral", "API-Oberfläche"),
    "docstring_coverage_pct": (
        "Docstring-Abdeckung (%)",
        "höher",
        "Anteil öffentlicher Module/Klassen/Funktionen mit Docstring",
    ),
    "typed_params_pct": ("Typannotierte Parameter (%)", "höher", "Prüfbarkeit der Schnittstellen"),
    "typed_returns_pct": ("Typannotierte Rückgaben (%)", "höher", "Prüfbarkeit der Schnittstellen"),
    "mean_function_lines": ("Funktionslänge, Mittel", "niedriger", "Lesbarkeit"),
    "max_function_lines": ("Funktionslänge, Maximum", "niedriger", "Lesbarkeit"),
    "functions_over_50_lines": ("Funktionen > 50 Zeilen", "niedriger", "Lesbarkeit"),
    "mean_cyclomatic": ("Zyklomatische Komplexität, Mittel", "niedriger", "Testbarkeit"),
    "max_cyclomatic": ("Zyklomatische Komplexität, Maximum", "niedriger", "Testbarkeit"),
    "functions_cyclomatic_over_10": ("Funktionen mit Komplexität > 10", "niedriger", "Testbarkeit"),
    "max_methods_per_class": ("Methoden je Klasse, Maximum", "niedriger", "God-Class-Indikator"),
    "broad_excepts": ("Pauschale except-Blöcke", "niedriger", "bare except oder except Exception ohne re-raise"),
    "mutable_default_args": (
        "Veränderliche Default-Argumente",
        "niedriger",
        "bekannte Fehlerklasse (list/dict/set als Default)",
    ),
    "print_calls_in_library": (
        "print() in Bibliothekscode",
        "niedriger",
        "Bibliothek schreibt unkontrolliert auf stdout",
    ),
    "todo_comments": ("TODO/FIXME-Kommentare", "niedriger", "bekannte offene Stellen"),
    "non_importable_filenames": (
        "Nicht importierbare Modulnamen",
        "niedriger",
        "Dateiname ist kein gültiger Python-Bezeichner + .py",
    ),
    "test_files": ("Testdateien", "höher", "Prüfumfang"),
    "test_functions": ("Testfunktionen", "höher", "Prüfumfang"),
    "test_functions_per_100_code_lines": ("Testfunktionen je 100 Code-Zeilen", "höher", "Prüfdichte, größenbereinigt"),
}

DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.IfExp,
    ast.comprehension,
)


@dataclass
class TreeStats:
    """Raw counters for one source tree; derived ratios come from :meth:`indicators`."""

    root: Path
    modules: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    classes: int = 0
    functions: int = 0
    public_symbols: int = 0
    documented: int = 0
    documentable: int = 0
    params_total: int = 0
    params_typed: int = 0
    returns_total: int = 0
    returns_typed: int = 0
    function_lengths: list[int] = field(default_factory=list)
    complexities: list[int] = field(default_factory=list)
    methods_per_class: list[int] = field(default_factory=list)
    broad_excepts: int = 0
    mutable_default_args: int = 0
    print_calls: int = 0
    todo_comments: int = 0
    non_importable_filenames: int = 0
    test_files: int = 0
    test_functions: int = 0
    parse_failures: list[str] = field(default_factory=list)

    def indicators(self) -> dict[str, float | int]:
        lengths = self.function_lengths or [0]
        cx = self.complexities or [0]
        return {
            "modules": self.modules,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "comment_ratio_pct": round(100.0 * self.comment_lines / max(1, self.code_lines), 2),
            "classes": self.classes,
            "functions": self.functions,
            "public_symbols": self.public_symbols,
            "docstring_coverage_pct": round(100.0 * self.documented / max(1, self.documentable), 2),
            "typed_params_pct": round(100.0 * self.params_typed / max(1, self.params_total), 2),
            "typed_returns_pct": round(100.0 * self.returns_typed / max(1, self.returns_total), 2),
            "mean_function_lines": round(sum(lengths) / len(lengths), 2),
            "max_function_lines": max(lengths),
            "functions_over_50_lines": sum(1 for n in self.function_lengths if n > 50),
            "mean_cyclomatic": round(sum(cx) / len(cx), 2),
            "max_cyclomatic": max(cx),
            "functions_cyclomatic_over_10": sum(1 for n in self.complexities if n > 10),
            "max_methods_per_class": max(self.methods_per_class or [0]),
            "broad_excepts": self.broad_excepts,
            "mutable_default_args": self.mutable_default_args,
            "print_calls_in_library": self.print_calls,
            "todo_comments": self.todo_comments,
            "non_importable_filenames": self.non_importable_filenames,
            "test_files": self.test_files,
            "test_functions": self.test_functions,
            "test_functions_per_100_code_lines": round(100.0 * self.test_functions / max(1, self.code_lines), 2),
        }


def _is_identifier_module(path: Path) -> bool:
    """A file Python can actually import as a module of that name."""
    return path.suffix == ".py" and path.stem.isidentifier()


def _iter_sources(root: Path) -> list[Path]:
    """Every ``*.py``-ish file under root, including misnamed ones like ``x,py``.

    Misnamed files are included deliberately: they are part of the shipped
    source and their unimportability is itself an indicator.
    """
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.suffix == ".py" or p.name.endswith(",py"):
            out.append(p)
    return out


def _count_lines(text: str) -> tuple[int, int, int]:
    """(code, comment, todo) physical line counts via tokenize, not regex."""
    code_lines: set[int] = set()
    comment_lines = 0
    todo = 0
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return len([ln for ln in text.splitlines() if ln.strip()]), 0, 0
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            comment_lines += 1
            upper = tok.string.upper()
            if "TODO" in upper or "FIXME" in upper:
                todo += 1
        elif tok.type not in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
            tokenize.STRING,
        ):
            code_lines.add(tok.start[0])
    return len(code_lines), comment_lines, todo


def _complexity(node: ast.AST) -> int:
    """McCabe cyclomatic complexity of one function body."""
    n = 1
    for child in ast.walk(node):
        if isinstance(child, DECISION_NODES):
            n += 1
        elif isinstance(child, ast.BoolOp):
            n += len(child.values) - 1
    return n


def _reraises(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(x, ast.Raise) for x in ast.walk(handler))


def analyse_tree(root: Path, tests_root: Path | None) -> TreeStats:
    st = TreeStats(root=root)
    for path in _iter_sources(root):
        st.modules += 1
        if not _is_identifier_module(path):
            st.non_importable_filenames += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        code, comments, todo = _count_lines(text)
        st.code_lines += code
        st.comment_lines += comments
        st.todo_comments += todo
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            st.parse_failures.append(f"{path}: {exc}")
            continue

        st.documentable += 1
        if ast.get_docstring(tree):
            st.documented += 1
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith(
                "_"
            ):
                st.public_symbols += 1

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                st.classes += 1
                st.documentable += 1
                if ast.get_docstring(node):
                    st.documented += 1
                st.methods_per_class.append(
                    sum(1 for b in node.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)))
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                st.functions += 1
                st.documentable += 1
                if ast.get_docstring(node):
                    st.documented += 1
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                st.function_lengths.append(end - node.lineno + 1)
                st.complexities.append(_complexity(node))
                args = node.args
                all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
                for a in all_args:
                    if a.arg in ("self", "cls"):
                        continue
                    st.params_total += 1
                    if a.annotation is not None:
                        st.params_typed += 1
                st.returns_total += 1
                if node.returns is not None:
                    st.returns_typed += 1
                for default in list(args.defaults) + [d for d in args.kw_defaults if d is not None]:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        st.mutable_default_args += 1
            elif isinstance(node, ast.ExceptHandler):
                bare = node.type is None
                broad = isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")
                if (bare or broad) and not _reraises(node):
                    st.broad_excepts += 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                st.print_calls += 1

    if tests_root and tests_root.exists():
        for path in _iter_sources(tests_root):
            st.test_files += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                    st.test_functions += 1
    return st


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--legacy-tests", type=Path, default=None)
    p.add_argument("--legacy-label", default="FACETpy 0.1.0 (Branch bachelor)")
    p.add_argument("--current-root", type=Path, required=True)
    p.add_argument("--current-tests", type=Path, default=None)
    p.add_argument("--current-label", default="FACETpy 2.0.0 (Branch feature/add-deeplearning)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    legacy = analyse_tree(args.legacy_root, args.legacy_tests)
    current = analyse_tree(args.current_root, args.current_tests)
    li, ci = legacy.indicators(), current.indicators()

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": SCOPE,
        "indicator_definitions": {
            k: {"label": v[0], "better": v[1], "evidence_for": v[2]} for k, v in INDICATORS.items()
        },
        "arms": {
            "legacy": {
                "label": args.legacy_label,
                "root": str(args.legacy_root),
                "tests_root": str(args.legacy_tests) if args.legacy_tests else None,
                "indicators": li,
                "parse_failures": legacy.parse_failures,
            },
            "current": {
                "label": args.current_label,
                "root": str(args.current_root),
                "tests_root": str(args.current_tests) if args.current_tests else None,
                "indicators": ci,
                "parse_failures": current.parse_failures,
            },
        },
    }
    (args.out / "engineering_indicators.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    width = max(len(v[0]) for v in INDICATORS.values())
    print(f"{'Indikator':{width}s} {'legacy':>12} {'aktuell':>12}  besser")
    print("-" * (width + 40))
    for key, (label, better, _) in INDICATORS.items():
        print(f"{label:{width}s} {li[key]:>12} {ci[key]:>12}  {better}")
    if legacy.parse_failures or current.parse_failures:
        print("\nParse-Fehler:", *legacy.parse_failures, *current.parse_failures, sep="\n  ")
    print(f"\nwrote {args.out / 'engineering_indicators.json'}")


if __name__ == "__main__":
    main()
