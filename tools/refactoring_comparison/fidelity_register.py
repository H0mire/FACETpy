"""Build a requirement-level fidelity register from the paper-accuracy reviews.

Thirteen model packages carry a ``paper_accuracy_review.md`` whose first table
already states, per paper requirement, what the paper specifies, what the
original implementation did, what this edition does, and how severe the gap was.
That is prose evidence: readable, but not checkable, and a thesis section that
claims "the implementation is faithful in N of M requirements" needs a register
that can be recounted.

This tool turns those tables into one register and — the part that matters —
**verifies each row against the code**:

* Every backticked identifier in the "this edition" cell is treated as a claimed
  code symbol and looked up in the model package. A requirement whose claimed
  symbols do not exist is reported as unverifiable, not silently accepted.
* Test files that mention the model package are linked, with the number of test
  functions they contain.

The register therefore has three independent columns per requirement — the
claim, the symbol that implements it, and the test that exercises it — and a
status derived from them rather than asserted.

What this tool does **not** do: it does not read the PDFs. The "paper specifies"
column is taken from the review verbatim, and re-checking that column against
the original publication is a human step that stays a documented limitation.

Usage::

    .venv/bin/python tools/refactoring_comparison/fidelity_register.py \\
        --out output/refactoring_comparison
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "src" / "facet" / "models"
TESTS = REPO / "tests"

#: Column-name keywords -> canonical field. Order matters: first match wins.
COLUMN_MAP = [
    ("aspect", "aspect"), ("paperstelle", "aspect"),
    ("paper specifies", "paper_requirement"), ("paper / restormer specifies", "paper_requirement"),
    ("vorgabe", "paper_requirement"),
    ("original", "original_state"),
    ("severity", "severity"),
    ("this edition", "edition_status"), ("status in this edition", "edition_status"),
    ("fix in this edition", "edition_status"), ("action in this edition", "edition_status"),
    ("resolution in this edition", "edition_status"), ("umsetzung", "edition_status"),
    ("severity addressed", "severity"), ("severity (orig.)", "severity"),
]

#: How the "edition" cell classifies the requirement's disposition.
DISPOSITION = [
    (re.compile(r"\bn/?a\b|not applicable|nicht anwendbar|\bomitted\b|\bexcluded\b|"
                r"\bdropped\b|\bnot used\b|\bnicht übernommen\b", re.I), "bewusst ausgelassen"),
    (re.compile(r"documented deviation|dokumentierte abweichung|\bdocumented\b|"
                r"\bdeviation\b|\babweichung\b", re.I), "dokumentierte Abweichung"),
    (re.compile(r"\bfixed\b|\bbehoben\b|\bimplemented\b|\bumgesetzt\b|\berfüllt\b|"
                r"\badded\b|\bergänzt\b|\bcorrected\b|\brewritten\b|\breplaced\b|"
                r"\bnow\b|\bexposed\b|\bconfigurable\b|\bdefault\b|\bsupports?\b", re.I),
     "umgesetzt"),
    (re.compile(r"\bkept\b|\bfaithful\b|unchanged|\bbeibehalten\b|\bunverändert\b|"
                r"\bcorrect\b|\bkonform\b", re.I), "bereits konform"),
]

SYMBOL = re.compile(r"`([^`]+)`")
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Rows whose claimed symbols use the paper's notation while the code uses its
#: own names. Each entry was checked by hand against the source at the given
#: location; the register reports them as manually verified rather than as
#: missing, and records what was checked so the check can be repeated.
#:
#: This is deliberately a small explicit table and not a fuzzy matcher. A fuzzy
#: name match would also silence a *real* missing implementation, which is the
#: one thing this register exists to catch.
MANUAL_VERIFICATIONS = {
    ("dhct_gan_v2_paper_accurate_edition", "4"): {
        "actual_symbols": "disc_clean; disc_noise; disc_fused",
        "location": "src/facet/models/dhct_gan_v2_paper_accurate_edition/training.py:558-566",
        "note": "Review nennt die Paper-Notation D_clean/D_noise/D_fused; der Code "
                "instanziiert drei DHCTGanV2PaperAccurateDiscriminator als disc_*.",
    },
    ("dpae_paper_accurate_edition", "2"): {
        "actual_symbols": "fusion_encoder; fusion_decoder",
        "location": "src/facet/models/dpae_paper_accurate_edition/training.py:186-193",
        "note": "Review schreibt fusion_encode/fusion_decode; der Code nennt die "
                "Sequentials fusion_encoder/fusion_decoder, Residuum in forward().",
    },
}

#: Backticked words that are Python syntax or plain prose, not claimed symbols.
NOT_A_SYMBOL = {"True", "False", "None", "and", "or", "not", "for", "in", "if",
                "else", "def", "class", "self", "int", "str", "float", "bool"}


def split_row(line: str) -> list[str]:
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts


def first_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        if i + 1 >= len(lines) or not re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            continue
        header = split_row(line)
        rows = []
        for body in lines[i + 2:]:
            if not body.lstrip().startswith("|"):
                break
            cells = split_row(body)
            if len(cells) == len(header):
                rows.append(cells)
        return header, rows
    return None


def map_columns(header: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for idx, name in enumerate(header):
        low = name.strip().lower()
        if low in ("#", "nr", "id"):
            out[idx] = "number"
            continue
        for needle, field in COLUMN_MAP:
            if needle in low:
                out[idx] = field
                break
    return out


def disposition_of(cell: str) -> str:
    for pattern, label in DISPOSITION:
        if pattern.search(cell):
            return label
    return "unklassifiziert"


def package_symbols(pkg: Path) -> set[str]:
    """Every name a package defines or assigns, plus its keyword arguments.

    Collected from the AST rather than by text search, so a symbol mentioned only
    in a comment does not count as implemented. String constants are included as
    well: a review that claims a loss can be selected as ``mse`` is pointing at a
    string literal, not at a Python identifier, and treating that as "symbol not
    found" would report a false defect.
    """
    names: set[str] = set()
    for path in sorted(pkg.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = node.args
                    for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                        names.add(arg.arg)
                    if a.vararg:
                        names.add(a.vararg.arg)
                    if a.kwarg:
                        names.add(a.kwarg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for piece in re.split(r"[\s/,;=()\[\]{}]+", node.value):
                    if IDENT.match(piece):
                        names.add(piece)
    return names


def claimed_symbols(cell: str) -> list[str]:
    """Backticked identifiers in a cell, split on the separators the reviews use."""
    out: list[str] = []
    for chunk in SYMBOL.findall(cell):
        for piece in re.split(r"[\s/,;=()\[\]{}]+", chunk):
            piece = piece.strip().lstrip(".")
            if IDENT.match(piece) and len(piece) > 2 and piece not in NOT_A_SYMBOL:
                out.append(piece)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def tests_for(pkg_name: str) -> tuple[list[str], int]:
    """Test files that import or name the package, and their test-function count."""
    files, count = [], 0
    for path in sorted(TESTS.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if pkg_name not in text:
            continue
        files.append(str(path.relative_to(REPO)))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        count += sum(1 for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith("test"))
    return files, count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    register: list[dict] = []
    per_model: list[dict] = []
    for review in sorted(MODELS.glob("*/documentation/paper_accuracy_review.md")):
        pkg = review.parent.parent
        pkg_name = pkg.name
        table = first_table(review.read_text(encoding="utf-8"))
        if table is None:
            per_model.append({"model_package": pkg_name, "n_requirements": 0,
                              "parse_status": "keine Tabelle gefunden"})
            continue
        header, rows = table
        cols = map_columns(header)
        symbols = package_symbols(pkg)
        test_files, n_tests = tests_for(pkg_name)

        counted = 0
        for i, cells in enumerate(rows, start=1):
            rec = {"model_package": pkg_name, "review_path": str(review.relative_to(REPO)),
                   "requirement_number": str(i), "aspect": "", "paper_requirement": "",
                   "original_state": "", "edition_status": "", "severity": ""}
            for idx, field in cols.items():
                value = cells[idx]
                if field == "number":
                    rec["requirement_number"] = value or str(i)
                else:
                    rec[field] = value
            if not rec["aspect"] and not rec["paper_requirement"]:
                continue
            counted += 1
            claimed = claimed_symbols(rec["edition_status"])
            found = [s for s in claimed if s in symbols]
            missing = [s for s in claimed if s not in symbols]
            manual = MANUAL_VERIFICATIONS.get((pkg_name, str(rec["requirement_number"]).strip()))
            disposition = disposition_of(rec["edition_status"])
            rec.update({
                "disposition": disposition,
                "claimed_code_symbols": "; ".join(claimed) or "—",
                "symbols_found_in_package": "; ".join(found) or "—",
                "symbols_not_found": "; ".join(missing) or "—",
                "code_evidence": ("verifiziert" if claimed and not missing else
                                  "manuell verifiziert (Namensabweichung)" if manual else
                                  "teilweise" if found else
                                  "keine Symbolangabe" if not claimed else "nicht auffindbar"),
                "manual_verification": (f"{manual['actual_symbols']} @ {manual['location']} — "
                                        f"{manual['note']}") if manual else "",
                # A row the wording does not classify is still classified if the
                # code says so: symbols that exist mean the requirement is in the
                # package, whatever the review's phrasing. Rows with neither a
                # keyword nor a symbol stay explicitly unclassified rather than
                # being counted as implemented.
                "disposition_source": "Formulierung" if disposition != "unklassifiziert"
                                      else ("Symbolbeleg" if claimed and not missing else "—"),
                "test_files": "; ".join(test_files) or "—",
                "n_test_functions_for_package": n_tests,
                "test_evidence": "vorhanden" if n_tests else "keine",
            })
            if rec["disposition"] == "unklassifiziert" and rec["code_evidence"].startswith(
                    ("verifiziert", "manuell verifiziert")):
                rec["disposition"] = "umgesetzt"
            register.append(rec)

        per_model.append({
            "model_package": pkg_name,
            "n_requirements": counted,
            "n_symbol_verified": sum(1 for r in register
                                     if r["model_package"] == pkg_name
                                     and r["code_evidence"].startswith(("verifiziert", "manuell verifiziert"))),
            "n_without_symbol_claim": sum(1 for r in register
                                          if r["model_package"] == pkg_name
                                          and r["code_evidence"] == "keine Symbolangabe"),
            "n_symbols_not_found": sum(1 for r in register
                                       if r["model_package"] == pkg_name
                                       and r["code_evidence"] == "nicht auffindbar"),
            "n_manually_verified": sum(1 for r in register
                                       if r["model_package"] == pkg_name
                                       and r["code_evidence"].startswith("manuell")),
            "n_test_functions": n_tests,
            "parse_status": "ok",
        })

    def _write(name: str, data: list[dict]) -> None:
        if not data:
            return
        fields = sorted({k for row in data for k in row})
        order = ["model_package", "requirement_number", "aspect", "paper_requirement",
                 "original_state", "edition_status", "severity", "disposition",
                 "claimed_code_symbols", "symbols_found_in_package", "symbols_not_found",
                 "disposition_source", "code_evidence", "manual_verification",
                 "test_files", "n_test_functions_for_package",
                 "test_evidence", "review_path"]
        fields = [f for f in order if f in fields] + [f for f in fields if f not in order]
        with (args.out / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in data:
                w.writerow({f: row.get(f, "") for f in fields})

    _write("fidelity_register.csv", register)
    _write("fidelity_per_model.csv", per_model)

    from collections import Counter
    disp = Counter(r["disposition"] for r in register)
    evid = Counter(r["code_evidence"] for r in register)
    sev = Counter((r["severity"] or "—").lower() for r in register)
    (args.out / "fidelity_register.json").write_text(json.dumps({
        "n_model_packages": len(per_model),
        "n_requirements": len(register),
        "disposition_counts": dict(disp),
        "code_evidence_counts": dict(evid),
        "severity_counts": dict(sev),
        "limitation": "Die Spalte 'paper_requirement' ist aus dem Review übernommen; "
                      "eine Rückprüfung gegen die Original-PDF ist ein menschlicher "
                      "Schritt und bleibt offen.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(per_model)} Modellpakete, {len(register)} Anforderungen")
    print("Disposition:", dict(disp))
    print("Codebeleg:  ", dict(evid))
    for r in per_model:
        print(f"  {r['model_package']:42s} {r.get('n_requirements', 0):>3} Anforderungen, "
              f"{r.get('n_symbol_verified', 0):>3} symbolverifiziert, "
              f"{r.get('n_test_functions', 0):>3} Testfunktionen  {r['parse_status']}")


if __name__ == "__main__":
    main()
