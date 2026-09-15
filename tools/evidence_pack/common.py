"""Shared primitives for assembling the results evidence pack.

Why a tool and not hand-copied numbers. The pack's own rules require every
figure to carry a reproducible locator, a source hash and the repository state
it was taken from (``docs/research/results_evidence_pack_execution_plan.md``,
principles 1, 6 and 8). A number typed into a Markdown table satisfies none of
those, and it silently rots the moment a run is repeated. Everything here is
therefore derived from primary JSON/CSV at build time, so re-running the builder
after the long-form runs land replaces the fast numbers without any manual edit.

What this module deliberately does not do: interpret. It formats, hashes and
registers. The wording lives in each section builder's ``results_note.md``, and
statistical decisions live in the evaluation tools that produced the inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]

#: Status vocabulary of the execution plan. Kept as a constant so a typo in a
#: section builder fails loudly instead of inventing a sixth status.
#: Artefakt-Stem -> Verwendungshinweis. Filled by the section builders' module so
#: every entry is authored in one place and its coverage can be checked, rather
#: than scattered across eighty call sites where a gap is invisible.
USAGE_REGISTRY: dict[str, dict[str, str]] = {}

STATUS = {
    "vorhanden",
    "nur aufzubereiten",
    "zu verifizieren",
    "Evaluation oder Run erforderlich",
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Full-file SHA-256. Large NPZ/checkpoints are registered, never copied."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_short(path: Path) -> str:
    return sha256(path)[:16]


def git_state() -> dict[str, Any]:
    """Commit, branch and dirty-state — required for every registered claim.

    A clean-looking table built from a dirty tree is not reproducible, so the
    dirty flag and the list of modified paths are recorded rather than hidden.
    """

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    porcelain = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "commit_short": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(porcelain),
        "dirty_paths": sorted(line[3:] for line in porcelain.splitlines())[:200],
    }


def rel(path: Path | str) -> str:
    """Repo-relative path; absolute only if the file lives outside the repo."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


@dataclass
class Source:
    """One registered primary source.

    ``locator`` is the part that makes a claim checkable: a JSON key path, a CSV
    primary key, a JSONL epoch, a test name or a code symbol. A bare file path
    is not a locator, because it does not say which number was read.
    """

    source_id: str
    path: str
    kind: str
    sha256: str
    size_bytes: int
    locator: str
    note: str = ""

    @classmethod
    def of(cls, source_id: str, path: Path | str, kind: str, locator: str, note: str = "") -> "Source":
        p = Path(path)
        if not p.is_absolute():
            p = REPO / p
        return cls(
            source_id=source_id,
            path=rel(p),
            kind=kind,
            sha256=sha256(p),
            size_bytes=p.stat().st_size,
            locator=locator,
            note=note,
        )


@dataclass
class Claim:
    """One registered claim, in the schema the plan's ``claim_evidence.csv`` wants."""

    claim_id: str
    evidence_question: str
    statement: str
    status: str
    source_ids: str
    locator: str
    run_id: str = ""
    dataset_split_id: str = ""
    checkpoint_id: str = ""
    metric_version: str = ""
    extraction_rule: str = ""
    target_artifact: str = ""
    limitation: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUS:
            raise ValueError(f"unknown status {self.status!r} for {self.claim_id}")


@dataclass
class Section:
    """Accumulates one subsection's claims, sources and written artefacts."""

    section_id: str
    title: str
    directory: Path
    claims: list[Claim] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    acceptance: list[tuple[bool, str]] = field(default_factory=list)
    open_limitations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    usage_entries: list[dict[str, str]] = field(default_factory=list)

    def source(self, source_id: str, path: Path | str, kind: str, locator: str, note: str = "") -> Source:
        src = Source.of(source_id, path, kind, locator, note)
        self.sources.append(src)
        return src

    def claim(self, **kwargs: Any) -> Claim:
        c = Claim(**kwargs)
        self.claims.append(c)
        return c

    def check(self, passed: bool, text: str) -> None:
        self.acceptance.append((bool(passed), text))

    # ----------------------------------------------------------------- writing

    def path(self, name: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.written.append(name)
        return self.directory / name

    def write_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.write_text(text.rstrip() + "\n", encoding="utf-8")
        return p

    def write_table(self, stem: str, rows: Sequence[dict[str, Any]], caption: str) -> None:
        """Write one table as both CSV (machine) and Markdown (paste-ready).

        Both come from the same rows, so the Markdown can never drift from the
        CSV — the failure mode of a hand-maintained thesis table.
        """
        if not rows:
            raise ValueError(f"{stem}: refusing to write an empty table")
        fields = list(rows[0])
        with self.path(f"{stem}.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        lines = [f"<!-- {caption} -->", "", "| " + " | ".join(fields) + " |",
                 "|" + "|".join("---" for _ in fields) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(_md_cell(row[f]) for f in fields) + " |")
        self.write_text(f"{stem}.md", "\n".join(lines))

    def usage(self, stem: str, *, shows: str, says: str, use_for: str,
              not_for: str, claims: str = "") -> None:
        """Register how one artefact is meant to be used in the thesis text.

        Four separate questions, deliberately not merged:

        ``shows``
            What is physically on the page — rows, axes, units, sample size.
        ``says``
            The statement it supports, and how strongly.
        ``use_for``
            Where it belongs in the argument.
        ``not_for``
            The misreading it invites. This field is the reason the guide exists:
            a caption tells a reader what a figure contains, but not what it
            cannot carry, and that is where a results chapter goes wrong.
        """
        self.usage_entries.append({
            "artifact": stem,
            "section_id": self.section_id,
            "shows": shows,
            "says": says,
            "use_for": use_for,
            "not_for": not_for,
            "claims": claims,
        })

    def write_caption(self, figure_stem: str, caption: str, source_ids: Iterable[str]) -> None:
        ids = ", ".join(source_ids)
        self.write_text(
            f"{figure_stem}_caption.md",
            f"{caption}\n\nQuellen: {ids}. Erzeugt von `tools/evidence_pack/build_pack.py`.",
        )

    def _write_usage_guide(self) -> None:
        """One usage guide per subsection, plus an explicit coverage check.

        Artefacts without an entry are listed by name rather than omitted: a
        guide that silently skips what it does not cover is worse than no guide,
        because the reader cannot tell the difference between "nothing to say"
        and "nobody wrote it down".
        """
        written = sorted({n for n in self.written
                          if n.endswith((".csv", ".png")) and not n.startswith("claim_")})
        stems = {}
        for name in written:
            stems.setdefault(name.rsplit(".", 1)[0], []).append(name)
        for stem in stems:
            if stem in USAGE_REGISTRY and stem not in {e["artifact"] for e in self.usage_entries}:
                self.usage(stem, **USAGE_REGISTRY[stem])
        documented = {e["artifact"] for e in self.usage_entries}
        missing = [s for s in stems if s not in documented]

        lines = [f"# Verwendungshinweise {self.section_id} — {self.title}", "",
                 "Je Artefakt vier getrennte Angaben: was darauf zu sehen ist, was es "
                 "aussagt, wofür es im Text taugt und wofür ausdrücklich nicht.", ""]
        for entry in self.usage_entries:
            files = ", ".join(f"`{f}`" for f in stems.get(entry["artifact"], []))
            lines += [f"## {entry['artifact']}", ""]
            if files:
                lines.append(f"*Dateien:* {files}")
            if entry["claims"]:
                lines.append(f"*Claims:* {entry['claims']}")
            lines += ["",
                      f"**Was es zeigt.** {entry['shows']}", "",
                      f"**Was es aussagt.** {entry['says']}", "",
                      f"**Wofür verwenden.** {entry['use_for']}", "",
                      f"**Wofür nicht.** {entry['not_for']}", ""]
        if missing:
            lines += ["## Ohne Verwendungshinweis", "",
                      "Diese Artefakte tragen noch keinen Hinweis:", ""]
            lines += [f"- `{m}`" for m in sorted(missing)]
            lines.append("")
        lines += ["---", "",
                  f"Abdeckung: {len(documented & set(stems))} von {len(stems)} Artefakten "
                  f"dieses Unterabschnitts."]
        self.write_text("usage_guide.md", "\n".join(lines))

    def finalise(self, git: dict[str, Any], generator: str) -> None:
        """Write claim register, provenance and acceptance for this subsection."""
        self._write_usage_guide()
        if self.claims:
            with self.path("claim_evidence.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(asdict(self.claims[0])))
                writer.writeheader()
                writer.writerows(asdict(c) for c in self.claims)
        provenance = {
            "section_id": self.section_id,
            "title": self.title,
            "generator": generator,
            "git": git,
            "sources": [asdict(s) for s in self.sources],
            "artifacts": sorted(set(self.written)),
        }
        self.path("provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        passed = sum(1 for ok, _ in self.acceptance if ok)
        lines = [
            f"# Abschnittsabnahme {self.section_id} — {self.title}",
            "",
            f"Kriterien erfüllt: {passed} / {len(self.acceptance)}",
            "",
        ]
        for ok, text in self.acceptance:
            lines.append(f"- [{'x' if ok else ' '}] {text}")
        if self.open_limitations:
            lines += ["", "## Offene Einschränkungen", ""]
            lines += [f"- {t}" for t in self.open_limitations]
        if self.notes:
            lines += ["", "## Hinweise", ""]
            lines += [f"- {t}" for t in self.notes]
        lines += [
            "",
            "## Freigabe",
            "",
            "Freigabe zur Übernahme in die Thesis: **offen** — erfordert Nutzerbestätigung.",
        ]
        self.write_text("section_acceptance.md", "\n".join(lines))


def _md_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:            # NaN
            return "n/a"
        if value in (float("inf"), float("-inf")):
            return "∞" if value > 0 else "−∞"
        return f"{value:.4g}"
    return str(value).replace("|", "\\|")


def fmt(value: float | None, digits: int = 3, unit: str = "") -> str:
    """Number formatting used inside prose sentences, so rounding is one rule."""
    if value is None or value != value:
        return "n/a"
    return f"{value:.{digits}f}{unit}"


def load_json(path: Path | str) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return json.loads(p.read_text(encoding="utf-8"))


def load_csv(path: Path | str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
