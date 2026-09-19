"""Render checked, explicit model architecture profiles with the FACETpy toolkit.

Profiles live at ``<model>/diagrams/architecture.json``. No model or weights are
imported. Run without arguments to rebuild every profile; ``--check`` compares
byte-for-byte without modifying files. PNG previews are opt-in and external.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from html import escape
from pathlib import Path

from facetpy_svg import (
    C,
    Diagram,
    card_shell,
    edge,
    node_dot,
    rounded_top,
    text,
    text_width,
)

ROOT = Path(__file__).resolve().parents[2]
KINDS = {"flow", "skip", "conditioning", "training", "merge"}


def wrap(value: str, width: float, size: float = 18) -> list[str]:
    """Wrap using the toolkit's conservative glyph-width estimate."""
    lines = []
    for paragraph in value.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and text_width(candidate, size) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
            while text_width(line, size) > width:
                cut = len(line) - 1
                while cut > 1 and text_width(line[:cut], size) > width:
                    cut -= 1
                lines.append(line[:cut])
                line = line[cut:]
        lines.append(line)
    return lines


def load_profile(path: Path, root: Path = ROOT) -> dict:
    """Validate the content contract, graph references and repository sources."""

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    profile = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if profile.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version 1")
    for name in ("model_dir", "title"):
        if not isinstance(profile.get(name), str) or not profile[name].strip():
            raise ValueError(f"{path}: missing {name}")
    model_dir = (root / profile["model_dir"]).resolve()
    if not model_dir.is_relative_to(root.resolve()) or not model_dir.is_dir():
        raise ValueError(f"{path}: model_dir is not a repository directory")
    if path.resolve().parent != model_dir / "diagrams":
        raise ValueError(f"{path}: profile must be in its model's diagrams directory")
    steps = profile.get("overview", [])
    if not 1 <= len(steps) <= 5:
        raise ValueError(f"{path}: overview needs 1 to 5 steps")
    for step in steps:
        if not step.get("title") or not isinstance(step.get("description"), str):
            raise ValueError(f"{path}: overview step needs title and description")
    nodes = profile.get("nodes", [])
    ids = {node["id"] for node in nodes}
    if not nodes or len(ids) != len(nodes):
        raise ValueError(f"{path}: node IDs must be unique and nonempty")
    positions = defaultdict(list)
    for node in nodes:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", node["id"]):
            raise ValueError(f"{path}: invalid node ID {node['id']}")
        row = node.get("row")
        if not isinstance(row, int) or row < 0:
            raise ValueError(f"{path}: rows must be nonnegative integers")
        column = node.get("column")
        if column not in (None, 0, 1):
            raise ValueError(f"{path}: columns must be 0, 1, or omitted")
        positions[row].append(column)
        if not node.get("title") or not isinstance(node.get("description"), list):
            raise ValueError(f"{path}: nodes require title and description list")
        if not all(isinstance(line, str) for line in node["description"]):
            raise ValueError(f"{path}: descriptions must contain strings")
    for columns in positions.values():
        if len(columns) != len(set(columns)) or (None in columns and len(columns) > 1):
            raise ValueError(f"{path}: overlapping nodes in a row")
    for relation in profile.get("edges", []):
        if relation.get("source") not in ids or relation.get("target") not in ids:
            raise ValueError(f"{path}: edge references an unknown node")
        if relation.get("kind") not in KINDS or not relation.get("label"):
            raise ValueError(f"{path}: edge requires a known kind and a label")
    if not profile.get("sources"):
        raise ValueError(f"{path}: at least one implementation source is required")
    for source in profile["sources"]:
        target = (root / source["path"]).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError(f"{path}: missing or unsafe source {source['path']}")
    return profile


def box(x, y, width, title, descriptions, *, compact=False):
    """Readable card variant built entirely from the shared branded primitives."""
    title_size = 23 if compact else 21
    titles = wrap(title, width - 62, title_size)
    body_size = 20 if compact else 18
    lines = [line for value in descriptions for line in wrap(value, width - 36, body_size)]
    header_height = 22 + len(titles) * (title_size + 4)
    height = header_height + 15 + len(lines) * (body_size + 6) + 12
    fragments = [
        card_shell(x, y, width, height),
        f'<path d="{rounded_top(x, y, width, header_height, 12)}" fill="url(#fp-header)"/>',
        node_dot(x + 21, y + 24),
    ]
    for index, line in enumerate(titles):
        fragments.append(
            text(x + 40, y + 29 + index * (title_size + 4), line, size=title_size, weight=600, fill=C["header_fg"])
        )
    for index, line in enumerate(lines):
        fragments.append(
            text(x + 18, y + header_height + 28 + index * (body_size + 6), line, size=body_size, fill=C["ink"])
        )
    return {"svg": "".join(fragments), "x": x, "y": y, "w": width, "h": height, "cx": x + width / 2}


def heading(diagram, profile, label):
    """Keep long model titles within the fixed canvas without shrinking text."""
    display_title = profile["title"] + (f" · {profile['subtitle']}" if profile.get("subtitle") else "")
    titles = wrap(display_title, 920, 27)
    for index, line in enumerate(titles):
        diagram.add(text(40, 38 + index * 32, line, size=27, weight=600))
    y = 38 + len(titles) * 32
    diagram.add(text(40, y, label, size=17, fill=C["slate"]))
    # Reuse the canonical title-block wave, without its fixed-width title text.
    from facetpy_svg import eeg_wave

    diagram.add(eeg_wave(790, y - 5, 160))
    return y + 28


def physical_svg(diagram: Diagram, profile: dict, *, overview=False) -> str:
    """190 mm width; compact overview never exceeds a half-page print area."""
    height_mm = diagram.height * 0.19
    if overview and height_mm > 128:
        raise ValueError(f"{profile['title']}: overview exceeds 190 × 128 mm; shorten content")
    output = diagram.render_svg().replace(
        f'width="1000" height="{diagram.height}" font-family=',
        f'width="190mm" height="{height_mm:.2f}mm" font-family=',
        1,
    )
    return output.replace(
        ">\n",
        ">\n<title>"
        + escape(profile["title"])
        + (" — overview" if overview else " — detailed architecture")
        + "</title>\n",
        1,
    )


def render_overview(profile: dict) -> str:
    diagram = Diagram(1)
    y = heading(diagram, profile, "Model overview")
    # Two columns keep five substantial steps readable at half-page print size.
    count = len(profile["overview"])
    columns = 1 if count <= 3 else 2
    width = 780 if columns == 1 else 420
    boxes = []
    for index, step in enumerate(profile["overview"]):
        row, column = divmod(index, columns)
        x = 110 if columns == 1 else 50 + column * 480
        item = box(x, 0, width, f"{index + 1}. {step['title']}", [step["description"]], compact=True)
        boxes.append((row, column, item, step))
    row_heights = {row: max(item["h"] for r, _, item, _ in boxes if r == row) for row, _, _, _ in boxes}
    starts = {}
    for row in sorted(row_heights):
        starts[row] = y
        y += row_heights[row] + 36
    actual = []
    for row, _column, old, step in boxes:
        item = box(
            old["x"], starts[row], width, f"{len(actual) + 1}. {step['title']}", [step["description"]], compact=True
        )
        diagram.add(item)
        actual.append(item)
    # Overview numbers express the ordered abstraction; adjacent cards connect.
    # Wrap-to-next-row routes run in the clear gap, never through other steps.
    for source, target in zip(actual, actual[1:], strict=False):
        if source["y"] == target["y"]:
            sy = source["y"] + source["h"] / 2
            ty = target["y"] + target["h"] / 2
            points = [(source["x"] + source["w"] + 4, sy), (500, sy), (500, ty), (target["x"] - 4, ty)]
        else:
            middle = target["y"] - 18
            points = [
                (source["cx"], source["y"] + source["h"] + 4),
                (source["cx"], middle),
                (target["cx"], middle),
                (target["cx"], target["y"] - 4),
            ]
        diagram.add_edge(edge(points))
    diagram.height = int(y + 12)
    return physical_svg(diagram, profile, overview=True)


def badge(number, x, y):
    return (
        f'<rect x="{x - 14}" y="{y - 13}" width="28" height="26" rx="7" '
        f'fill="{C["tint"]}" stroke="{C["blue"]}" stroke-width="1"/>'
        + text(x, y + 6, str(number), size=17, anchor="middle", weight=600)
    )


def assert_route_clear(points, boxes):
    """Reject any connector segment crossing a card interior."""
    for (x1, y1), (x2, y2) in zip(points, points[1:], strict=False):
        if x1 != x2 and y1 != y2:
            raise ValueError("Connectors must be orthogonal")
        for node_id, item in boxes.items():
            left, top = item["x"], item["y"]
            right, bottom = left + item["w"], top + item["h"]
            horizontal = y1 == y2 and top < y1 < bottom and max(x1, x2) > left and min(x1, x2) < right
            vertical = x1 == x2 and left < x1 < right and max(y1, y2) > top and min(y1, y2) < bottom
            if horizontal or vertical:
                raise ValueError(f"Connector crosses node {node_id}; revise its row/column layout")


def label_position(preferred, points, boxes, occupied):
    """Place edge-number chips without covering a card or another chip."""
    candidates = [preferred]
    segments = sorted(
        zip(points, points[1:], strict=False),
        key=lambda pair: abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]),
        reverse=True,
    )
    for start, end in segments:
        for fraction in (0.5, 0.25, 0.75, 0.125, 0.875):
            candidates.append((start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction))
    for x, y in candidates:
        if not 18 <= x <= 982:
            continue
        if any(abs(x - ox) < 34 and abs(y - oy) < 32 for ox, oy in occupied):
            continue
        if any(
            x + 18 > item["x"]
            and x - 18 < item["x"] + item["w"]
            and y + 17 > item["y"]
            and y - 17 < item["y"] + item["h"]
            for item in boxes.values()
        ):
            continue
        occupied.append((x, y))
        return x, y
    raise ValueError("No unobstructed position for relationship number; adjust layout")


def render_architecture(profile: dict) -> str:
    diagram = Diagram(1)
    by_id = {node["id"]: node for node in profile["nodes"]}
    maximum_degree = max(
        sum(relation["source"] == node_id or relation["target"] == node_id for relation in profile["edges"])
        for node_id in by_id
    )
    gap = max(120, maximum_degree * 26 + 48)
    rows = sorted({node["row"] for node in profile["nodes"]})
    first_row_incoming = sum(by_id[relation["target"]]["row"] == rows[0] for relation in profile["edges"])
    title_clearance = first_row_incoming * 26 + 24 if first_row_incoming else 0
    y = heading(diagram, profile, "Detailed architecture · numbered relationships below") + title_clearance
    row_index = {row: index for index, row in enumerate(rows)}
    boxes = {}
    row_ends = {}
    row_starts = {}
    for row in rows:
        row_starts[row] = y
        maximum = 0
        for node in profile["nodes"]:
            if node["row"] != row:
                continue
            column = node.get("column")
            width, x = (560, 220) if column is None else (300, 170 + column * 360)
            item = box(x, y, width, node["title"], node["description"])
            boxes[node["id"]] = item
            diagram.add(item)
            maximum = max(maximum, item["h"])
        row_ends[row] = y + maximum
        # Distinct edge tracks fit between cards without crossing their content.
        y += maximum + gap
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for index, relation in enumerate(profile["edges"]):
        outgoing[relation["source"]].append(index)
        incoming[relation["target"]].append(index)
    bypass_index = 0
    occupied_labels = []
    for index, relation in enumerate(profile["edges"]):
        source, target = boxes[relation["source"]], boxes[relation["target"]]
        source_row, target_row = by_id[relation["source"]]["row"], by_id[relation["target"]]["row"]
        out = outgoing[relation["source"]]
        into = incoming[relation["target"]]
        sx = source["cx"] + (out.index(index) - (len(out) - 1) / 2) * 26
        tx = target["cx"] + (into.index(index) - (len(into) - 1) / 2) * 26
        sy, ty = source["y"] + source["h"] + 4, target["y"] - 4
        adjacent = row_index[target_row] == row_index[source_row] + 1
        if adjacent:
            track = row_ends[source_row] + 32 + out.index(index) * 24
            points = [(sx, sy), (sx, track), (tx, track), (tx, ty)]
            label_x, label_y = ((sx + tx) / 2, track) if abs(sx - tx) > 36 else (sx, (sy + track) / 2)
        else:
            side = bypass_index % 2
            lane = (bypass_index // 2) % 6
            outer = 30 + lane * 23 if side == 0 else 970 - lane * 23
            bypass_index += 1
            above = target["y"] - 28 - into.index(index) * 22
            below = row_ends[source_row] + 30 + out.index(index) * 22
            points = [(sx, sy), (sx, below), (outer, below), (outer, above), (tx, above), (tx, ty)]
            label_x, label_y = outer, (above + below) / 2
        assert_route_clear(points, boxes)
        fragment = edge(points)
        if relation["kind"] in {"training", "conditioning"}:
            fragment = fragment.replace('fill="none" stroke=', 'fill="none" stroke-dasharray="7 5" stroke=', 1)
        diagram.add_edge(fragment)
        label_x, label_y = label_position((label_x, label_y), points, boxes, occupied_labels)
        diagram.add(badge(index + 1, label_x, label_y))
    y += 4
    diagram.add(text(60, y, "Relationships", size=23, weight=600))
    y += 32
    for index, relation in enumerate(profile["edges"]):
        source, target = by_id[relation["source"]], by_id[relation["target"]]
        content = f"{source['title']} → {target['title']} · {relation['kind']}"
        default_label = f"{source['title']} to {target['title']}"
        if relation["label"] != default_label:
            content += f": {relation['label']}"
        lines = wrap(content, 840, 18)
        diagram.add(badge(index + 1, 74, y - 5))
        for line in lines:
            diagram.add(text(100, y, line, size=18))
            y += 25
        y += 12
    if profile.get("notes"):
        y += 14
        diagram.add(text(60, y, "Reading notes", size=23, weight=600))
        y += 32
        for note in profile["notes"]:
            for line in wrap(note, 870, 18):
                diagram.add(text(60, y, line, size=18))
                y += 25
            y += 9
    y += 18
    diagram.add(text(60, y, "Implementation sources", size=23, weight=600))
    y += 30
    for source in profile["sources"]:
        label = f"{source.get('label', 'Source')}: {source['path']}"
        for line in wrap(label, 870, 17):
            diagram.add(text(60, y, line, size=17, fill=C["blue"]))
            y += 24
    diagram.height = int(y + 44)
    return physical_svg(diagram, profile)


def model_documentation(profile):
    """Read the summary and owning Sphinx page from a model README."""
    readme_path = ROOT / profile["model_dir"] / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    paragraphs = readme.split("\n\n")
    if len(paragraphs) < 2 or not paragraphs[1].strip():
        raise ValueError(f"{readme_path}: missing model summary")
    reference = re.search(
        r"\[Model reference\]\(https://facetpy\.readthedocs\.io/en/latest/([a-z0-9_/]+)\.html\)",
        readme,
    )
    if reference is None:
        raise ValueError(f"{readme_path}: missing model reference")
    document = ROOT / "docs/source" / (reference.group(1) + ".rst")
    if not document.is_file():
        raise ValueError(f"{readme_path}: missing Sphinx owner {document}")
    return document, " ".join(paragraphs[1].splitlines())


def render_model_sections(entries, document):
    """Render variant diagrams inside their owning model description."""
    lines = [
        ".. model-diagrams-start",
        "",
        ".. Generated by tools/diagrams/build_model_diagrams.py; edit model profiles and READMEs.",
        "",
    ]
    for profile, summary in sorted(
        entries,
        key=lambda entry: (
            "experimental/" in entry[0]["model_dir"],
            len(Path(entry[0]["model_dir"]).parts),
            entry[0]["model_dir"],
        ),
    ):
        variant = str(Path(profile["model_dir"]).relative_to("src/facet/models"))
        anchor = "diagram-" + variant.replace("/", "-").replace("_", "-")
        # A distinct readable heading keeps the per-page table of contents useful.
        if "paper_accurate" in variant:
            heading = profile["title"] + " — experimental paper_accurate"
        elif variant.endswith("/deployment"):
            heading = profile["title"] + " — deployment"
        else:
            heading = profile["title"]
        directory = Path(os.path.relpath(ROOT / profile["model_dir"] / "diagrams", document.parent)).as_posix()
        lines.extend(
            [
                f".. _{anchor}:",
                "",
                heading,
                "~" * len(heading),
                "",
                f"Variant: ``{variant}``.",
                "",
                summary,
                "",
                f".. figure:: {directory}/overview.svg",
                "   :width: 190mm",
                f"   :alt: {heading} — compact model overview",
                "",
                "   Compact overview; native print size is within half an A4 page.",
                "",
                ".. raw:: html",
                "",
                '   <details class="model-architecture">',
                "   <summary>Show detailed architecture</summary>",
                "",
                f".. figure:: {directory}/architecture.svg",
                "   :width: 100%",
                f"   :alt: {heading} — detailed architecture",
                "",
                "   Detailed data paths, branches and implementation sources.",
                "",
                ".. raw:: html",
                "",
                "   </details>",
                "",
                f":download:`Overview (SVG) <{directory}/overview.svg>`",
                "|",
                f":download:`Detailed architecture (SVG) <{directory}/architecture.svg>`",
                "|",
                f":download:`Editable profile (JSON) <{directory}/architecture.json>`",
                "",
            ]
        )
    lines.extend([".. model-diagrams-end", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*", type=Path, help="Specific architecture.json profiles")
    parser.add_argument("--check", action="store_true", help="Validate profiles and fail if SVGs differ")
    parser.add_argument("--preview-dir", type=Path, help="External PNG preview directory (requires rsvg-convert)")
    args = parser.parse_args(argv)
    paths = [
        path.resolve()
        for path in (args.profiles or sorted((ROOT / "src/facet/models").glob("**/diagrams/architecture.json")))
    ]
    if not paths:
        parser.error("No model architecture profiles found")
    if args.preview_dir and args.preview_dir.resolve().is_relative_to(ROOT):
        parser.error("Keep PNG previews outside the repository")
    failures = []
    profiles = []
    for path in paths:
        profile = load_profile(path)
        profiles.append(profile)
        for name, render in (("overview", render_overview), ("architecture", render_architecture)):
            output = path.parent / f"{name}.svg"
            content = render(profile)
            if args.check:
                if not output.is_file() or output.read_text(encoding="utf-8") != content:
                    failures.append(str(output.relative_to(ROOT)))
            else:
                output.write_text(content, encoding="utf-8")
            if args.preview_dir:
                converter = shutil.which("rsvg-convert")
                if converter is None:
                    parser.error("rsvg-convert is required for PNG previews")
                destination = args.preview_dir / Path(profile["model_dir"]).relative_to("src/facet/models")
                destination.mkdir(parents=True, exist_ok=True)
                temporary = destination / f"{name}.svg"
                temporary.write_text(content, encoding="utf-8")
                subprocess.run(
                    [converter, "-b", "white", "-w", "1280", str(temporary), "-o", str(destination / f"{name}.png")],
                    check=True,
                )
    if not args.profiles:
        owners = defaultdict(list)
        for profile in profiles:
            document, summary = model_documentation(profile)
            owners[document].append((profile, summary))
        for document, entries in owners.items():
            original = document.read_text(encoding="utf-8")
            marker = re.compile(r"^\.\. model-diagrams-start\n.*?^\.\. model-diagrams-end\n", re.M | re.S)
            matches = list(marker.finditer(original))
            if len(matches) != 1:
                raise ValueError(f"{document}: expected one model-diagrams marker pair")
            region = matches[0]
            content = original[: region.start()] + render_model_sections(entries, document) + original[region.end() :]
            if args.check:
                if original != content:
                    failures.append(str(document.relative_to(ROOT)))
            else:
                document.write_text(content, encoding="utf-8")
    if failures:
        parser.exit(1, "Outdated model diagrams:\n" + "\n".join(failures) + "\n")
    print(f"{'Verified' if args.check else 'Rendered'} {len(paths)} profiles / {len(paths) * 2} SVGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
