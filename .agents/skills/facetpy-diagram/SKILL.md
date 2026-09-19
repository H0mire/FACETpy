---
name: facetpy-diagram
description: Create FACETpy-branded SVG architecture, model, pipeline, class and sequence diagrams from the repository implementation. Use for FACETpy documentation and model visuals.
---

# FACETpy diagrams

Use the English FACETpy visual language defined in [the design system](references/design-system.md)
and compose SVGs with `assets/facetpy_svg.py`. The toolkit is the shared source
for colours, typography, cards, connectors, the connectome dot and EEG motif.

## Establish the content

Read the relevant implementation before drawing. Record the exact model variant,
input/output contract, main operations, parallel branches, skip connections and
training-only paths. Label optional behavior and distinguish model output from
pipeline output. Derive dimensions from an explicit configuration or use symbolic
sizes; do not present one run's parameters as universal defaults. A package named
`paper_accurate` is an implementation variant, not proof of exact paper fidelity.

## Model diagram pairs

Each model package owns `diagrams/architecture.json`, `diagrams/overview.svg` and
`diagrams/architecture.svg`. The JSON is the editable content source and links
the implementation files. Use the existing renderer:

```sh
uv run python tools/diagrams/build_model_diagrams.py
uv run python tools/diagrams/build_model_diagrams.py --check
```

The overview shows at most five conceptual stages. Group parallel operations in
one stage instead of drawing a false sequence. Keep its physical size at most
190 mm by 128 mm, within half an A4 page. The detailed diagram shows explicit
relationships and variant-specific behavior; it can grow vertically.

Read the renderer's profile schema and an existing architecture.json when adding
a model. Keep the profile next to its model, link both SVGs from the model README,
and embed each pair in the owning Sphinx model page. The renderer uses the
README's Model reference link to find that page and replaces only the region
between its `model-diagrams-start` and `model-diagrams-end` comments. Do not add
a separate diagram gallery or navigation category.

## Rendering and review

Use a dedicated drawing agent for substantial layout work when delegation is
available: give it the concrete elements, relationships, source paths and output
location. Use a capable available model; this skill is not tied to one provider.
The drawing agent reads the design system and toolkit. For small edits or when
no agent is available, perform the same checks locally.

Keep the SVG viewBox width at 1000 units, the background transparent, and colours
from the toolkit. Route arrows outside unrelated cards; keep labels and markers
visible. Generate PNG previews outside the repository using `--preview-dir`
and `rsvg-convert` (librsvg). Inspect the rendered diagrams, including a dedicated
occlusion pass, and correct overlaps, clipping, small text or ambiguous arrows.
Inspect both a compact overview and detailed examples with branches and skips.

Do not run training or load checkpoints to draw the architecture. Diagram work
must preserve model behavior. Use the repository's lint, test and Sphinx checks.
