Documentation policy
====================

Keep maintained project explanations in English reStructuredText under Sphinx.
Use familiar words, active verbs and short sentences. Explain necessary technical
terms. Remove repetition and stale claims, while preserving relevant qualifications.

Allow one short introductory README per model package and strategic repository,
guide and experimental-collection entry points. Keep implementation instructions
needed by repository workflows. Additional Markdown documentation requires an
explicit, justified exception.

The thesis-reference pages own scientific interpretation. The thesis guide owns
acquisition and execution instructions. Experiment folders own configuration and
compact evidence. The catalog owns associations; generated indexes provide views
of that catalog. A paragraph or result table has one maintained source.

Preserve titles, paths and anchors cited by the thesis. Content inside a protected
page can improve; the page must not be merged away or replaced with a redirect.
Original run records remain unchanged, including their machine-generated fields.

Earlier phases remain relevant whenever the thesis uses their results. Age alone
is never a reason to exclude evidence. Keep necessary negative findings and
variant differences, and remove material only after checking its scientific and
runtime dependencies.

Diagram sources and skill
-------------------------

Each model package owns two generated SVGs and their editable JSON architecture
profile in ``diagrams/``. The overview fits within half an A4 page; the detailed
view can grow vertically. The model diagram gallery is generated from the same
profiles. The canonical FACETpy diagram skill lives in
``.agents/skills/facetpy-diagram``; the Claude skill path links to it. Skill Markdown
is an implementation-instruction exception, not a second documentation tree.

Regenerate and check diagrams with::

    uv run python tools/diagrams/build_model_diagrams.py
    uv run python tools/diagrams/build_model_diagrams.py --check

PNG previews are optional local review files. Rendering them requires librsvg's
``rsvg-convert``; generating and checking SVGs requires only Python.
