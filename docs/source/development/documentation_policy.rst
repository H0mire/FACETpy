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
view can grow vertically. The diagram sections on the model pages are generated from the same
profiles. The canonical FACETpy diagram skill lives in
``.agents/skills/facetpy-diagram``; the Claude skill path links to it. Skill Markdown
is an implementation-instruction exception, not a second documentation tree.

Regenerate and check diagrams with::

    uv run python tools/diagrams/build_model_diagrams.py
    uv run python tools/diagrams/build_model_diagrams.py --check

PNG previews are optional local review files. Rendering them requires librsvg's
``rsvg-convert``; generating and checking SVGs requires only Python.

Per-model result views
----------------------

Each model package has a ``results/`` directory. Its ``index.rst`` and
``manifest.json`` list experiments assigned to that exact variant in the thesis
catalog. Each experiment has ``hyperparameters.yaml`` and ``evaluation.json``.
A ``training_curve.svg`` is added only when an epoch history exists.

The original records remain under ``masterthesis_guide``. Generated views retain
source paths, hashes, dataset and protocol. Shared comparison tables stay shared
unless the catalog specifies a row or the table names the experiment explicitly.
A missing association must not be filled from a similarly named model. Original
record keys and values remain unchanged, including historical language.

Regenerate and check the result views with::

    uv run python -m masterthesis_guide.reproduce model-results
    uv run python -m masterthesis_guide.reproduce model-results --check

The model pages include these indexes below each variant's diagrams. The curve
shows recorded training and validation loss without smoothing. Its objective and
scale belong to that run; losses from different objectives are not comparable.
Weights remain in the Git LFS artifact tree and datasets remain external.
