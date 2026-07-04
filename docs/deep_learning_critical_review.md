# Deep-Learning Subsystem — Critical Review (over-abstraction lens)

**Status:** Draft
**Owner:** Janik Müller
**Context:** `feature/add-deeplearning` branch, master's thesis
**Lens:** The same perspective-change applied to the artifact-type discussion —
*where does this branch add structure / abstraction to buy flexibility it does not
actually use, and pay for it in complexity, dead code, or maintenance traps?* Deliberately
critical; the strengths are documented elsewhere
([`deep_learning_architecture_review.md`](deep_learning_architecture_review.md)).
**Out of scope:** training-data / AAS-surrogate ground truth (known TODO).
**Last updated:** 2026-06-09

---

## 0. The recurring pattern

The artifact-type enum debate generalized: an abstraction was proposed to handle a
*hypothetical* range of cases (every artifact kind), when the flexibility either already
existed (a string / tags) or wasn't needed. The same shape recurs across the branch,
and it is the single most important critique:

> **Repeatedly, the branch adds a framework-agnostic / future-proofing layer that gains
> flexibility it never exercises, and the cost — dead code, a reimplementation of mature
> tooling, an anti-recommended class, repo bloat — is real and ongoing.**

"Flexibility you don't use is not flexibility; it's surface area." Each finding below is
an instance, rated by how strongly the evidence supports cutting or simplifying.

## 1. Dual-framework (TF + PyTorch) training abstraction — **cut to PyTorch** (strong)

**Evidence.**
- 15 / 15 model inference adapters declare `runtime=DeepLearningRuntime.PYTORCH`. There
  are **zero** TensorFlow models.
- Yet the branch ships `TensorFlowModelWrapper` (`training/wrapper.py:334`),
  `TensorFlowInferenceAdapter` (`correction/deep_learning.py:458`), `TensorFlowTensorLayout`,
  and the `_to_tf`/`_from_tf` axis-shuffling conversions.
- The entire reason the `Trainer` works in **numpy at the boundary** (converting to
  framework tensors inside each wrapper) is to support both backends. With one backend,
  that conversion layer is pure overhead — every batch is numpy→torch→numpy round-tripped
  for no reason.

**Critique.** This is the artifact-enum mistake at the largest scale. Dual-backend support
is *hypothetical flexibility*: no model uses TF, the thesis will not use TF, and the cost
is a parallel TF code path that is untested-by-use, plus a numpy boundary that taxes the
hot loop and complicates every wrapper.

**Perspective change.** Commit to PyTorch. Delete the TF training wrapper and let the
trainer pass tensors (or keep numpy only at the dataset edge). Keep TF/ONNX on the
*inference* side only if there is a concrete deployment reason (see §5 — that side is more
defensible). The "what if a contributor wants TF" argument is exactly "what if there's a
future artifact type" — answer it with a string-sized escape hatch (the `CUSTOM` runtime
already exists), not a maintained second implementation.

## 2. The numpy loss layer + `TorchLossWrapper` — **delete** (strong)

**Evidence.**
- `losses.py` defines numpy `mse_loss`, `mae_loss`, `spectral_loss`, `snr_loss`,
  `CompositeLoss`, and `TorchLossWrapper` (~267 lines).
- **Zero** model `build_loss` factories import any of them. Every model implements its
  loss directly as a `torch.nn.Module`.
- `TorchLossWrapper`'s own docstring says: *"For most cases the simpler, leak-free approach
  is to implement the loss directly as a differentiable `torch.nn.Module` … so the logged
  and optimised loss are identical."* i.e. the class documents that you should not use it.
- A whole commit (`a708bc6`, "harden TorchLossWrapper against silent MSE substitution")
  was spent fixing a trap that only exists *because* numpy losses can't backprop — a
  problem the class itself creates.
- `losses.py:267` is a `from typing import Any  # noqa` at the bottom of the file — a tell
  that the numpy-loss abstraction was bolted on awkwardly.

**Critique.** This is a self-inflicted problem. Numpy losses exist to be "framework
agnostic," but a numpy loss can't be a gradient, so a bridge (`TorchLossWrapper`) is
needed, and the bridge has a footgun (log one loss, optimise another) that needed
hardening. None of it is used. The flexibility (framework-agnostic losses) bought a class
nobody should call and a bug to fix.

**Perspective change.** Delete the numpy losses and the wrapper. If shared loss building
blocks are wanted, provide them as `torch.nn.Module`s (what the models already do). Loss
*logging* falls out for free because the optimised loss is the logged loss.

## 3. The bespoke `Trainer` + callback system — **reuse, don't reinvent** (medium–strong)

**Evidence.**
- `Callback` (`callbacks.py:22`) reimplements the Keras/Lightning hook API verbatim:
  `on_train_begin` / `on_epoch_begin` / `on_batch_begin` / `on_batch_end` / `on_epoch_end`
  / `on_train_end`, plus a `state.stop_training` flag.
- `CheckpointConfig` (`config.py:16`: `monitor`, `mode`, `save_top_k`, `save_last`,
  `filename_template`) and `EarlyStoppingConfig` (`monitor`, `mode`, `patience`,
  `min_delta`) are PyTorch Lightning's `ModelCheckpoint` / `EarlyStopping` configs,
  field-for-field.
- `trainer.py` (577 lines) + `callbacks.py` (628) + `config.py` (320) + `cli.py` (771)
  ≈ 2,300 lines re-deriving a training loop, checkpointing, early stopping, LR scheduling,
  metric logging, and a config system that Lightning provides and battle-tests.

**Critique.** The *only* architectural justification for a hand-rolled framework-agnostic
trainer is multi-backend support — which §1 shows doesn't exist. Once you are PyTorch-only,
this is ~2,300 lines reimplementing Lightning, carrying the maintenance and the subtle bugs
(e.g. framework seeding, resume — see the requirements doc) that Lightning already solved.

**Caveat (be fair).** The bespoke trainer does give a nice Rich live dashboard and zero
heavy dependency, and a thesis benefits from a loop the author fully understands. So this
is "medium," not "strong-delete." But the cost/benefit only looks good *because* the
multi-backend premise inflated the perceived benefit. If kept, it should be justified on
its own merits ("I want a dependency-free transparent loop"), not on agnosticism.

**Perspective change.** Either (a) adopt Lightning and keep only the FACETpy-specific
glue (dataset → adapter, the Rich dashboard as a callback), or (b) explicitly own the
decision to hand-roll and document *why*, dropping the agnosticism rationale.

## 4. Committed evaluation artifacts — **commit the record, not the run** (medium)

**Evidence.**
- 219 files under `output/` and `training_output/` are committed. `.gitignore` line 95
  (`!output/model_evaluations`) whitelists them *deliberately*, so this is a choice, not
  an accident.
- `evaluation_standard.md` itself says: *"Large generated artifacts should stay in
  `output/` unless a small figure is explicitly useful enough to version"* — yet `plots/`
  subdirectories are committed for many runs.

**Critique.** Git is being used as an artifact store. The intent (a reproducible record
of thesis results) is legitimate, but the mechanism scales badly: every future run adds
more committed binaries, the repo only grows, and the standard's own "small figure"
exception is being stretched to whole plot directories.

**Perspective change.** Commit the *machine-readable record* (`metrics.json` /
`evaluation_manifest.json` — small, diffable, the actual evidence) and at most one curated
hero figure per model. Keep plots and bulk artifacts out of git (release asset, DVC, or an
external store). The flexibility "anyone can browse every run in the repo" isn't worth the
permanent weight.

## 5. Where the abstraction IS earned (counter-balance)

Being critical cuts both ways — these are *not* over-abstraction and should stay:

- **The inference-side multi-runtime contract** (TF / ONNX / Numpy adapters in
  `deep_learning.py`). Unlike training, inference has a real reason to load checkpoints in
  multiple formats, and ONNX is a legitimate deployment target. The flexibility here maps
  to plausible use, even if today's models are torch.
- **`EpochContextArtifactAdapter`** consolidating trigger/epoch boilerplate — this removes
  duplication that genuinely existed, the opposite of speculative abstraction.
- **The single subtractive correction semantic** and `accumulate_noise` composition — minimal
  and load-bearing.

The distinction is the whole point: abstraction that collapses *existing* duplication is
earned; abstraction that anticipates *hypothetical* variation is the tax.

## 6. Summary

| # | Finding | Strength | Perspective change |
|---|---------|----------|--------------------|
| 1 | Dual-framework training (15/15 torch, TF unused) | Strong | Commit to PyTorch; delete TF training path + numpy boundary |
| 2 | numpy losses + `TorchLossWrapper` (0 users, self-deprecating docstring) | Strong | Delete; use `torch.nn.Module` losses |
| 3 | Bespoke Trainer/callbacks reimplementing Lightning | Medium–strong | Adopt Lightning, or justify hand-rolling on its own merits |
| 4 | 219 committed run artifacts | Medium | Commit metrics/manifests + one hero figure; artifacts out of git |
| 5 | Inference multi-runtime, `EpochContextArtifactAdapter`, subtractive semantic | (keep) | Earned abstraction — leave as-is |

**One-line takeaway.** The branch's dominant risk is not bugs; it is *speculative
generality* — a framework-agnostic training stack and loss layer built for a second backend
that never arrived. Collapsing to PyTorch-only would remove a large fraction of the code
with zero loss of capability, and would make the remaining abstractions (which are good)
easier to see.

## 7. Related documents

- [`deep_learning_architecture_review.md`](deep_learning_architecture_review.md) — contract
  design and the multi-artifact conditioning gap (the strengths + structural changes).
- [`deep_learning_requirements.md`](deep_learning_requirements.md) — evaluation,
  reproducibility, and contributor-tooling requirements.
