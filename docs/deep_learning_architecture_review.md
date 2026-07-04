# Deep-Learning Subsystem — Architecture & Contract Review

**Status:** Draft
**Owner:** Janik Müller
**Context:** `feature/add-deeplearning` branch, master's thesis
**Scope:** Technical/conceptual review of the DL integration contracts, training abstraction, and extensibility. The training-data / ground-truth (AAS-surrogate) question is tracked separately as a known TODO and is **out of scope** here.
**Last updated:** 2026-06-09

---

## 1. Summary rating

| Dimension | Rating | One-line |
|-----------|--------|----------|
| Engineering / abstraction quality | **9 / 10** | One of the cleaner DL-integration contracts in a research codebase. |
| Conceptual fit to the multi-artifact goal | **6 / 10** | Abstracts the *model* well; the context can't represent a model's conditioning needs (named trigger sources). |
| Training-contract generality | **5 / 10** | Built for single-loss regression; GANs/diffusion are shoehorned in. |
| Contributor extensibility | **7 / 10** | Strong base classes; weak discovery and duplicated per-model surface. |

The correction-side contract is excellent. The two substantive weaknesses are (a) the
context cannot represent which conditioning signal a model consumes (so a BCG model and
a gradient model are indistinguishable to the validator), and (b) the training wrapper
is shaped for supervised regression and strains under adversarial/diffusion families —
which matters precisely because the thesis compares model *families*. Note: the artifact
*label* is deliberately **not** something to structure into an enum — it stays a free
string for grouping (see C0a).

## 2. What the design gets conceptually right

- **One subtractive correction semantic.** Every model — `artifact`, `clean`, or
  `both` — collapses to an artifact estimate subtracted from the raw, with `clean`
  reconstructed as `original − artifact` and a consistency check between the two
  (`_resolve_artifact_prediction`, `deep_learning.py:2290`). The pipeline has exactly
  one correction operation to reason about; multi-artifact correction is just stacking
  processors; `accumulate_noise` (additive) composes their removed-artifact estimates
  correctly (`context.py:264`).
- **Spec-driven declarative validation.** `DeepLearningModelSpec` front-loads runtime
  availability, checkpoint-format/runtime compatibility, trigger/artifact-length/
  channel-position prerequisites, execution-granularity legality, and chunk-geometry
  invariants (`__post_init__`, `deep_learning.py:216`). Contributors declare
  capabilities; the framework enforces them before inference runs.
- **Three orthogonal execution modes that compose.** single-pass / chunked-with-
  overlap-add / trigger-aligned, optionally wrapped by a channel-group outer loop with
  index- or position-based neighbor selection (`process`, `deep_learning.py:2506`). The
  constant-overlap-add tapered window (`_overlap_add_window`, `deep_learning.py:28`)
  cross-fades chunk seams, and the no-overlap case reduces to the old rectangular
  behavior through the same path.
- **Framework-agnostic training boundary.** numpy at the wrapper edge, framework
  tensors inside, `to_inference_adapter()` closing the train→infer loop, and a registry
  giving config round-trip (`to_config_dict` / `from_config_dict`) for registered,
  self-registering adapters.

## 3. Contract changes to make

Priority: **C0** (blocks the multi-artifact goal structurally) ·
**C1** (materially improves generality) · **C2** (extensibility polish).

### C0 — The spec conflates two separable concerns; only one needs structure

There are two different jobs hiding behind "what artifact does this model target," and
they should not be solved by the same mechanism.

**Job A — identity / grouping** ("which models are trying to do the same thing"):
used for evaluation comparison, docs, and discovery. This needs a label, nothing more.
A closed enum (`gradient` / `bcg` / …) is the *wrong* tool here — it fights the
"future, currently unspecified artifacts" goal because every new artifact would require
editing core. The flexibility already exists: the spec has a `tags: tuple[str, ...]`
field, and `"bcg_aware"` is already used as one (`deep_learning.py:1667`).

> **Recommendation (A).** Do **not** add an `artifact_type` enum. For grouping, either
> reuse `tags` or add a single free-form `artifact_target: str` field for a canonical,
> singular key. No validation behavior is attached to this label — it is for humans and
> for grouping evaluation runs.

**Job B — the actual contract problem** ("does this model have the conditioning input
it needs, and is it the *right* one"): this is the part with teeth, and crucially it
**should not be keyed off the artifact label at all.** What disambiguates a stacked
gradient→BCG pipeline is not the string `"bcg"`; it is that the BCG model *consumes
cardiac R-peaks* while the gradient model *consumes volume markers*.

**Consequences of the current design (Job B gap).**
- A BCG model and a gradient model are indistinguishable at the contract level; both
  read the same `context.metadata.triggers` blindly.
- The pipeline cannot validate "this model needs cardiac R-peaks, not volume markers,"
  nor warn when gradient triggers are silently fed to a BCG model.
- Stacking gradient→BCG correction is semantically ambiguous about which triggers apply.

**Why decoupling label from conditioning is the better design.** Binding behavior to an
artifact taxonomy would be wrong even if the taxonomy were open: two models with the
same label can condition differently (one BCG model uses ECG R-peaks, another estimates
them from EEG, a third needs none). A model's *goal* (a string, for people) and its
*mechanical inputs* (structured, for the validator) are orthogonal axes. So the only
thing that needs structure is the conditioning requirement — see C0b, which is the real
fix. The artifact label is a red herring wrapped around it.

### C0b — Named conditioning sources (the one structural change)

**Finding.** `ProcessingMetadata` (`context.py:24`) is built around the fMRI gradient
problem: a single `triggers` array plus `artifact_to_trigger_offset`,
`slices_per_volume`, `volume_gaps`, `upsampling_factor`, `artifact_length`. There is no
concept of multiple, named conditioning sources.

**Why it matters.** BCG is not "gradient with different triggers" — its timing is
quasi-periodic and derived from ECG/EEG, and a realistic pipeline needs *both* trigger
families present at once. The current single-`triggers` schema cannot represent that.

**Recommendation.** Introduce named annotation/trigger *sources* on the context
(e.g. `{"volume": ..., "rpeak": ...}`) and have each model declare which it consumes
(a structured `conditioning_requirements` on the spec). The processor validates *this*,
not the artifact label. This — not any artifact taxonomy — is what makes "gradient,
BCG, future artifacts" true at the architecture level rather than aspirational. A new
artifact needs a new conditioning source only *if* it actually requires new
conditioning; otherwise it is just a new `artifact_target` string (C0a).

### C1 — The training wrapper is single-loss-regression-shaped

**Finding.** `TrainableModelWrapper.train_step` returns one loss and the wrapper owns
one optimizer (`wrapper.py:75`). To make DHCT-GAN fit, the discriminator and its own
optimizer live **inside the generator `nn.Module`**, and `forward()` performs a
discriminator update as a side effect gated on `torch.is_grad_enabled()`
(`models/dhct_gan_v2/training.py:344`).

**Why it matters.** It works, and the code is honest about why, but it couples
optimization into the module, relies on grad-enabled state to distinguish
train/eval/inference, and will not generalize to the next adversarial or diffusion
variant. For a thesis whose premise is comparing model *families* including GANs and
diffusion, this is the weakest abstraction.

**Recommendation.** Extend the wrapper contract to natively support multi-objective
optimization: either let `train_step` own its own stepping (the wrapper provides
tensor-conversion / clipping utilities, not the loop), or add an optional
`configure_optimizers()`-style hook returning N optimizers. This removes the
"hide the second optimizer in `forward()`" smell and makes GAN/diffusion first-class.

### C2 — Model discovery is lazy and manual

**Finding.** `models/__init__.py` is empty; adapters register only as an import side
effect, so `get_deep_learning_model("dpae")` raises unless the caller already imported
`facet.models.dpae`.

**Recommendation.** Add entry-point-based or lazy-by-name discovery so the registry is
populated without the caller knowing the import path.

### C2 — Audit per-model `Correction` subclasses

**Finding.** There is near-duplicated adapter/processor scaffolding across the 15
models, and at least one ships its own `…Correction` processor subclass
(`DualPathwayAutoencoderCorrection`) although `models/README.md` says to avoid
model-specific processors unless the generic contract is insufficient.

**Recommendation.** Audit whether those subclasses earn their keep or should collapse
into `DeepLearningCorrection` + adapter. Document the bar for a model-specific processor.

### C2 — Config round-trip constraints (document, don't necessarily change)

**Finding.** `from_config_dict` reconstructs adapters via a fixed
`(checkpoint_path, spec_overrides)` constructor signature (`deep_learning.py:2634`) and
cannot round-trip `NumpyInferenceAdapter` runtime callables.

**Recommendation.** Document this as an explicit contract constraint on adapter
constructors so contributors don't author adapters that silently break serialization.

## 4. Premade code for contributors — more or less?

- **Keep the generous core.** `DeepLearningModelAdapter`, `EpochContextArtifactAdapter`,
  shared `_overlap_add_window` / `_resample_1d`, and spec-driven validation are at the
  right altitude; `EpochContextArtifactAdapter` already absorbs the right boilerplate.
- **Add premade code** in the C0/C1 places: named-conditioning-source abstraction
  (C0b), multi-optimizer training hook (C1), auto-discovery (C2), plus the shared
  evaluation harness tracked in the requirements document. (A free-form `artifact_target`
  label, C0a, is a trivial addition with no new machinery.)
- **Reduce** the per-model surface (C2 audit above): collapse duplicated scaffolding and
  unjustified model-specific processors.

## 5. Most important next development step (technical)

**Introduce named conditioning sources so BCG is a first-class citizen, not
gradient-with-different-triggers** (C0b). Let `ProcessingContext` carry named
trigger/annotation sources (e.g. `{"volume": ..., "rpeak": ...}`) and add structured
`conditioning_requirements` to `DeepLearningModelSpec` so each model declares which
source(s) it consumes; the processor validates that. The artifact *label* (C0a) is a
free string for grouping only — not part of this step and not a taxonomy. Until named
conditioning exists, the multi-artifact thesis claim rests on a contract shaped
entirely around one artifact. The wrapper multi-optimizer hook (C1), discovery, and
eval harness are valuable but secondary to making the framework structurally honest
about being multi-artifact.

## 6. Change summary

| ID | Priority | Change | Primary files |
|----|----------|--------|---------------|
| C0a | C0 | Free-form `artifact_target: str` label (or reuse `tags`) for grouping only — **no enum, no validation behavior** | `correction/deep_learning.py` |
| C0b | C0 | Named multi-source triggers/annotations on the context + `conditioning_requirements` on the spec; validate in processor | `core/context.py`, `correction/deep_learning.py` |
| C1  | C1 | Multi-optimizer / custom-optimization training hook | `training/wrapper.py`, `training/trainer.py` |
| C2a | C2 | Entry-point / lazy model discovery | `models/__init__.py`, registry |
| C2b | C2 | Audit & collapse per-model `Correction` subclasses | `models/*/processor.py` |
| C2c | C2 | Document adapter-constructor constraint for config round-trip | `correction/deep_learning.py` |

## 7. Related documents

- [`deep_learning_requirements.md`](deep_learning_requirements.md) — evaluation,
  reproducibility, and contributor-tooling requirements (the experimental side).
- [`../src/facet/models/README.md`](../src/facet/models/README.md) — core-vs-model
  governance rules referenced throughout this review.
