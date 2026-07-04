# Design Lesson: Speculative Generality vs. Earned Abstraction

**Status:** Reflection / thesis material
**Owner:** Janik Müller
**Context:** A methodological lesson distilled from the engineering of the FACETpy
deep-learning artifact-correction subsystem (`feature/add-deeplearning`).
**Intended use:** Discussion / "lessons learned" material for the thesis, and a
decision heuristic for future contributors.
**Last updated:** 2026-06-09

---

## 1. The lesson in one sentence

> Add abstraction to **collapse duplication that already exists**, not to **anticipate
> variation that might exist**; flexibility you do not use is not flexibility, it is
> surface area you must maintain.

## 2. The concept

**Speculative generality** is a named design smell (Fowler, *Refactoring*, 1999): code
made general "in case we need it later," where the later case has not arrived and shows no
concrete sign of arriving. It is the structural symptom of violating **YAGNI** — *You
Aren't Gonna Need It* (Beck, Extreme Programming). The cost is paid immediately (extra
indirection, dead paths, maintenance, cognitive load); the benefit is conditional on a
predicted future.

The diagnostic tell is the *justification*. A generality is speculative when the sentence
defending it begins with **"what if someone wants to…"** or **"this way we could also…"**
and the "could" has not happened. It is **earned** when the sentence is **"this is copied
in N places today."**

Two layers can be mechanically identical — both a level of indirection — yet have opposite
economics:

| | Earned abstraction | Speculative generality |
|---|---|---|
| Generalizes over | cases that exist now | cases imagined for later |
| Pays off | immediately (removes real duplication) | conditionally (if the future arrives) |
| Risk | low — the shape is known | high — the predicted shape is usually wrong |
| When the real case arrives | already fits | rarely fits; refactored anyway |

The deep point: you are usually **wrong about the shape** of a future requirement. So
pre-building for it is doubly costly — you pay now, and when the genuine second case
appears it almost never matches the abstraction you guessed, so you refactor regardless.
Waiting until you have two or three *concrete* cases lets the abstraction be derived from
real constraints instead of imagined ones ("the rule of three").

## 3. Case studies from this project

These are real instances encountered while building and reviewing the DL subsystem. They
are valuable precisely because they were committed in good faith — speculative generality
rarely looks foolish at the time; it looks responsible.

### 3.1 The artifact-type enum (caught at design time)

A structured `artifact_type` enum (`gradient` / `bcg` / `custom`) was proposed so the
contract could "handle all artifact kinds." But (a) the grouping need was already met by a
free-text `tags` field, and (b) the thing that actually needed structure was not the
*label* but the model's *conditioning requirement* (which trigger source it consumes) — an
orthogonal axis. A closed enum would also have fought the stated goal of supporting
*future, unspecified* artifacts, since each new artifact would require editing core.

**Resolution:** keep the label a free string; structure only the conditioning requirement,
which is a real, existing constraint. *Lesson: structure the axis that has teeth, leave the
human-facing label flexible.*

### 3.2 Dual-framework training (committed, recommend cutting)

The training stack was built **framework-agnostic** — a numpy boundary in the trainer, a
`TensorFlowModelWrapper` beside the PyTorch one, TF inference adapters and tensor-layout
enums — so the toolbox "could" support TensorFlow models. Outcome: **15 of 15 models are
PyTorch; zero are TensorFlow.** The agnosticism is the *only* justification for the numpy
boundary, which round-trips every batch numpy→torch→numpy for no benefit.

**Lesson: the second backend never arrived, and its absence cost a whole parallel code
path plus a hot-loop tax.** A `CUSTOM` escape hatch would have preserved the option at
near-zero cost.

### 3.3 The numpy loss layer + `TorchLossWrapper` (committed, recommend deleting)

Losses were made framework-agnostic by implementing them in numpy. But a numpy loss cannot
backpropagate, so a bridge class (`TorchLossWrapper`) was needed to pair a numpy *logging*
loss with a separate torch *gradient* loss — which introduced a footgun (optimise one loss
while logging another) that required a dedicated *hardening commit* to fix. End state:
**zero models use any of it**, and the wrapper's own docstring recommends implementing
losses directly as `torch.nn.Module` instead.

**Lesson: the abstraction manufactured a problem (non-differentiable losses), then a class
to manage the problem, then a bug in that class — all for flexibility nobody exercised.**
This is the purest specimen: generality that is net-negative even before counting the
maintenance.

### 3.4 The bespoke Trainer/callbacks (committed, defensible — but for a different reason)

~2,300 lines re-derive a training loop, a Keras/Lightning-style callback API, and
checkpoint/early-stopping configs that PyTorch Lightning already provides. The *stated*
justification was framework-agnostic portability (§3.2) — speculative. But there is a
*legitimate* non-speculative justification hiding underneath: a dependency-free, fully
transparent loop that a thesis author understands end-to-end has real pedagogical and
debugging value.

**Lesson: a speculative justification can mask a sound one.** The fix is not always to
delete — it is to **re-justify on real grounds or not at all.** If the honest reason is "I
want to own the loop," say that; do not lean on portability that does not exist.

## 4. The decision heuristic

Before adding a layer of generality, ask in order:

1. **How many concrete cases need it *today*?** Zero → don't. One → don't (write the one
   case directly). Two or three with a clear shared shape → now abstract.
2. **What is the cheapest escape hatch?** A string field, a `CUSTOM` enum member, a plugin
   hook. Prefer the escape hatch over the full implementation until forced.
3. **Is this collapsing duplication or anticipating variation?** Collapsing → likely
   earned. Anticipating → likely speculative.
4. **What does it cost if I'm wrong about the future shape?** If the answer is "refactor
   anyway," the pre-build bought nothing.

## 5. Why this matters specifically for ML research codebases

ML research code is unusually prone to speculative generality, which makes it good thesis
material:

- **Framework-agnosticism is seductive** ("support torch *and* TF *and* JAX") but the
  research almost always settles on one framework; the abstraction outlives its rationale.
- **Architecture zoos** (here, 15 model families) invite a "support every possible model"
  contract, when the thesis only needs to compare a chosen few rigorously.
- **The cost is hidden** because research code is often write-once: the dead TF path never
  errors, so it never announces that it is dead. It simply enlarges every future reader's
  search space.

The counter-discipline: build for the experiment in front of you, keep escape hatches
cheap, and let the *second real need* — not the imagined one — drive every abstraction.

## 6. Using this in the thesis

This makes a strong **methodology / discussion** point: the engineering contribution of
the thesis is not only the models but a *contract* for integrating them, and a defensible
contract is one that resists premature generality. The three committed case studies
(§3.2–3.4) are honest, concrete evidence — far more convincing than asserting "the design
is clean." Framing the artifact-type decision (§3.1) as *caught at design time* versus the
others as *caught in review* also demonstrates a maturing design judgment over the course
of the work, which is itself a thesis-worthy narrative.

## 7. References

- M. Fowler, *Refactoring: Improving the Design of Existing Code*, 1999 — "Speculative
  Generality" smell; the "rule of three."
- K. Beck, *Extreme Programming Explained*, 1999 — YAGNI.
- Related project documents:
  [`deep_learning_critical_review.md`](deep_learning_critical_review.md) (the full
  evidence), [`deep_learning_architecture_review.md`](deep_learning_architecture_review.md)
  (the conditioning-source counterpoint of earned abstraction).
