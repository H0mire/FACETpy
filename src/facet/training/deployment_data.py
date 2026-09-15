"""One dataset wrapper for every family's packing, carrying the loss's extra rows.

Thirteen editions need the same two things: the noisy context in their own tensor
layout, and a target that carries the clean and noisy signals so
:class:`facet.training.deployment_losses.RecoveredCleanObjective` can score the
*recovered* EEG rather than the predicted artifact.

The layouts themselves were already written down once, in
``src/facet/models/masterthesis/adapters.py``, for **inference**. Writing them a
second time for training is how the two drift apart, and a packing mismatch does
not raise — it produces a plausible-looking prediction of the wrong thing. So the
names here are the same names, ``tests/test_deployment_data.py`` asserts this
module reproduces the inference-side packing element for element, and any new
layout has to be added in both places or the test fails.

Channel handling splits the families in two, exactly as it does at inference:

* **Per-channel** families (``b1s``, ``bt1s``, ``bts``, ``b1ts``) see one
  electrode at a time, so one dataset example becomes ``n_channels`` training
  examples.
* **Multichannel** families (``bcts``, ``btcs``) see all electrodes at once and
  one dataset example stays one training example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from facet.training.dataset import NPZContextArtifactDataset
from facet.training.deployment_model import PACKINGS

#: Packings whose model consumes a single electrode.
PER_CHANNEL_PACKINGS = frozenset({"b1s", "bt1s", "bts", "b1ts"})


def _pack_context(ctx: np.ndarray, packing: str) -> np.ndarray:
    """``ctx`` is ``(T, S)`` for a single channel or ``(T, C, S)`` multichannel."""
    if packing == "b1s":
        return ctx[ctx.shape[0] // 2][np.newaxis, :]  # (1, S), centre only
    if packing == "bcs":
        return ctx[ctx.shape[0] // 2]  # (C, S), centre only
    if packing == "bt1s":
        return ctx[:, np.newaxis, :]  # (T, 1, S)
    if packing == "bts":
        return ctx  # (T, S)
    if packing == "b1ts":
        return ctx.reshape(1, -1)  # (1, T*S)
    if packing == "bcts":
        return ctx.transpose(1, 0, 2).reshape(ctx.shape[1], -1)  # (C, T*S)
    if packing == "btcs":
        return ctx  # (T, C, S)
    raise ValueError(f"unknown packing {packing!r}; known: {sorted(PACKINGS)}")


def model_input_shape(
    packing: str, n_channels: int = 30, context_epochs: int = 7, epoch_samples: int = 512
) -> tuple[int, ...]:
    """Per-example input shape for a packing — what facet-train injects as ``input_shape``.

    Spelled out so an edition can be built without the CLI: a probe script, a
    doctest or a trace check should not need a dataset on disk to instantiate a
    model.
    """
    return {
        "b1s": (1, epoch_samples),
        "bcs": (n_channels, epoch_samples),
        "bt1s": (context_epochs, 1, epoch_samples),
        "bts": (context_epochs, epoch_samples),
        "b1ts": (1, context_epochs * epoch_samples),
        "bcts": (n_channels, context_epochs * epoch_samples),
        "btcs": (context_epochs, n_channels, epoch_samples),
    }[packing]


def single_row_target_shape(packing: str, n_channels: int = 30, epoch_samples: int = 512) -> tuple[int, ...]:
    """Shape of **one** target row, i.e. what the model actually returns.

    facet-train injects ``target_shape`` into ``build_model`` straight from the
    dataset, and the dataset's target now carries three rows. A family that sizes
    its output head from ``target_shape`` therefore builds a head three times too
    wide — ``cascaded_context_dae`` failed with "512 must match 1536", and a
    family that silently reshaped instead would have trained on nonsense. Every
    edition passes this value instead of the stacked one.
    """
    if packing in ("b1s", "bt1s", "b1ts"):
        return (1, epoch_samples)
    if packing == "bcs":
        return (n_channels, epoch_samples)
    if packing == "bts":
        return (epoch_samples,)
    return (n_channels, epoch_samples)


class PackedDeploymentDataset:
    """Wraps :class:`NPZContextArtifactDataset` into one family's layout.

    Parameters
    ----------
    base_dataset : NPZContextArtifactDataset
        Must have been built with ``target_extras=("clean", "noisy")``.
    packing : str
        One of :data:`facet.training.deployment_model.PACKINGS`.
    max_examples : int, optional
        Cap *after* the channel expansion, so it means what it says.

    Examples
    --------
    ::

        base = NPZContextArtifactDataset(path, target_extras=("clean", "noisy"))
        ds = PackedDeploymentDataset(base, packing="bt1s")
        noisy, target = ds[0]        # (7, 1, 512) and (3, 1, 512)
    """

    def __init__(self, base_dataset: Any, packing: str, max_examples: int | None = None) -> None:
        if packing not in PACKINGS:
            raise ValueError(f"unknown packing {packing!r}; known: {sorted(PACKINGS)}")
        first_noisy, first_target = base_dataset[0]
        if first_noisy.ndim != 3:
            raise ValueError("base input must be (context_epochs, channels, samples)")
        if first_target.ndim != 3:
            raise ValueError(
                "base target must be (rows, channels, samples) — build the base dataset "
                "with target_extras=('clean', 'noisy')"
            )

        self.base_dataset = base_dataset
        self.packing = packing
        self.per_channel = packing in PER_CHANNEL_PACKINGS
        self.context_epochs = int(first_noisy.shape[0])
        self.n_channels = int(first_noisy.shape[1])
        self.epoch_samples = int(first_noisy.shape[2])
        self.n_rows = int(first_target.shape[0])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))
        self.target_rows = tuple(getattr(base_dataset, "target_rows", ("artifact", "clean", "noisy")))

        total = len(base_dataset) * (self.n_channels if self.per_channel else 1)
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        idx = int(idx)
        if self.per_channel:
            base_idx, channel = divmod(idx, self.n_channels)
            ctx, target = self.base_dataset[base_idx]
            ctx = ctx[:, channel]  # (T, S)
            target = target[:, channel]  # (rows, S)
            if self.packing != "bts":
                target = target[:, np.newaxis, :]  # (rows, 1, S)
        else:
            ctx, target = self.base_dataset[idx]  # (T, C, S), (rows, C, S)
        return (
            _pack_context(ctx, self.packing).astype(np.float32, copy=True),
            np.ascontiguousarray(target, dtype=np.float32),
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self[0][0].shape

    @property
    def target_shape(self) -> tuple[int, ...]:
        return (self.n_rows, *single_row_target_shape(self.packing, self.n_channels, self.epoch_samples))

    @property
    def n_chunks(self) -> int:
        return len(self)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        """Split by *source example*, never by channel.

        Splitting the flat index would put channel 3 of a window in train and
        channel 4 of the same window in validation. The two share the artifact
        epoch and most of their background, so the validation loss would be
        measuring memorisation and reporting generalisation.
        """
        n_base = len(self.base_dataset)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_base)
        n_val = max(1, int(n_base * val_ratio))
        val_base = set(order[:n_val].tolist())
        stride = self.n_channels if self.per_channel else 1
        train, val = [], []
        for i in range(len(self)):
            (val if (i // stride) in val_base else train).append(i)
        return _Subset(self, train), _Subset(self, val)


class _Subset:
    def __init__(self, parent: PackedDeploymentDataset, indices: list[int]) -> None:
        self._parent, self._indices = parent, indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


def build_packed_dataset(
    path: str | Path, packing: str, max_examples: int | None = None, target_key: str = "artifact_center", **_: Any
) -> PackedDeploymentDataset:
    """facet-train dataset factory shared by every deployment edition."""
    base = NPZContextArtifactDataset(
        path, target_key=target_key, demean_input=True, demean_target=True, target_extras=("clean", "noisy")
    )
    return PackedDeploymentDataset(base, packing=packing, max_examples=max_examples)


# ---------------------------------------------------------------------------
# Weg A in den Vertrag der Deployment-Editionen
# ---------------------------------------------------------------------------


class _WegABasis:
    """``NPZSpatioTemporalDataset`` in der Form, die ``PackedDeploymentDataset`` erwartet.

    Zwei Unterschiede sind zu überbrücken:

    * **Zielform.** Weg A liefert ``(Zeilen, S)``, der Packer erwartet
      ``(Zeilen, C, S)``. Die Kanalachse wird ergänzt.
    * **Zeilen.** ``RecoveredCleanObjective`` braucht ``artifact``, ``clean`` und
      ``noisy``; Weg A kennt als Extra nur ``clean`` und ``spike``, weil ``noisy``
      ableitbar ist. Es wird hier aus dem **fenstergeschobenen** Paar gerechnet,
      nicht aus dem Archiv -- sonst passte es nicht zum ebenfalls geschobenen
      Eingang.
    """

    def __init__(self, path: str | Path, include_spike: bool = False, **kwargs: Any) -> None:
        from facet.training.dataset import NPZSpatioTemporalDataset

        extras = ("clean", "spike") if include_spike else ("clean",)
        self._inner = NPZSpatioTemporalDataset(path, target_key="artifact_center", target_extras=extras, **kwargs)
        self.sfreq = float(getattr(self._inner, "sfreq", float("nan")))
        self.include_spike = bool(include_spike)
        self.target_rows = (
            ("artifact", "clean", "noisy", "spike") if self.include_spike else ("artifact", "clean", "noisy")
        )
        probe, _ = self._inner[0]
        self.context_epochs, self.n_channels, self.epoch_samples = (int(v) for v in probe.shape)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        ctx, ziel = self._inner[idx]
        artefakt, clean = ziel[0], ziel[1]
        gestapelt = np.stack([artefakt, clean, clean + artefakt], axis=0)
        if self.include_spike:
            gestapelt = np.concatenate([gestapelt, ziel[2:3]], axis=0)
        return ctx, gestapelt[:, np.newaxis, :]


class WegAPackedDataset(PackedDeploymentDataset):
    """Weg A in einer Familienpackung, mit dem **eingefrorenen** Split des Datensatzes.

    ``example_split`` im Archiv bedeutet: ``0`` Training, ``1`` Selektion,
    ``2`` gesperrter Holdout, ``-1`` an der Nahtstelle verworfen. Der Holdout ist
    ausgeschnitten worden, damit eine Konfiguration nicht auf derselben Menge
    ausgewählt und berichtet wird; er darf hier **nicht** auftauchen, auch nicht
    als Validierung. Ein zufälliger Split würde beides zunichtemachen: er zöge
    Beispiele aus dem gesperrten Teil und zerrisse ausserdem die
    Epochendisjunktheit, für die die Nahtstelle überhaupt verworfen wurde.
    """

    TRAINING, SELEKTION, GESPERRT, VERWORFEN = 0, 1, 2, -1

    def __init__(self, base_dataset: Any, packing: str, split: np.ndarray, max_examples: int | None = None) -> None:
        super().__init__(base_dataset, packing, max_examples=max_examples)
        self._split = np.asarray(split, dtype=np.int64)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        """``val_ratio`` und ``seed`` werden ignoriert -- der Split steht im Archiv."""
        del val_ratio, seed
        stride = self.n_channels if self.per_channel else 1
        train, val = [], []
        for i in range(len(self)):
            marke = int(self._split[i // stride])
            if marke == self.TRAINING:
                train.append(i)
            elif marke == self.SELEKTION:
                val.append(i)
        if not train or not val:
            raise ValueError(
                f"eingefrorener Split liefert {len(train)} Trainings- und {len(val)} "
                f"Selektionsbeispiele; Werte im Archiv: "
                f"{dict(zip(*[a.tolist() for a in np.unique(self._split, return_counts=True)], strict=False))}"
            )
        return _Subset(self, train), _Subset(self, val)


def build_weg_a_packed_dataset(
    path: str | Path,
    packing: str,
    max_examples: int | None = None,
    max_shift: int | None = None,
    background_mix_prob: float = 0.0,
    include_spike: bool = False,
    **_: Any,
) -> WegAPackedDataset:
    """Datensatzfabrik für Weg A, nutzbar von jeder Deployment-Edition.

    ``max_shift`` und ``background_mix_prob`` sind die Augmentierungen des
    Weg-A-Pfads (Fensterversatz, Hintergrundtausch). Vorgabe ist aus: run 7 hat
    ohne sie trainiert, und eine Augmentierung, die nur auf einer Seite eines
    Vergleichs läuft, macht ihn unlesbar.

    **``max_examples`` ist hier eine Falle.** Es schneidet vorne ab, und vorne
    liegt ausschliesslich Training -- der Selektionsteil kommt erst danach. Ein
    Probelauf mit kleinem ``max_examples`` bekommt deshalb null
    Validierungsbeispiele; ``WegAPackedDataset.train_val_split`` bricht dann mit
    einer Meldung ab, statt stillschweigend ohne Validierung zu laufen. Für einen
    kurzen Probelauf lieber ``max_epochs`` verkleinern.
    """
    basis = _WegABasis(
        path,
        max_shift=max_shift,
        background_mix_prob=background_mix_prob,
        demean_input=True,
        demean_target=True,
        include_spike=include_spike,
    )
    with np.load(Path(path).expanduser(), allow_pickle=True) as b:
        split = b["example_split"]
    return WegAPackedDataset(basis, packing=packing, split=split, max_examples=max_examples)
