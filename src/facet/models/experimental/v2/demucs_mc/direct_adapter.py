"""Pipeline adapter for a *direct* artifact model — FARM replaced, not preceded.

The cascade adapter refuses to run without FARM, because its input is
``signal - template``. This adapter is the counterpart for the models that were
trained on the raw signal and predict the **whole** artifact, so they can take
FARM's place in the chain rather than follow it.

Everything else about the input contract is identical to the cascade: seven
trigger-to-trigger epochs resampled to 512 samples, the target electrode first
followed by its nearest neighbours. Only the expected input differs, which is
exactly the distinction chapter 5.5 is about.

A warning that belongs next to this class rather than in a footnote: on the
Weg-A evaluation these direct models score *worse than returning zero*
(5.4.2/5.5.3). Using this adapter in place of FARM is therefore a legitimate
experiment, not a recommended configuration.
"""

from __future__ import annotations

import numpy as np

from facet.core import ProcessingContext, ProcessorValidationError
from facet.correction.deep_learning import DeepLearningModelSpec
from facet.models.experimental.v2.demucs_mc.cascade_adapter import CascadeDemucsAdapter


class DirectDemucsAdapter(CascadeDemucsAdapter):
    """Predicts the full artifact from the raw signal; replaces FARM in the chain."""

    spec = DeepLearningModelSpec(
        **{
            **CascadeDemucsAdapter.spec.__dict__,
            "name": "DirectDemucsAdapter",
            "description": "Direct multichannel Demucs predicting the full gradient artifact from the raw signal.",
            "tags": ("demucs", "direct", "artifact_prediction"),
        },
    )

    def validate_context(self, context: ProcessingContext) -> None:
        # Deliberately NOT calling CascadeDemucsAdapter.validate_context: its
        # FARM-ordering check is the one thing that must not apply here.
        super(CascadeDemucsAdapter, self).validate_context(context)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ProcessorValidationError("context_epochs must be a positive odd integer")
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < self.context_epochs + 1:
            raise ProcessorValidationError(
                f"Need at least {self.context_epochs + 1} triggers for a "
                f"{self.context_epochs}-epoch context, got {len(triggers)}"
            )
        history = [entry.get("processor", "") for entry in getattr(context, "history", []) or []]
        if any("farm" in h.lower() or "aas" in h.lower() for h in history):
            raise ProcessorValidationError(
                "DirectDemucsAdapter predicts the FULL artifact from the raw signal, but FARM "
                "has already removed most of it. Use CascadeDemucsAdapter after FARM, or place "
                f"this adapter instead of FARM. History so far: {history}"
            )
