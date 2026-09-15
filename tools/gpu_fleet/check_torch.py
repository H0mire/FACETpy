"""Verify torch import and CUDA availability after `uv sync`.

Run on the RunPod immediately before training so a broken venv aborts the job
with a clear message instead of silently falling back to CPU.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        import torch
    except ImportError as exc:
        print(f"ERROR: torch import failed: {exc}", file=sys.stderr)
        print(
            "The .venv on this pod may have lost access to the system PyTorch. "
            "Check the worker environment and its CUDA-enabled PyTorch installation; "
            "use the worker’s intended interpreter and dependency setup.",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available():
        print(
            f"ERROR: torch.cuda.is_available() is False (torch {torch.__version__})",
            file=sys.stderr,
        )
        print(
            "Either the venv shadowed the CUDA-enabled system torch, or the pod "
            "has no GPU exposed. Check the worker environment and "
            "verify nvidia-smi works on this pod.",
            file=sys.stderr,
        )
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__} cuda OK on {gpu_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
