"""Plotting utilities shared by interactive processors."""

from __future__ import annotations

from matplotlib import pyplot as plt


def show_matplotlib_figure(fig, poll_interval: float = 0.05) -> None:
    """Show one matplotlib figure and wait until the user closes it.

    Uses ``plt.pause`` instead of manual ``flush_events`` to keep backend
    event loops responsive across backends.
    """
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        plt.pause(poll_interval)
