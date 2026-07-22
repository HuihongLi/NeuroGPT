"""Publication figures for the geometry + decoding + V-vs-S comparison.

Follow the repo `plot-style` skill conventions. Reuse the Tol muted palette already used
in src/spike_vis.py so figures match the existing spikegpt_activity.png aesthetic.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tol muted palette (same as src/spike_vis.py)
C = {"blue": "#4477AA", "red": "#CC3311", "green": "#228833",
     "orange": "#EE7733", "purple": "#AA3377", "cyan": "#66CCEE", "gray": "#BBBBBB"}
SIGNAL_COLOR = {"voltage": C["blue"], "spikes": C["orange"], "current": C["green"]}


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def dimensionality_vs_depth(curves, out_path, ylabel="Participation ratio"):
    """H1: overlay per-layer geometry metric for voltage / spikes / current."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _clean(ax)
    for sig, y in curves.items():
        ax.plot(range(len(y)), y, "-o", ms=3, color=SIGNAL_COLOR.get(sig), label=sig)
    ax.set_xlabel("Block index"); ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); return fig


def decoding_gap(gap, out_path, target=""):
    """H2: acc(voltage) and acc(spikes) vs layer + the gap on a twin axis."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _clean(ax)
    ax.plot(gap["voltage"], "-o", ms=3, color=SIGNAL_COLOR["voltage"], label="voltage")
    ax.plot(gap["spikes"], "-s", ms=3, color=SIGNAL_COLOR["spikes"], label="spikes")
    ax.set_xlabel("Block index"); ax.set_ylabel(f"Decode {target} (bal. acc.)")
    ax2 = ax.twinx()
    ax2.bar(range(len(gap["gap"])), gap["gap"], alpha=0.15, color=C["red"])
    ax2.set_ylabel("V − S gap"); ax2.spines["top"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); return fig


def trajectories(traj_by_signal, out_path, layer=0):
    """PC1–PC2 trajectory of one stimulus for voltage vs spikes side by side."""
    fig, axes = plt.subplots(1, len(traj_by_signal), figsize=(4 * len(traj_by_signal), 3.4))
    for ax, (sig, tr) in zip(np.atleast_1d(axes), traj_by_signal.items()):
        _clean(ax)
        ax.plot(tr[:, 0], tr[:, 1], "-", color=SIGNAL_COLOR.get(sig), lw=1)
        ax.scatter(tr[0, 0], tr[0, 1], color=C["green"], s=25, zorder=3, label="start")
        ax.scatter(tr[-1, 0], tr[-1, 1], color=C["red"], s=25, zorder=3, label="end")
        ax.set_title(f"{sig} (L{layer})", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); return fig


import numpy as np  # noqa: E402  (used by trajectories)
