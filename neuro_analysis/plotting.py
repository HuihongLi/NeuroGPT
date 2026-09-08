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
SIGNAL_COLOR = {"voltage": C["blue"], "spikes": C["orange"], "current": C["green"],
                "joint": C["purple"], "driven": C["blue"], "spont": C["red"]}


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


def decoding_gap(gap, out_path, target="", null=None):
    """H2: acc(voltage) and acc(spikes) vs layer + the gap on a twin axis.
    `null` (optional) = shuffled-label baseline curve, drawn as a dashed reference."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _clean(ax)
    ax.plot(gap["voltage"], "-o", ms=3, color=SIGNAL_COLOR["voltage"], label="voltage")
    ax.plot(gap["spikes"], "-s", ms=3, color=SIGNAL_COLOR["spikes"], label="spikes")
    if null is not None:
        ax.plot(null, "--", lw=1, color=C["gray"], label="shuffled null")
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


def line_vs_depth(curves, out_path, ylabel, title="", hline=None):
    """Generic per-layer overlay for {label: (n_layer,) array}."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _clean(ax)
    markers = "os^v"
    for (label, y), mk in zip(curves.items(), markers * 4):
        ax.plot(range(len(y)), y, f"-{mk}", ms=3,
                color=SIGNAL_COLOR.get(label, None), label=label)
    if hline is not None:
        ax.axhline(hline, color=C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("Block index"); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); return fig


def bar_vs_depth(y, out_path, ylabel, title="", color=None):
    """Per-layer bar chart (e.g. spike–uncertainty correlation, ISI-CV)."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _clean(ax)
    ax.bar(range(len(y)), y, color=color or C["purple"])
    ax.axhline(0, color=C["gray"], lw=0.8)
    ax.set_xlabel("Block index"); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); return fig


def psd_overlay(psds, out_path, title="synthetic LFP power spectrum"):
    """log-log PSD overlay: {label: (freq_grid, mean_log10_power)}. Frequency in cycles/token."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _clean(ax)
    for label, (f, lp) in psds.items():
        ax.plot(np.log10(f), lp, "-o", ms=3, color=SIGNAL_COLOR.get(label), label=label)
    ax.set_xlabel("log10 frequency (cycles/token)")
    ax.set_ylabel("log10 power")
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); return fig


def lag_curves(curves, out_path):
    """H3/memory: decoding accuracy vs lag for voltage vs spikes."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _clean(ax)
    for sig, d in curves.items():
        lags = sorted(d)
        ax.plot(lags, [d[k] for k in lags], "-o", ms=3,
                color=SIGNAL_COLOR.get(sig), label=sig)
    ax.set_xlabel("Lag (tokens into the past)"); ax.set_ylabel("Balanced accuracy")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); return fig


import numpy as np  # noqa: E402  (used by trajectories)
