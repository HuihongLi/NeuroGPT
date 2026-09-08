"""Rollout dynamics — the state-space signature of self-generation vs being driven.

Motivating question (now the project's primary axis): when SpikeGPT runs on its own
predictions (spontaneous) instead of ground-truth text (driven), does its population
trajectory move differently — faster/slower, more/less persistent, more recurrent (looping
toward attractors)? These metrics operate on the per-sequence voltage trajectory V[t] and
on the generated token stream.

All curves are per-layer so they can be overlaid driven-vs-spontaneous.
"""

import numpy as np

from . import track
from .representations import trajectory


# ------------------------------------------------------------------ trajectory geometry
def _seq_trajectories(resp, signal, branch, layer):
    n_stim = resp[signal][branch].shape[0]
    trs = [trajectory(resp, signal, branch, layer, s) for s in range(n_stim)]
    return [t for t in trs if len(t) >= 3]


def speed_by_layer(resp, branch, n_layer, signal="voltage"):
    """Mean step speed ‖V[t]−V[t−1]‖ per layer (how fast the state moves per token)."""
    out = np.full(n_layer, np.nan)
    for l in range(n_layer):
        sp = [np.linalg.norm(np.diff(t, axis=0), axis=1).mean()
              for t in _seq_trajectories(resp, signal, branch, l)]
        if sp:
            out[l] = float(np.mean(sp))
    return out


def velocity_autocorr_by_layer(resp, branch, n_layer, signal="voltage"):
    """Lag-1 cosine autocorrelation of the velocity vector per layer. High (→1) = the
    trajectory keeps moving in a consistent direction (smooth flow); low/→0 = it jitters /
    reverses (diffusive). Distinguishes directed drift from random-walk-like motion."""
    out = np.full(n_layer, np.nan)
    for l in range(n_layer):
        vals = []
        for t in _seq_trajectories(resp, signal, branch, l):
            d = np.diff(t, axis=0)
            if len(d) < 2:
                continue
            num = (d[:-1] * d[1:]).sum(1)
            den = np.linalg.norm(d[:-1], axis=1) * np.linalg.norm(d[1:], axis=1) + 1e-12
            vals.append((num / den).mean())
        if vals:
            out[l] = float(np.mean(vals))
    return out


def recurrence_by_layer(resp, branch, n_layer, signal="voltage", eps_quantile=0.1, min_gap=3):
    """Recurrence rate per layer: fraction of token pairs (t, t') with |t−t'|>min_gap whose
    states are closer than a per-sequence distance threshold (the eps_quantile of all pairwise
    distances). High recurrence ⇒ the trajectory revisits earlier states — a signature of
    attractor/looping dynamics, expected to rise in degenerate self-rollout."""
    out = np.full(n_layer, np.nan)
    for l in range(n_layer):
        rates = []
        for t in _seq_trajectories(resp, signal, branch, l):
            T = len(t)
            D = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=2)
            iu = np.triu_indices(T, k=min_gap)
            if iu[0].size == 0:
                continue
            d = D[iu]
            thr = np.quantile(d, eps_quantile)
            rates.append(float((d < thr).mean()))
        if rates:
            out[l] = float(np.mean(rates))
    return out


# ------------------------------------------------------------------ token-level looping
def token_repetition(gen_tokens, window=16):
    """Fraction of generated tokens that already appeared in the preceding `window` tokens —
    a direct index of self-rollout degeneration/looping (spontaneous condition only).
    gen_tokens : list[list[int]] returned by capture.capture_rollout."""
    reps = []
    for toks in gen_tokens:
        if len(toks) < 2:
            continue
        hit = sum(toks[i] in toks[max(0, i - window):i] for i in range(1, len(toks)))
        reps.append(hit / (len(toks) - 1))
    return float(np.mean(reps)) if reps else np.nan


def dynamics_contrast(resp_driven, resp_spont, branch, n_layer):
    """Per-layer driven-vs-spontaneous dynamics table (dict of arrays)."""
    out = {"layer": np.arange(n_layer)}
    for name, fn in (("speed", speed_by_layer),
                     ("velcorr", velocity_autocorr_by_layer),
                     ("recurrence", recurrence_by_layer)):
        out[f"{name}_driven"] = fn(resp_driven, branch, n_layer)
        for _ in track([0], desc=f"dynamics:{name}", leave=False):
            out[f"{name}_spont"] = fn(resp_spont, branch, n_layer)
    return out
