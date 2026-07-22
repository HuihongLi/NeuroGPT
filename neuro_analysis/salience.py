"""M4 — salience analysis (H2): do spikes preferentially mark salient/uncertain tokens?

Relates per-token spiking to token-level uncertainty from the model's OWN logits:
    surprisal = -log p(observed token)     (how surprising this token was)
    entropy   = H[p(next token)]           (how uncertain the model is about what comes next)
Both are produced during capture (see capture.capture / datasets.build_labels) and stored
in the labels dict, so no extra forward passes are needed.
"""

import numpy as np


def population_spike_count(resp, branch, layer):
    """Per-(stim, token) count of spiking neurons at one layer -> (n_stim, T), NaN-padded."""
    S = resp["spikes"][branch][:, :, layer, :]
    cnt = S.sum(-1)
    cnt[np.isnan(S).any(-1)] = np.nan
    return cnt


def _pad_label(label_ragged, T):
    """Pad a ragged (list-per-stim) label to a (n_stim, T) float grid with NaN tails,
    so it aligns with the padded population-spike-count grid."""
    out = np.full((len(label_ragged), T), np.nan)
    for s, row in enumerate(label_ragged):
        out[s, : len(row)] = np.asarray(row, dtype=float)
    return out


def spike_uncertainty_corr(resp, labels, branch, n_layer, key="surprisal"):
    """H2: Pearson r between population spike count and token surprisal/entropy, per layer.
    Positive r supports 'spikes mark salient events'. Compare to a voltage-magnitude control
    (does V track uncertainty as sharply as S flags it?)."""
    T = resp["spikes"][branch].shape[1]
    u = _pad_label(labels[key], T).ravel()
    out = np.full(n_layer, np.nan)
    for l in range(n_layer):
        c = population_spike_count(resp, branch, l).ravel()
        m = ~(np.isnan(c) | np.isnan(u))
        if m.sum() > 2:
            out[l] = np.corrcoef(c[m], u[m])[0, 1]
    return out


def spike_triggered_contrast(resp, labels, branch, layer, key="surprisal", q=0.25):
    """Compare mean spike count for high- vs low-uncertainty tokens (top/bottom quartile).
    Returns (mean_high, mean_low, cohens_d). The core H2 effect-size statistic.

    TODO (M4): add a matched-position / matched-frequency control so the contrast is not
    confounded by sentence position or token frequency.
    """
    raise NotImplementedError("Implement quartile split + Cohen's d per docstring.")
