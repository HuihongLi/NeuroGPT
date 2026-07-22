"""M1 — turn the canonical archive into (neuron x token) matrices for one condition.

A "condition" is a (signal, branch, layer) triple. Analyses pull a matrix per condition
and either pool tokens across stimuli (geometry / decoding) or keep the (stim, token)
structure (trajectories, lagged decoding).
"""

import numpy as np


def condition_matrix(resp, signal, branch, layer, lengths=None):
    """Return X (n_samples x n_embd) pooling valid tokens across stimuli, plus a
    (stim_index, token_index) provenance array for grouped cross-validation."""
    arr = resp[signal][branch][:, :, layer, :]          # (n_stim, T, n_embd)
    n_stim, T, C = arr.shape
    rows, prov = [], []
    for s in range(n_stim):
        Ls = T if lengths is None else int(lengths[s])
        for t in range(Ls):
            v = arr[s, t]
            if not np.isnan(v).any():
                rows.append(v)
                prov.append((s, t))
    return np.asarray(rows), np.asarray(prov)


def trajectory(resp, signal, branch, layer, stim):
    """Return the (T x n_embd) trajectory for one stimulus (NaN-padded tail trimmed)."""
    arr = resp[signal][branch][stim, :, layer, :]
    keep = ~np.isnan(arr).any(1)
    return arr[keep]


def spike_rate(resp, branch, layer, window=5):
    """Derived 'rate' signal: causal moving-average of spikes over `window` tokens.
    Returns an array shaped like resp['spikes'][branch][..., layer, :]."""
    S = resp["spikes"][branch][:, :, layer, :]
    kernel = np.ones(window) / window
    out = np.full_like(S, np.nan)
    for s in range(S.shape[0]):
        valid = ~np.isnan(S[s]).any(1)
        seq = S[s, valid]
        if len(seq):
            out[s, :len(seq)] = np.apply_along_axis(
                lambda c: np.convolve(c, kernel, mode="same"), 0, seq)
    return out


def sparsity(resp, branch, layer):
    """Population + lifetime sparseness sanity metric (should match spike_vis firing rates)."""
    S = resp["spikes"][branch][:, :, layer, :]
    valid = ~np.isnan(S).any(-1)
    flat = S[valid]
    return {"mean_rate": float(flat.mean()),
            "frac_silent_neurons": float((flat.mean(0) == 0).mean())}
