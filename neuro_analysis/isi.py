"""M5 — firing irregularity (H4): ISI and CV-ISI by layer and linguistic condition.

Time is measured in TOKEN STEPS (the RNN's unit of time), so an inter-spike interval is the
number of tokens between two spikes of the same neuron. CV-ISI = std(ISI)/mean(ISI) is the
standard irregularity index (1.0 ≈ Poisson; <1 regular; >1 bursty).
"""

import numpy as np


def neuron_isis(spike_train):
    """ISIs (in token steps) for one neuron's 1-D binary spike train."""
    idx = np.flatnonzero(spike_train > 0)
    return np.diff(idx) if len(idx) > 1 else np.array([])


def cv_isi_by_layer(resp, branch, n_layer):
    """Mean CV-ISI across neurons/stimuli at each layer -> (n_layer,). Depth profile of
    firing irregularity."""
    out = np.full(n_layer, np.nan)
    for l in range(n_layer):
        S = resp["spikes"][branch][:, :, l, :]     # (n_stim, T, n_embd)
        cvs = []
        for s in range(S.shape[0]):
            valid = ~np.isnan(S[s]).any(-1)
            seq = S[s, valid]
            for n in range(seq.shape[1]):
                isi = neuron_isis(seq[:, n])
                if len(isi) > 1 and isi.mean() > 0:
                    cvs.append(isi.std() / isi.mean())
        if cvs:
            out[l] = float(np.mean(cvs))
    return out


def cv_isi_by_condition(resp, labels, branch, layer,
                        condition_key="entropy", hi_q=0.75, lo_q=0.25):
    """H4: contrast CV-ISI during high- vs low-uncertainty tokens (ambiguity / prediction
    uncertainty / semantic drift, selected via `condition_key`). Returns
    dict(cv_high, cv_low, delta).

    Simplification: ISIs are computed over the *subset* of tokens selected by the condition
    mask (concatenated), so an interval can span a skipped low-condition token. This measures
    irregularity of firing restricted to high/low-condition tokens rather than within
    contiguous windows — adequate for a first pass; document if reported.
    """
    from .salience import _pad_label
    T = resp["spikes"][branch].shape[1]
    u = _pad_label(labels[condition_key], T)                 # (n_stim, T)
    S = resp["spikes"][branch][:, :, layer, :]               # (n_stim, T, n_embd)
    hi_th, lo_th = np.nanquantile(u, hi_q), np.nanquantile(u, lo_q)

    def mean_cv(mask):
        vals = []
        for s in range(S.shape[0]):
            sel = mask[s] & ~np.isnan(S[s]).any(-1)
            seq = S[s, sel]
            for n in range(seq.shape[1]):
                isi = neuron_isis(seq[:, n])
                if len(isi) > 1 and isi.mean() > 0:
                    vals.append(isi.std() / isi.mean())
        return float(np.mean(vals)) if vals else np.nan

    cv_hi, cv_lo = mean_cv(u >= hi_th), mean_cv(u <= lo_th)
    return {"cv_high": cv_hi, "cv_low": cv_lo, "delta": cv_hi - cv_lo}
