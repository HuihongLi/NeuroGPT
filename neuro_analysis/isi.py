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
    """H4: contrast CV-ISI during high- vs low-uncertainty spans (ambiguity / prediction
    uncertainty / semantic drift, selected via `condition_key` in labels).

    TODO (M5): segment token time into high/low-`condition_key` spans (quantile split),
    compute per-neuron CV-ISI within each span, and return (cv_high, cv_low, stat). Requires
    a windowing choice — document the window length used.
    """
    raise NotImplementedError("Implement condition-split CV-ISI per docstring.")
