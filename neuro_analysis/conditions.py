"""Driven vs spontaneous contrast — evoked (teacher-forced on real text) vs self-rollout.

The neuroscience analogue: stimulus-evoked vs spontaneous activity. Here "spontaneous" =
the network running autoregressively on its own sampled tokens (see capture.capture_rollout),
"driven" = teacher-forced on ground-truth text (capture.capture). We ask how the internal
activity regime differs between the two: firing level, irregularity, dimensionality, and
whether the two regimes occupy the SAME population subspace.
"""

import numpy as np
from sklearn.decomposition import PCA

from . import geometry, isi, track
from .representations import condition_matrix


def firing_rate_by_layer(resp, branch, n_layer):
    """Mean spike probability per layer (population + time)."""
    S = resp["spikes"][branch]
    return np.array([np.nanmean(S[:, :, l, :]) for l in range(n_layer)])


def mean_abs_voltage_by_layer(resp, branch, n_layer):
    """Mean |membrane voltage| per layer — subthreshold drive level."""
    V = resp["voltage"][branch]
    return np.array([np.nanmean(np.abs(V[:, :, l, :])) for l in range(n_layer)])


def subspace_capture(resp_fit, resp_test, signal, branch, layer, k=10):
    """Alignment index ∈ [0,1]: fraction of `resp_test` variance captured by the top-k PC
    DIRECTIONS of `resp_fit`, relative to the test set's own top-k ceiling. ≈1 → the two
    conditions share a subspace; ≪1 → they live in different subspaces.

    Both numerator and denominator use the test data centered by ITS OWN mean and only differ
    in the projection subspace, so cross ≤ self and the ratio is bounded by 1 (the earlier
    version mixed centering means and could exceed 1).
    """
    Xf, _ = condition_matrix(resp_fit, signal, branch, layer)
    Xt, _ = condition_matrix(resp_test, signal, branch, layer)
    if len(Xf) <= k or len(Xt) <= k:
        return np.nan
    kk = min(k, Xf.shape[1])
    Vf = PCA(n_components=kk).fit(Xf).components_           # fit directions (mean-free)
    pt = PCA(n_components=kk).fit(Xt)
    Xt_c = Xt - Xt.mean(0)                                  # center test by its OWN mean
    total = (Xt_c ** 2).sum() + 1e-12
    cross = ((Xt_c @ Vf.T) ** 2).sum() / total             # test variance in fit directions
    self_ = pt.explained_variance_ratio_.sum()             # test variance in its own top-k
    return float(cross / (self_ + 1e-12))


def alignment_by_layer(resp_driven, resp_spont, signal, branch, n_layer, k=10):
    """Per-layer subspace-alignment index (driven subspace ↦ spontaneous variance)."""
    return np.array([subspace_capture(resp_driven, resp_spont, signal, branch, l, k)
                     for l in track(range(n_layer), desc=f"align {signal}/{branch}", leave=False)])


def cross_decode(resp_tr, lab_tr, resp_te, lab_te, signal, branch, layer, target):
    """Train a decoder in one regime, test it in the other (no CV — full train set, full
    test set). Measures whether the *same* linguistic code is used in both regimes."""
    from .representations import condition_matrix
    from .decoding import _labels_for, _clf
    from sklearn.metrics import balanced_accuracy_score
    Xtr, ptr = condition_matrix(resp_tr, signal, branch, layer)
    Xte, pte = condition_matrix(resp_te, signal, branch, layer)
    ytr, yte = _labels_for(ptr, lab_tr[target]), _labels_for(pte, lab_te[target])
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return np.nan
    return float(balanced_accuracy_score(yte, _clf().fit(Xtr, ytr).predict(Xte)))


def cross_decode_curves(resp_d, lab_d, resp_s, lab_s, branch, n_layer,
                        target="is_content", signal="voltage"):
    """Per-layer cross-condition transfer for one signal: driven→spont and spont→driven.
    High transfer ⇒ a shared code; a drop vs within-condition ⇒ regime-specific representation."""
    d2s = np.array([cross_decode(resp_d, lab_d, resp_s, lab_s, signal, branch, l, target)
                    for l in track(range(n_layer), desc="cross d→s", leave=False)])
    s2d = np.array([cross_decode(resp_s, lab_s, resp_d, lab_d, signal, branch, l, target)
                    for l in track(range(n_layer), desc="cross s→d", leave=False)])
    return {"d2s": d2s, "s2d": s2d}


def contrast(resp_driven, resp_spont, branch, n_layer, k=10):
    """Assemble the full per-layer driven-vs-spontaneous comparison table (dict of arrays)."""
    out = {"layer": np.arange(n_layer)}
    out["rate_driven"] = firing_rate_by_layer(resp_driven, branch, n_layer)
    out["rate_spont"] = firing_rate_by_layer(resp_spont, branch, n_layer)
    out["absV_driven"] = mean_abs_voltage_by_layer(resp_driven, branch, n_layer)
    out["absV_spont"] = mean_abs_voltage_by_layer(resp_spont, branch, n_layer)
    out["cv_driven"] = isi.cv_isi_by_layer(resp_driven, branch, n_layer)
    out["cv_spont"] = isi.cv_isi_by_layer(resp_spont, branch, n_layer)
    out["pr_driven"] = geometry.depth_curve(resp_driven, None, "voltage", branch, n_layer)
    out["pr_spont"] = geometry.depth_curve(resp_spont, None, "voltage", branch, n_layer)
    out["align_v"] = alignment_by_layer(resp_driven, resp_spont, "voltage", branch, n_layer, k)
    out["align_s"] = alignment_by_layer(resp_driven, resp_spont, "spikes", branch, n_layer, k)
    return out
