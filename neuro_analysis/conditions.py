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
    """Fraction of `resp_test` variance captured by the top-k PCs fit on `resp_fit`, at one
    layer. Compared to the test set's OWN top-k ceiling, this says whether the two conditions
    share a subspace: cross/self ≈ 1 → aligned; ≪ 1 → the regimes live in different subspaces.
    """
    Xf, _ = condition_matrix(resp_fit, signal, branch, layer)
    Xt, _ = condition_matrix(resp_test, signal, branch, layer)
    if len(Xf) <= k or len(Xt) <= k:
        return np.nan
    kk = min(k, Xf.shape[1])
    pf = PCA(n_components=kk).fit(Xf)                       # centers on Xf mean internally
    pt = PCA(n_components=kk).fit(Xt)
    Xt_c = Xt - Xf.mean(0)
    total = (Xt_c ** 2).sum() + 1e-12
    cross = (pf.transform(Xt) ** 2).sum() / total          # test variance in fit's subspace
    self_ = pt.explained_variance_ratio_.sum()             # test variance in its own subspace
    return float(cross / (self_ + 1e-12))


def alignment_by_layer(resp_driven, resp_spont, signal, branch, n_layer, k=10):
    """Per-layer subspace-alignment index (driven subspace ↦ spontaneous variance)."""
    return np.array([subspace_capture(resp_driven, resp_spont, signal, branch, l, k)
                     for l in track(range(n_layer), desc=f"align {signal}/{branch}", leave=False)])


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
