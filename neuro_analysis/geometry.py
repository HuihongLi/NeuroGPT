"""M2 — population geometry / state-space (project idea #1).

Same code runs on voltage V, spikes S, and current I so the geometry can be compared
directly (H1: V is smoother and higher-dimensional than S).
"""

import numpy as np
from sklearn.decomposition import PCA


def participation_ratio(X):
    """Intrinsic dimensionality PR = (Σλ)² / Σλ² from the covariance eigenspectrum.
    The primary H1 metric — higher = activity spread over more dimensions."""
    Xc = X - X.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (np.square(lam).sum() + 1e-12))


def pca_fit(X, n_components=10):
    """Fit PCA; return (fitted PCA, explained_variance_ratio)."""
    p = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
    p.fit(X - X.mean(0, keepdims=True))
    return p, p.explained_variance_ratio_


def project_trajectory(traj, pca):
    """Project a single (T x n_embd) trajectory into a fitted PCA space."""
    return pca.transform(traj - pca.mean_)


def trajectory_smoothness(traj):
    """Path length and mean curvature of a (T x d) trajectory (H1/H3 smoothness).
    Voltage trajectories should be smoother (lower curvature) than spike trajectories."""
    d = np.diff(traj, axis=0)
    seg = np.linalg.norm(d, axis=1)
    path_len = float(seg.sum())
    if len(d) < 2:
        return {"path_length": path_len, "mean_curvature": np.nan}
    cos = (d[:-1] * d[1:]).sum(1) / (seg[:-1] * seg[1:] + 1e-12)
    curv = np.arccos(np.clip(cos, -1, 1))
    return {"path_length": path_len, "mean_curvature": float(curv.mean())}


def jpca(X, k=6):
    """Minimal jPCA: fit the best skew-symmetric dynamics on the top-k PCs and return
    the leading rotational plane's eigenfrequency (rotational structure strength).

    TODO (M2/M4): full Churchland-style jPCA (state + state-derivative regression,
    projection onto the top skew-symmetric eigenplane). Stub returns the top PCA plane.
    """
    p, evr = pca_fit(X, n_components=k)
    raise NotImplementedError("Implement skew-symmetric dynamics fit; PCA scaffold ready.")


def depth_curve(resp, representations, signal, branch, n_layer, metric=participation_ratio):
    """Sweep a scalar geometry metric across all layers for one (signal, branch).
    Returns (n_layer,) array — the core H1 depth curve, computed once per signal to overlay."""
    from .representations import condition_matrix
    from . import track
    out = np.full(n_layer, np.nan)
    for l in track(range(n_layer), desc=f"geometry {signal}/{branch}", leave=False):
        X, _ = condition_matrix(resp, signal, branch, l)
        if len(X) > 2:
            out[l] = metric(X)
    return out
