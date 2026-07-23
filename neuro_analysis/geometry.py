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


def jpca(trajs, k=6):
    """Minimal jPCA (Churchland et al.): fit the best skew-symmetric dynamics matrix M on
    the top-k PCs of the pooled states, so state-derivative ≈ state · M with M = -Mᵀ.

    trajs : list of (T_i x n_embd) trajectories (temporal order matters — one per stimulus).
    Returns dict(M_skew, eig, rotation_freq); rotation_freq = max |Im(eig)| indexes the
    strength of rotational (oscillatory) population dynamics.
    """
    trajs = [t for t in trajs if len(t) >= 2]
    allX = np.concatenate(trajs, 0)
    p, _ = pca_fit(allX, n_components=k)
    Xs, Xd = [], []
    for tr in trajs:
        z = p.transform(tr - p.mean_)
        Xs.append(z[:-1]); Xd.append(np.diff(z, axis=0))
    Xs, Xd = np.concatenate(Xs, 0), np.concatenate(Xd, 0)
    M, *_ = np.linalg.lstsq(Xs, Xd, rcond=None)     # unconstrained fit
    M_skew = 0.5 * (M - M.T)                          # skew-symmetric (pure rotation) part
    eig = np.linalg.eigvals(M_skew)
    return {"M_skew": M_skew, "eig": eig, "rotation_freq": float(np.max(np.abs(eig.imag)))}


def smoothness_depth(resp, signal, branch, n_layer, k=6):
    """Mean trajectory curvature per layer (H1): project each stimulus trajectory into a
    per-layer PCA(k) space and average curvature over stimuli. Voltage should be smoother
    (lower curvature) than spikes."""
    from .representations import condition_matrix, trajectory
    from . import track
    out = np.full(n_layer, np.nan)
    n_stim = resp[signal][branch].shape[0]
    for l in track(range(n_layer), desc=f"smoothness {signal}/{branch}", leave=False):
        X, _ = condition_matrix(resp, signal, branch, l)
        if len(X) <= k:
            continue
        p, _ = pca_fit(X, n_components=k)
        curvs = []
        for s in range(n_stim):
            tr = trajectory(resp, signal, branch, l, s)
            if len(tr) >= 3:
                curvs.append(trajectory_smoothness(project_trajectory(tr, p))["mean_curvature"])
        if curvs:
            out[l] = float(np.nanmean(curvs))
    return out


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
