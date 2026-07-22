"""M3/M4 — neural decoding (project idea #4).

Linear read-out of stimulus features from the population vector at each
(signal, branch, layer). The headline result is the V-vs-S accuracy gap across depth (H2)
and decoding of lagged tokens as a memory probe (H3).

Cross-validation is GROUPED BY STIMULUS: tokens from the same passage share recurrent
state and are not independent, so a token-level random split leaks. Group folds by stim.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _clf():
    # StandardScaler makes lbfgs converge in a handful of iterations instead of grinding to
    # max_iter on the raw 768-dim activations (the main cost of the depth sweep).
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))


def _labels_for(prov, label_grid):
    """Gather the per-sample label from a (n_stim, T) label grid using provenance."""
    return np.array([label_grid[s][t] for (s, t) in prov])


def decode(resp, labels, signal, branch, layer, target,
           task="classify", n_splits=5, shuffle_null=False, seed=0):
    """Grouped-CV decode of `target` from one condition.

    Returns dict(score_mean, score_std, scores). `shuffle_null=True` permutes labels
    within the CV to give the chance baseline. Provenance groups folds by stimulus.
    """
    from .representations import condition_matrix
    X, prov = condition_matrix(resp, signal, branch, layer)
    y = _labels_for(prov, labels[target])
    groups = prov[:, 0]
    if shuffle_null:
        rng = np.random.default_rng(seed)
        y = rng.permutation(y)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    scores = []
    for tr, te in gkf.split(X, y, groups):
        if task == "classify":
            m = _clf()
            m.fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[te], m.predict(X[te])))
        else:
            m = Ridge(alpha=1.0)
            m.fit(X[tr], y[tr])
            scores.append(r2_score(y[te], m.predict(X[te])))
    scores = np.array(scores)
    return {"score_mean": float(scores.mean()),
            "score_std": float(scores.std()), "scores": scores}


def depth_accuracy(resp, labels, signal, branch, n_layer, target, **kw):
    """Decoding accuracy vs layer for one signal — one curve of the H2 comparison."""
    from . import track
    return np.array([decode(resp, labels, signal, branch, l, target, **kw)["score_mean"]
                     for l in track(range(n_layer),
                                    desc=f"decode {signal}/{branch}:{target}", leave=False)])


def voltage_spike_gap(resp, labels, branch, n_layer, target, **kw):
    """H2: (acc_voltage - acc_spikes) per layer — 'information lost to spiking' curve."""
    acc_v = depth_accuracy(resp, labels, "voltage", branch, n_layer, target, **kw)
    acc_s = depth_accuracy(resp, labels, "spikes", branch, n_layer, target, **kw)
    return {"voltage": acc_v, "spikes": acc_s, "gap": acc_v - acc_s}


def joint_decode(resp, labels, branch, layer, target, signals=("voltage", "spikes"), **kw):
    """H3: decode from concatenated feature sets ([V ‖ S], optionally ‖ I) and compare to
    the best single-signal decoder. Returns per-signal scores, the joint score, and the
    joint-minus-best-single gain.

    Uses condition_matrix per signal (they share the same (stim, token) provenance grid,
    so columns concatenate directly). z-score per signal block before concatenation so the
    binary spikes and graded voltage are on comparable scales for the linear model.
    """
    from .representations import condition_matrix
    blocks, prov0, singles = [], None, {}
    for sig in signals:
        X, prov = condition_matrix(resp, sig, branch, layer)
        prov0 = prov if prov0 is None else prov0
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        blocks.append(Xz)
        singles[sig] = decode(resp, labels, sig, branch, layer, target, **kw)["score_mean"]
    Xj = np.concatenate(blocks, axis=1)
    y = _labels_for(prov0, labels[target])
    groups = prov0[:, 0]
    gkf = GroupKFold(n_splits=min(kw.get("n_splits", 5), len(np.unique(groups))))
    js = [balanced_accuracy_score(
              y[te], _clf().fit(Xj[tr], y[tr]).predict(Xj[te]))
          for tr, te in gkf.split(Xj, y, groups)]
    joint = float(np.mean(js))
    return {"singles": singles, "joint": joint, "gain": joint - max(singles.values())}


def lagged_decode(resp, labels, signal, branch, layer, target, max_lag=8, **kw):
    """H3: decode the token at t-lag from activity at t, for lag=0..max_lag.
    Voltage should retain past-token information over more lags than spikes.

    TODO (M4): build lagged label grids (shift `labels[target]` by lag within stimulus,
    dropping the first `lag` tokens) and call decode() for each; return acc-vs-lag.
    """
    raise NotImplementedError("Build lagged label grids per docstring; decode() is ready.")
