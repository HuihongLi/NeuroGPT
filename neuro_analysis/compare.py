"""M6 — orchestrate the full H1–H4 voltage-vs-spike battery and emit figures + a table.

Reads a captured archive and runs:
  H1  geometry     : participation ratio (V/S) + trajectory smoothness (V/S) vs depth
  H2  identity     : V-only / S-only decoding + V−S gap vs depth
  H2  salience     : spike↔surprisal/entropy correlation + spike-triggered contrast vs depth
  H3  joint        : V-only / S-only / joint [V‖S] decoding + joint gain vs depth
  H3  memory       : lagged decoding (V vs S) at a representative layer
  H4  irregularity : CV-ISI vs depth + high/low-condition CV-ISI contrast

Every stage is toggleable via `run(..., stages=...)`; each writes its own PNG and columns
into summary.csv. Uses the standardized fast decoder, so the whole battery is minutes on CPU.
"""

import os
import numpy as np

from . import geometry, decoding, salience, isi, plotting, track
from .capture import load_archive

ALL_STAGES = ("geometry", "identity", "salience", "joint", "memory", "isi", "lfp")


def run(archive_path, out_dir, branch="attn", target="is_content",
        uncertainty_key="surprisal", stages=ALL_STAGES, max_lag=8):
    os.makedirs(out_dir, exist_ok=True)
    resp, labels, meta = load_archive(archive_path)
    n_layer = int(meta["n_layer"])
    cols = {"layer": np.arange(n_layer)}

    def p(name):  # figure path
        return os.path.join(out_dir, name)

    if "geometry" in stages:
        pr_v = geometry.depth_curve(resp, None, "voltage", branch, n_layer)
        pr_s = geometry.depth_curve(resp, None, "spikes", branch, n_layer)
        sm_v = geometry.smoothness_depth(resp, "voltage", branch, n_layer)
        sm_s = geometry.smoothness_depth(resp, "spikes", branch, n_layer)
        cols.update(pr_voltage=pr_v, pr_spikes=pr_s, curv_voltage=sm_v, curv_spikes=sm_s)
        plotting.line_vs_depth({"voltage": pr_v, "spikes": pr_s}, p("H1_dimensionality.png"),
                               "Participation ratio", "H1: dimensionality V vs S")
        plotting.line_vs_depth({"voltage": sm_v, "spikes": sm_s}, p("H1_smoothness.png"),
                               "Mean trajectory curvature", "H1: smoothness (lower = smoother)")

    if "identity" in stages:
        gap = decoding.voltage_spike_gap(resp, labels, branch, n_layer, target)
        null = decoding.depth_accuracy(resp, labels, "voltage", branch, n_layer, target,
                                       shuffle_null=True)   # shuffled-label chance baseline
        cols.update(acc_voltage=gap["voltage"], acc_spikes=gap["spikes"], gap=gap["gap"],
                    acc_null=null)
        plotting.decoding_gap(gap, p("H2_decoding_gap.png"), target=target, null=null)

    if "salience" in stages:
        r = salience.spike_uncertainty_corr(resp, labels, branch, n_layer, key=uncertainty_key)
        d = np.array([salience.spike_triggered_contrast(
                          resp, labels, branch, l, key=uncertainty_key)["cohens_d"]
                      for l in track(range(n_layer), desc="salience contrast", leave=False)])
        cols.update(spike_unc_r=r, spike_trig_d=d)
        plotting.bar_vs_depth(r, p("H2_salience_corr.png"),
                              f"corr(spike count, {uncertainty_key})",
                              f"H2: spikes track {uncertainty_key}")
        plotting.bar_vs_depth(d, p("H2_salience_contrast.png"),
                              "Cohen's d (high−low)", "H2: spike-triggered contrast")

    if "joint" in stages:
        j = decoding.joint_depth(resp, labels, branch, n_layer, target)
        cols.update(acc_joint=j["joint"], joint_gain=j["gain"])
        plotting.line_vs_depth({"voltage": j["voltage"], "spikes": j["spikes"], "joint": j["joint"]},
                               p("H3_joint_decode.png"), f"Decode {target} (bal. acc.)",
                               "H3: joint [V‖S] vs single signals")

    if "memory" in stages:
        L = n_layer // 2
        curves = {sig: decoding.lagged_decode(resp, labels, sig, branch, L, target, max_lag=max_lag)
                  for sig in track(("voltage", "spikes"), desc="memory (lagged)", leave=False)}
        plotting.lag_curves(curves, p("H3_memory_lagged.png"))

    if "isi" in stages:
        cv = isi.cv_isi_by_layer(resp, branch, n_layer)
        cols["cv_isi"] = cv
        plotting.bar_vs_depth(cv, p("H4_cv_isi.png"), "mean CV-ISI",
                              "H4: firing irregularity vs depth", color=plotting.C["green"])
        # high vs low uncertainty contrast at a representative layer
        cond = isi.cv_isi_by_condition(resp, labels, branch, n_layer // 2,
                                       condition_key="entropy")
        print(f"H4 CV-ISI (entropy hi vs lo @ L{n_layer//2}): "
              f"{cond['cv_high']:.3f} vs {cond['cv_low']:.3f} (Δ={cond['delta']:+.3f})")

    if "lfp" in stages:
        from . import lfp
        slope = lfp.slope_by_layer(resp, "current", branch, n_layer)
        low = lfp.lowband_by_layer(resp, "current", branch, n_layer)
        ts = lfp.timescale_by_layer(resp, "current", branch, n_layer)
        cols.update(lfp_slope=slope, lfp_lowband=low, lfp_timescale=ts)
        f, lp = lfp.mean_psd(resp, "current", branch)
        plotting.psd_overlay({"LFP": (f, lp)}, p("H8_lfp_psd.png"),
                             "synthetic LFP spectrum (all layers)")
        plotting.line_vs_depth({"1/f slope": slope}, p("H8_lfp_slope.png"),
                               "aperiodic slope", "LFP spectral slope vs depth")
        print(f"LFP cross-layer synchrony: {lfp.cross_layer_synchrony(resp, 'current', branch):.3f}")

    _write_table(cols, os.path.join(out_dir, "summary.csv"))
    print(f"wrote {len([f for f in os.listdir(out_dir) if f.endswith('.png')])} figures "
          f"+ summary.csv to {out_dir}")
    return cols


def run_conditions(driven_archive, spont_archive, out_dir, branch="attn", k=10):
    """Driven-vs-spontaneous contrast: firing rate, |V|, CV-ISI, dimensionality, and subspace
    alignment per layer. Writes contrast.csv + one figure per metric."""
    from . import conditions, rollout_dynamics as dyn
    os.makedirs(out_dir, exist_ok=True)
    rd, ld, md = load_archive(driven_archive)
    rs, ls, _ = load_archive(spont_archive)
    n_layer = int(md["n_layer"])
    c = conditions.contrast(rd, rs, branch, n_layer, k=k)
    xc = conditions.cross_decode_curves(rd, ld, rs, ls, branch, n_layer,
                                        target="is_content", signal="voltage")
    c["cross_d2s"] = xc["d2s"]; c["cross_s2d"] = xc["s2d"]
    c.update(dyn.dynamics_contrast(rd, rs, branch, n_layer))       # speed / velcorr / recurrence

    def p(name):
        return os.path.join(out_dir, name)

    plotting.line_vs_depth({"driven": c["rate_driven"], "spont": c["rate_spont"]},
                           p("C_firing_rate.png"), "mean firing rate", "driven vs spontaneous")
    plotting.line_vs_depth({"driven": c["absV_driven"], "spont": c["absV_spont"]},
                           p("C_abs_voltage.png"), "mean |membrane V|", "driven vs spontaneous")
    plotting.line_vs_depth({"driven": c["cv_driven"], "spont": c["cv_spont"]},
                           p("C_cv_isi.png"), "mean CV-ISI", "driven vs spontaneous")
    plotting.line_vs_depth({"driven": c["pr_driven"], "spont": c["pr_spont"]},
                           p("C_dimensionality.png"), "participation ratio (V)", "driven vs spontaneous")
    plotting.line_vs_depth({"voltage": c["align_v"], "spikes": c["align_s"]},
                           p("C_subspace_alignment.png"),
                           "alignment (cross / self var, ~1 = shared)",
                           "spontaneous variance in the driven subspace", hline=1.0)
    plotting.line_vs_depth({"driven→spont": c["cross_d2s"], "spont→driven": c["cross_s2d"]},
                           p("C_cross_decode.png"), "transfer bal. acc. (voltage)",
                           "cross-condition decoding of is_content", hline=0.5)
    # rollout dynamics (primary axis): why spontaneous firing differs
    plotting.line_vs_depth({"driven": c["speed_driven"], "spont": c["speed_spont"]},
                           p("D_speed.png"), "mean state speed ‖ΔV‖", "trajectory speed")
    plotting.line_vs_depth({"driven": c["velcorr_driven"], "spont": c["velcorr_spont"]},
                           p("D_velocity_autocorr.png"), "lag-1 velocity cosine",
                           "trajectory directedness", hline=0.0)
    plotting.line_vs_depth({"driven": c["recurrence_driven"], "spont": c["recurrence_spont"]},
                           p("D_recurrence.png"), "recurrence rate", "state revisiting (looping)")
    rep_d = dyn.token_repetition(ld["token_id"]); rep_s = dyn.token_repetition(ls["token_id"])
    print(f"token repetition (16-tok window): driven={rep_d:.3f}  spontaneous={rep_s:.3f}")

    # synthetic LFP/EEG spectral contrast (driven vs spontaneous)
    from . import lfp
    c["lfp_slope_driven"] = lfp.slope_by_layer(rd, "current", branch, n_layer)
    c["lfp_slope_spont"] = lfp.slope_by_layer(rs, "current", branch, n_layer)
    fd, ld_ = lfp.mean_psd(rd, "current", branch)
    fs, ls_ = lfp.mean_psd(rs, "current", branch)
    plotting.psd_overlay({"driven": (fd, ld_), "spont": (fs, ls_)}, p("C_lfp_psd.png"),
                         "synthetic LFP spectrum: driven vs spontaneous")
    plotting.line_vs_depth({"driven": c["lfp_slope_driven"], "spont": c["lfp_slope_spont"]},
                           p("C_lfp_slope.png"), "aperiodic 1/f slope", "LFP spectral slope")
    print(f"LFP cross-layer synchrony: driven={lfp.cross_layer_synchrony(rd,'current',branch):.3f}"
          f"  spont={lfp.cross_layer_synchrony(rs,'current',branch):.3f}")
    _write_table(c, os.path.join(out_dir, "contrast.csv"))
    print(f"wrote condition-contrast + dynamics figures + contrast.csv to {out_dir}")
    return c


def _write_table(cols, path):
    names = list(cols)
    data = np.column_stack([np.asarray(cols[k], float) for k in names])
    fmt = ["%d"] + ["%.4f"] * (len(names) - 1)
    np.savetxt(path, data, header=",".join(names), delimiter=",", fmt=fmt, comments="")
