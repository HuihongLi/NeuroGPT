"""M5 — orchestrate the voltage-vs-spike comparison and emit the summary table.

Reads a captured archive and produces the H1/H2/H3 figures + a per-layer summary:
    layer | PR_voltage | PR_spikes | acc_voltage | acc_spikes | gap
"""

import os
import numpy as np

from . import geometry, decoding, plotting
from .capture import load_archive


def summary_table(resp, labels, n_layer, branch="attn", target="is_content", **decode_kw):
    """Assemble the headline per-layer comparison table (structured ndarray)."""
    pr_v = geometry.depth_curve(resp, None, "voltage", branch, n_layer)
    pr_s = geometry.depth_curve(resp, None, "spikes", branch, n_layer)
    gap = decoding.voltage_spike_gap(resp, labels, branch, n_layer, target, **decode_kw)
    rows = np.zeros(n_layer, dtype=[("layer", int), ("pr_voltage", float),
                                    ("pr_spikes", float), ("acc_voltage", float),
                                    ("acc_spikes", float), ("gap", float)])
    rows["layer"] = np.arange(n_layer)
    rows["pr_voltage"], rows["pr_spikes"] = pr_v, pr_s
    rows["acc_voltage"], rows["acc_spikes"], rows["gap"] = \
        gap["voltage"], gap["spikes"], gap["gap"]
    return rows, gap


def run(archive_path, out_dir, branch="attn", target="is_content"):
    os.makedirs(out_dir, exist_ok=True)
    resp, labels, meta = load_archive(archive_path)
    n_layer = int(meta["n_layer"])

    table, gap = summary_table(resp, labels, n_layer, branch=branch, target=target)
    np.savetxt(os.path.join(out_dir, "summary.csv"), table,
               header=",".join(table.dtype.names), delimiter=",",
               fmt=["%d"] + ["%.4f"] * 5, comments="")

    plotting.dimensionality_vs_depth(
        {"voltage": table["pr_voltage"], "spikes": table["pr_spikes"]},
        os.path.join(out_dir, "H1_dimensionality.png"))
    plotting.decoding_gap(gap, os.path.join(out_dir, "H2_decoding_gap.png"), target=target)
    print(f"wrote comparison outputs to {out_dir}")
    return table
