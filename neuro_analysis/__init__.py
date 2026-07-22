"""Neural-data-analysis toolkit applied to SpikeGPT's internal LIF activations.

Two pipelines share one captured response tensor:
  - geometry:  population state-space / dimensionality  (project idea #1)
  - decoding:  linear read-out of stimulus features     (project idea #4)

The organizing axis is a voltage-vs-spike comparison: for every neuron we have the
graded membrane voltage V (continuous) and the emitted spike S (binary), plus the
pre-neuron current I. See PROJECT_PLAN.md.
"""

SIGNALS = ("spikes", "voltage", "current")
BRANCHES = ("attn", "ffn")


def track(iterable, desc=None, total=None, leave=True):
    """Progress bar wrapper — tqdm if installed, otherwise a no-op passthrough."""
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=leave)
    except Exception:
        return iterable
