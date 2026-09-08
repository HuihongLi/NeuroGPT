"""Synthetic LFP / EEG from SpikeGPT and its spectral analysis.

In vivo, the local field potential is well approximated by the summed synaptic (transmembrane)
currents of a local population, and EEG by a further spatially-smoothed sum. Here we build the
analogues directly from the captured signals:

  LFP_layer[t]  = mean over the layer's neurons of the pre-neuron CURRENT  I  (default)
  EEG_global[t] = sum over layers of LFP_layer  (a whole-network, spatially-pooled proxy)

IMPORTANT — units. "Time" is the token step, so the sampling rate is 1 sample/token and all
frequencies are in **cycles per token** (Nyquist = 0.5), NOT Hz. Sequences are short, so this is
coarse; we report robust, low-resolution summaries (aperiodic 1/f slope, low/high band-power
fraction, intrinsic timescale, cross-layer synchrony) rather than sharp spectral peaks.

Everything is computed per condition so driven vs spontaneous spectra can be contrasted — e.g.
does spontaneous rollout have a steeper 1/f slope / longer timescale (slower, more low-frequency
drift from looping) than driven activity?
"""

import warnings
import numpy as np


# --------------------------------------------------------------------- synthesis
def population_lfp(resp, signal="current", branch="attn"):
    """(n_stim, T, n_layer) population-mean LFP proxy (mean over neurons per token/layer).
    Padded token steps have no valid neurons -> NaN (expected; callers drop them)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN padded steps
        return np.nanmean(resp[signal][branch], axis=-1)           # (n_stim, T, n_layer)


def global_eeg(resp, signal="current", branch="attn"):
    """(n_stim, T) whole-network EEG proxy: sum of the per-layer LFPs."""
    return np.nansum(population_lfp(resp, signal, branch), axis=-1)


def _valid(x):
    x = np.asarray(x, float)
    return x[~np.isnan(x)]


# --------------------------------------------------------------------- spectra (1D)
def psd(x):
    """Hann-windowed periodogram of a 1-D series. Returns (freqs [cycles/token], power).
    DC removed by mean-subtraction; freqs[0]=0 kept but excluded by callers."""
    x = _valid(x)
    n = len(x)
    if n < 4:
        return np.array([]), np.array([])
    x = x - x.mean()
    X = np.fft.rfft(x * np.hanning(n))
    return np.fft.rfftfreq(n, d=1.0), (np.abs(X) ** 2)


def spectral_slope(x):
    """Aperiodic 1/f exponent: slope of log10(power) vs log10(freq) over positive freqs.
    More negative = steeper (more low-frequency dominance)."""
    f, p = psd(x)
    m = f > 0
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log10(f[m]), np.log10(p[m] + 1e-20), 1)[0])


def band_power_fraction(x, split=0.15):
    """Fraction of spectral power below `split` cycles/token (low-frequency share)."""
    f, p = psd(x)
    m = f > 0
    if m.sum() < 2:
        return np.nan
    p = p[m]; f = f[m]
    return float(p[f < split].sum() / (p.sum() + 1e-20))


def intrinsic_timescale(x):
    """Lag-1 autocorrelation of the series — a simple intrinsic-timescale proxy
    (higher = slower, more temporally correlated activity)."""
    x = _valid(x)
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    denom = (x * x).sum() + 1e-20
    return float((x[:-1] * x[1:]).sum() / denom)


# --------------------------------------------------------------------- per-layer aggregates
def _reduce_by_layer(resp, fn, signal, branch, n_layer):
    lfp = population_lfp(resp, signal, branch)                 # (n_stim, T, n_layer)
    out = np.full(n_layer, np.nan)
    for l in range(n_layer):
        vals = [fn(lfp[s, :, l]) for s in range(lfp.shape[0])]
        vals = [v for v in vals if v == v]                    # drop NaN
        if vals:
            out[l] = float(np.mean(vals))
    return out


def slope_by_layer(resp, signal="current", branch="attn", n_layer=18):
    return _reduce_by_layer(resp, spectral_slope, signal, branch, n_layer)


def lowband_by_layer(resp, signal="current", branch="attn", n_layer=18):
    return _reduce_by_layer(resp, band_power_fraction, signal, branch, n_layer)


def timescale_by_layer(resp, signal="current", branch="attn", n_layer=18):
    return _reduce_by_layer(resp, intrinsic_timescale, signal, branch, n_layer)


def mean_psd(resp, signal="current", branch="attn", layer=None, nbins=24):
    """Mean PSD over sequences (and layers if layer=None), interpolated onto a common
    frequency grid in (0, 0.5] cycles/token so variable-length sequences can be averaged.
    Returns (freq_grid, mean_log_power)."""
    lfp = population_lfp(resp, signal, branch)
    layers = range(lfp.shape[2]) if layer is None else [layer]
    grid = np.linspace(1.0 / nbins * 0.5, 0.5, nbins)
    curves = []
    for s in range(lfp.shape[0]):
        for l in layers:
            f, p = psd(lfp[s, :, l])
            m = f > 0
            if m.sum() < 3:
                continue
            lp = np.interp(grid, f[m], np.log10(p[m] + 1e-20))
            curves.append(lp)
    if not curves:
        return grid, np.full(nbins, np.nan)
    return grid, np.mean(curves, axis=0)


def cross_layer_synchrony(resp, signal="current", branch="attn"):
    """Mean pairwise Pearson correlation between layer LFPs (a cross-'channel' synchrony
    index, analogous to inter-electrode EEG coherence), averaged over sequences."""
    lfp = population_lfp(resp, signal, branch)                # (n_stim, T, n_layer)
    vals = []
    for s in range(lfp.shape[0]):
        M = lfp[s]                                            # (T, n_layer)
        keep = ~np.isnan(M).any(1)
        M = M[keep]
        if M.shape[0] < 3 or M.shape[1] < 2:
            continue
        C = np.corrcoef(M.T)
        iu = np.triu_indices(C.shape[0], k=1)
        vals.append(np.nanmean(C[iu]))
    return float(np.nanmean(vals)) if vals else np.nan
