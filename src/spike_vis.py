########################################################################################################
# Spiking-activity visualization for the RWKV_RNN inference path (src/model_run.py).
#
# Usage (see run.py / visualize.py):
#   stores = attach_hooks(mem1, mem2)        # after the LIF neuron lists are created
#   ... run generation, collecting input embeddings per token ...
#   render(*stores, input_emb, out_path="spikegpt_activity.png")
#
# Captures the spikes (0/1) and pre-neuron current of every block's LIF neurons:
#   mem1[i] -> LIF after self-attention (SA) ; mem2[i] -> LIF after channel-mix (FF)
# The membrane potential V[t] is reconstructed exactly from the input current for the
# LIFNode defaults used in model_run.py (tau=2, v_th=1, v_reset=0, decay_input=True, hard reset).
########################################################################################################

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save to file instead of opening a window
import matplotlib.pyplot as plt

# LIFNode defaults (neuron.LIFNode() in model_run.py)
TAU, V_TH, V_RESET = 2.0, 1.0, 0.0

# Tol muted palette (colorblind-safe)
_C = {"blue": "#4477AA", "red": "#CC3311", "green": "#228833",
      "orange": "#EE7733", "purple": "#AA3377", "cyan": "#66CCEE", "gray": "#BBBBBB"}


def attach_hooks(mem1, mem2):
    """Register forward hooks on the LIF neurons and return (spikes_sa, spikes_ff, curr_sa) stores.
    Each store maps block index -> list (one [n_embd] array per forward call)."""
    n = len(mem1)
    spikes_sa = {i: [] for i in range(n)}
    spikes_ff = {i: [] for i in range(n)}
    curr_sa = {i: [] for i in range(n)}

    def mk(spk_store, cur_store, i):
        def hook(module, inputs, output):
            spk_store[i].append(output.detach().cpu().float().numpy().reshape(-1))
            if cur_store is not None:
                cur_store[i].append(inputs[0].detach().cpu().float().numpy().reshape(-1))
        return hook

    for i in range(n):
        mem1[i].register_forward_hook(mk(spikes_sa, curr_sa, i))
        mem2[i].register_forward_hook(mk(spikes_ff, None, i))
    return spikes_sa, spikes_ff, curr_sa


def _reconstruct_V(cur):
    """Pre-fire LIF membrane potential V[t] from the input current, shape [T, N]."""
    T, N = cur.shape
    V = np.zeros((T, N))
    v = np.zeros(N)
    for t in range(T):
        h = v * (1.0 - 1.0 / TAU) + cur[t] / TAU
        V[t] = h
        v = np.where(h >= V_TH, V_RESET, h)
    return V


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render(spikes_sa, spikes_ff, curr_sa, input_emb, out_path="spikegpt_activity.png",
           length=48, vis_layer=None, raster_neurons=40, dpi=150):
    """Build the combined activity figure and save it to out_path. Returns the figure."""
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=list(_C.values()))
    plt.rcParams["font.size"] = 8

    n_layer = len(spikes_sa)
    if vis_layer is None:
        vis_layer = n_layer // 2

    input_emb = np.asarray(input_emb)
    avail = min(len(spikes_sa[0]), len(spikes_ff[0]), len(input_emb))
    L = min(length, avail)
    if L < 1:
        raise ValueError("No spike steps recorded — attach_hooks() before generation.")

    def stack(store, layer):
        return np.stack(store[layer], axis=0)[-L:]     # [L, n_embd]

    rate_sa = np.stack([stack(spikes_sa, l).mean(1) for l in range(n_layer)], 0)   # [layer, L]
    rate_ff = np.stack([stack(spikes_ff, l).mean(1) for l in range(n_layer)], 0)
    V_layers = np.stack([_reconstruct_V(stack(curr_sa, l)).mean(1) for l in range(n_layer)], 0)
    V_vis = _reconstruct_V(stack(curr_sa, vis_layer))    # [L, N]
    S_vis = stack(spikes_sa, vis_layer)                  # [L, N]
    emb = input_emb[-L:]                                 # [L, n_embd]
    all_sa = np.concatenate([stack(spikes_sa, l) for l in range(n_layer)], 1)
    all_ff = np.concatenate([stack(spikes_ff, l) for l in range(n_layer)], 1)
    tok = np.arange(L)

    fig = plt.figure(figsize=(17, 15), constrained_layout=True)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.0, 1.35])
    sf_top.suptitle("SpikeGPT — spiking activity during generation", fontsize=13, fontweight="bold")
    sf_bot.suptitle(f"Per-layer spike raster (attention LIF, first {raster_neurons} channels)", fontsize=11)
    axT = sf_top.subplots(2, 4)

    # (0,0) Input embedding heatmap -> continuous (non-binary) input
    ax = axT[0, 0]
    vmax = np.abs(emb).max()
    im = ax.imshow(emb[:, :100].T, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Token step"); ax.set_ylabel("Input channel (0-100)")
    sf_top.colorbar(im, ax=ax, label="Embedding value")

    # (0,1) Input value histogram -> answers "spikes in the input?"
    ax = axT[0, 1]; _clean(ax)
    ax.hist(emb.flatten(), bins=60, color=_C["blue"])
    ax.axvline(0, color=_C["gray"], lw=0.8)
    ax.set_xlabel("Input embedding value"); ax.set_ylabel("Count")
    ax.text(0.5, 0.92, "Continuous values\n=> NO input spikes", transform=ax.transAxes,
            ha="center", va="top", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec=_C["red"], alpha=0.8))

    # (0,2) / (0,3) Firing rate per layer over tokens
    for ax, rate, name in [(axT[0, 2], rate_sa, "attn"), (axT[0, 3], rate_ff, "ffn")]:
        im = ax.imshow(rate, aspect="auto", cmap="magma", vmin=0)
        ax.set_xlabel("Token step"); ax.set_ylabel("Block index")
        sf_top.colorbar(im, ax=ax, label=f"Firing rate ({name})")

    # (1,0) Mean firing rate per layer
    ax = axT[1, 0]; _clean(ax)
    ax.plot(np.arange(n_layer), rate_sa.mean(1), "-o", ms=3, color=_C["blue"], label="Attention")
    ax.plot(np.arange(n_layer), rate_ff.mean(1), "-s", ms=3, color=_C["orange"], label="FFN")
    ax.set_xlabel("Block index"); ax.set_ylabel("Mean firing rate")
    ax.legend(loc="best", framealpha=0.4, edgecolor="none", fontsize=6)

    # (1,1) Membrane voltage traces for the busiest neurons of vis_layer
    ax = axT[1, 1]; _clean(ax)
    active = np.argsort(S_vis.sum(0))[-4:][::-1]
    for n, c in zip(active, [_C["blue"], _C["orange"], _C["green"], _C["purple"]]):
        ax.plot(tok, V_vis[:, n], color=c, lw=1, label=f"ch {n}")
        sp = np.where(S_vis[:, n] > 0)[0]
        ax.plot(sp, np.full_like(sp, V_TH, dtype=float), "|", color=c, ms=8, mew=1.4)
    ax.axhline(V_TH, color=_C["red"], ls="--", lw=0.8, label="Threshold")
    ax.set_xlabel("Token step"); ax.set_ylabel("Membrane potential V")
    ax.set_title(f"Block {vis_layer}", fontsize=8)
    ax.legend(loc="best", framealpha=0.4, edgecolor="none", fontsize=5, ncol=2)

    # (1,2) Mean membrane potential per layer over tokens
    ax = axT[1, 2]
    im = ax.imshow(V_layers, aspect="auto", cmap="viridis")
    ax.set_xlabel("Token step"); ax.set_ylabel("Block index")
    sf_top.colorbar(im, ax=ax, label="Mean membrane V")

    # (1,3) Active-neuron count per token across the whole network
    ax = axT[1, 3]; _clean(ax)
    ax.plot(tok, all_sa.sum(1), color=_C["blue"], lw=1, label="Attention")
    ax.plot(tok, all_ff.sum(1), color=_C["orange"], lw=1, label="FFN")
    ax.set_xlabel("Token step"); ax.set_ylabel("# spiking neurons (all layers)")
    ax.legend(loc="best", framealpha=0.4, edgecolor="none", fontsize=6)

    # Bottom: spike raster for every layer
    nrows, ncols = 3, (n_layer + 2) // 3
    axB = sf_bot.subplots(nrows, ncols)
    flat = np.array(axB).flatten()
    for l in range(n_layer):
        ax = flat[l]
        sp = stack(spikes_sa, l)[:, :raster_neurons].T   # [N, L]
        for n in range(sp.shape[0]):
            ax.eventplot(tok[sp[n] > 0], lineoffsets=n, colors=_C["blue"],
                         linelengths=0.8, linewidths=0.6)
        ax.set_ylim(-0.5, raster_neurons - 0.5); ax.set_xlim(-0.5, L - 0.5)
        ax.invert_yaxis()
        ax.text(0.02, 0.98, f"L{l}", transform=ax.transAxes, va="top", ha="left",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec=_C["gray"], alpha=0.7))
        ax.set_xlabel("Token" if l >= n_layer - ncols else "")
        ax.set_ylabel("Neuron" if l % ncols == 0 else "")
    for k in range(n_layer, len(flat)):
        flat[k].set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"\nsaved {out_path}")
    print(f"Mean firing rate: attention={all_sa.mean():.3f}, ffn={all_ff.mean():.3f}")
    is_binary = np.isin(emb, [0.0, 1.0]).all()
    print(f"Input embedding binary/spiking? {is_binary} "
          f"(min={emb.min():.3f}, max={emb.max():.3f}) -> network input is CONTINUOUS, not spiking")
    return fig
