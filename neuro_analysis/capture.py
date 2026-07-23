"""M0 — capture harness.

Records, for every block and BOTH LIF branches (mem1=attn, mem2=ffn), the emitted
spikes S (0/1) and the pre-neuron current I on the RWKV_RNN inference path, then
reconstructs the membrane voltage V from I. Exports one canonical archive that every
downstream analysis reads.

Design notes (see PROJECT_PLAN.md "Critical correctness constraints"):
  * We reset between stimuli WITHOUT dropping hooks: node.reset() on every LIF node +
    a zeroed recurrent `state`. Passing state=None would recreate the nodes and lose hooks.
  * Voltage is reconstructed from the node's own LIF params (not hard-coded) so the
    export stays correct if a checkpoint uses non-default tau/v_th/v_reset.
"""

import numpy as np
import torch


# ----------------------------------------------------------------------------- hooks
def attach_hooks(mem1, mem2):
    """Register forward hooks on both LIF branches.

    Returns a `stores` dict:
        stores["spikes"]["attn"][block] -> list of [n_embd] arrays (one per token)
        stores["current"]["attn"][block] -> ...
        (same for "ffn")
    Unlike src/spike_vis.attach_hooks, current is recorded for the FFN branch too, so
    voltage can be reconstructed for both branches.
    """
    n = len(mem1)
    stores = {sig: {"attn": {i: [] for i in range(n)},
                    "ffn":  {i: [] for i in range(n)}}
              for sig in ("spikes", "current")}
    handles = []

    def mk(branch, i):
        def hook(module, inputs, output):
            stores["spikes"][branch][i].append(
                output.detach().cpu().float().numpy().reshape(-1))
            stores["current"][branch][i].append(
                inputs[0].detach().cpu().float().numpy().reshape(-1))
        return hook

    for i in range(n):
        handles.append(mem1[i].register_forward_hook(mk("attn", i)))
        handles.append(mem2[i].register_forward_hook(mk("ffn", i)))
    return stores, handles


def clear_stores(stores):
    for sig in stores:
        for br in stores[sig]:
            for i in stores[sig][br]:
                stores[sig][br][i].clear()


def lif_params(node):
    """Read (tau, v_th, v_reset, decay_input, hard_reset) from a spikingjelly LIFNode.
    Falls back to the model_run defaults if an attribute is missing."""
    tau = float(getattr(node, "tau", 2.0))
    v_th = float(getattr(node, "v_threshold", 1.0))
    v_reset = getattr(node, "v_reset", 0.0)
    v_reset = 0.0 if v_reset is None else float(v_reset)
    decay_input = bool(getattr(node, "decay_input", True))
    return tau, v_th, v_reset, decay_input


def reconstruct_V(cur, tau=2.0, v_th=1.0, v_reset=0.0, decay_input=True):
    """Pre-fire LIF membrane potential V[t] from input current, shape [T, N] -> [T, N].

    Exact for the LIFNode hard-reset dynamics used in model_run.py. Mirrors
    src/spike_vis._reconstruct_V but parameterized by the node's actual config.
    """
    cur = np.asarray(cur)
    T, N = cur.shape
    V = np.zeros((T, N))
    v = np.full(N, float(v_reset))
    for t in range(T):
        if decay_input:
            h = v + (cur[t] - (v - v_reset)) / tau
        else:
            h = v - (v - v_reset) / tau + cur[t]
        V[t] = h
        v = np.where(h >= v_th, v_reset, h)
    return V


# ------------------------------------------------------------------- stimulus driver
def fresh_state(template):
    """A properly initialized recurrent state matching `template`'s shape/device/dtype.

    NOT zeros: row 5*i+4 of each block holds the WKV running-max and must start at -1e30
    (see model_run.forward's state init). Zeroing it corrupts the first tokens' time-mix.
    """
    state = torch.zeros_like(template)
    n_layer = state.shape[0] // 5
    for i in range(n_layer):
        state[5 * i + 4] -= 1e30
    return state


def reset_state(state, mem1, mem2):
    """Reset recurrent + membrane state between stimuli WITHOUT recreating LIF nodes.

    Resets every LIF node in place (hooks survive) and returns a freshly initialized
    recurrent state. Passing state=None to model.forward instead would rebuild the nodes
    and silently drop the attached hooks — hence we reset by hand.
    """
    for node in list(mem1) + list(mem2):
        if hasattr(node, "reset"):
            node.reset()
    return fresh_state(state)


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _pull(stores, per, n_layer, vparams):
    """Append one just-run sequence's activity (spikes, current, reconstructed voltage) to
    the per-condition accumulators. stores[sig][br][l] holds L arrays of [n_embd]."""
    for br in ("attn", "ffn"):
        for sig in ("spikes", "current"):
            arr = np.stack([np.stack(stores[sig][br][l], 0) for l in range(n_layer)], 1)
            per[sig][br].append(arr)                           # [L, n_layer, n_embd]
        tau, vth, vr, di = vparams[br]
        cur = per["current"][br][-1]
        V = np.stack([reconstruct_V(cur[:, l, :], tau, vth, vr, di)
                      for l in range(n_layer)], 1)
        per["voltage"][br].append(V)


def _assemble(per, n_stim, T, n_layer, n_embd):
    """Pad the per-sequence lists into (n_stim, T, n_layer, n_embd) arrays, NaN in the tail."""
    resp = {}
    for sig in per:
        resp[sig] = {}
        for br in per[sig]:
            out = np.full((n_stim, T, n_layer, n_embd), np.nan, dtype=np.float32)
            for s, a in enumerate(per[sig][br]):
                out[s, : a.shape[0]] = a
            resp[sig][br] = out
    return resp


def capture(model, stimuli_tokens, ctx_len, n_layer):
    """Run every stimulus with reset-in-between and return (resp, lengths, aux).

    stimuli_tokens : list[list[int]]  (already tokenized)
    Returns:
        resp[signal][branch] : ndarray (n_stim, T, n_layer, n_embd), T=max len, NaN-padded
                               signal ∈ {spikes, current, voltage}, branch ∈ {attn, ffn}
        lengths              : (n_stim,) int array of true token counts
        aux                  : per-stim lists used to build H2/H4 labels:
                                 surprisal[s] (L,), entropy[s] (L,), embedding[s] (L, n_embd)
    """
    # 1. prime once to create mem1/mem2, then attach hooks (priming call is not recorded)
    _out, state, mem1, mem2 = model.forward(stimuli_tokens[0][:1], None, None, None)
    stores, _handles = attach_hooks(mem1, mem2)
    n_embd = state.shape[1]
    lengths = np.array([len(t) for t in stimuli_tokens], dtype=int)
    T = int(lengths.max())
    p_attn = lif_params(mem1[0])
    p_ffn = lif_params(mem2[0])
    vparams = {"attn": p_attn, "ffn": p_ffn}

    per = {sig: {"attn": [], "ffn": []} for sig in ("spikes", "current", "voltage")}
    aux = {"surprisal": [], "entropy": [], "embedding": []}

    from . import track
    for toks in track(stimuli_tokens, desc="capture (stimuli)"):
        state = reset_state(state, mem1, mem2)
        clear_stores(stores)
        logit_seq = []
        for i in range(len(toks)):
            x = toks[: i + 1][-ctx_len:]
            out, state, mem1, mem2 = model.forward(x, state, mem1, mem2)
            logit_seq.append(out.detach().cpu().float().numpy().reshape(-1))

        _pull(stores, per, n_layer, vparams)

        # H2/H4 label sources from the model's own logits + input embeddings
        P = np.stack([_softmax(z) for z in logit_seq], 0)      # [L, vocab]
        L = len(toks)
        entropy = -(P * np.log(P + 1e-12)).sum(1)              # uncertainty about next token
        surpr = np.full(L, np.nan)
        for t in range(1, L):
            surpr[t] = -np.log(P[t - 1, toks[t]] + 1e-12)      # -log p(observed token_t)
        emb = model.w.emb.weight.detach().cpu().float().numpy()[toks]   # [L, n_embd]
        aux["surprisal"].append(surpr)
        aux["entropy"].append(entropy)
        aux["embedding"].append(emb)

    resp = _assemble(per, len(stimuli_tokens), T, n_layer, n_embd)
    return resp, lengths, aux


def capture_rollout(model, tokenizer, prompt_tokens, n_generate, ctx_len, n_layer,
                    temperature=0.8, top_p=0.7, top_p_newline=0.9):
    """SPONTANEOUS condition: prime on a short prompt, then let the model generate its own
    continuation and record internal activity during the SELF-GENERATED tokens only.

    Artificial analogue of spontaneous (vs stimulus-evoked) neural activity: the network runs
    on its own predictions instead of ground-truth input. The prompt builds state but its
    activity is discarded (clear_stores) so only rollout tokens are kept.

    Returns (resp, lengths, aux, gen_tokens); T = n_generate (all sequences equal length).
    aux surprisal is the SELF-surprisal −log p(sampled token) under the model's own dist.
    """
    _out, state, mem1, mem2 = model.forward(prompt_tokens[0][:1], None, None, None)
    stores, _handles = attach_hooks(mem1, mem2)
    n_embd = state.shape[1]
    vparams = {"attn": lif_params(mem1[0]), "ffn": lif_params(mem2[0])}
    per = {sig: {"attn": [], "ffn": []} for sig in ("spikes", "current", "voltage")}
    aux = {"surprisal": [], "entropy": [], "embedding": []}
    gen_tokens = []
    embw = model.w.emb.weight.detach().cpu().float().numpy()

    from . import track
    for prompt in track(prompt_tokens, desc="rollout (prompts)"):
        state = reset_state(state, mem1, mem2)
        ctx = list(prompt)
        out = None
        for i in range(len(ctx)):                              # prime on the prompt
            out, state, mem1, mem2 = model.forward(ctx[: i + 1][-ctx_len:], state, mem1, mem2)
        clear_stores(stores)                                   # discard prompt activity

        toks, surpr, entropy = [], [], []
        for _ in range(n_generate):
            out_np = out.detach().cpu().float().numpy().reshape(-1)
            out_np[0] = -1e30                                  # disable <|endoftext|> (pile)
            p = _softmax(out_np)
            nxt = int(tokenizer.sample_logits(out, ctx, ctx_len, temperature=temperature,
                                              top_p_usual=top_p, top_p_newline=top_p_newline))
            surpr.append(-np.log(p[nxt] + 1e-12))              # self-surprisal of the sample
            entropy.append(-(p * np.log(p + 1e-12)).sum())     # uncertainty that produced it
            ctx.append(nxt); toks.append(nxt)
            out, state, mem1, mem2 = model.forward(ctx[-ctx_len:], state, mem1, mem2)  # records nxt

        _pull(stores, per, n_layer, vparams)
        aux["surprisal"].append(np.array(surpr))
        aux["entropy"].append(np.array(entropy))
        aux["embedding"].append(embw[toks])
        gen_tokens.append(toks)

    resp = _assemble(per, len(prompt_tokens), n_generate, n_layer, n_embd)
    lengths = np.full(len(prompt_tokens), n_generate, dtype=int)
    return resp, lengths, aux, gen_tokens


def save_archive(path, resp, labels, meta):
    """Persist the canonical archive as a compressed .npz (see PROJECT_PLAN.md data model)."""
    flat = {f"resp/{sig}/{br}": resp[sig][br] for sig in resp for br in resp[sig]}
    # labels are ragged (one list per stimulus) -> store as object arrays
    flat.update({f"label/{k}": np.array(v, dtype=object) for k, v in labels.items()})
    np.savez_compressed(path, meta=np.array(meta, dtype=object), **flat)


def load_archive(path):
    """Inverse of save_archive -> (resp, labels, meta)."""
    z = np.load(path, allow_pickle=True)
    resp, labels = {}, {}
    for key in z.files:
        if key.startswith("resp/"):
            _, sig, br = key.split("/")
            resp.setdefault(sig, {})[br] = z[key]
        elif key.startswith("label/"):
            labels[key.split("/", 1)[1]] = z[key]
    return resp, labels, z["meta"].item()
