# Project Plan — Population geometry & decoding of SpikeGPT, with a voltage-vs-spike comparison

**Goal.** Treat SpikeGPT's internal LIF activations as a neural recording and apply two
standard neural-data-analysis pipelines — **(1) population geometry / state-space** and
**(4) neural decoding** — to characterize how information is represented across the 18 layers
and across token time. The *organizing question* threaded through both pipelines is:

> **What does binarization cost?** For every layer/branch/token we have three co-registered
> signals from the *same* neurons — the graded pre-neuron current `I`, the reconstructed
> membrane voltage `V` (continuous), and the emitted spike `S ∈ {0,1}`. Comparing the geometry
> and decodability of `V` vs `S` (vs `I`) quantifies how much structure the spike-generating
> nonlinearity discards vs. preserves.

This is the systems-neuroscience version of the rate-vs-membrane-potential coding debate, run on
an artificial spiking network where we have ground-truth access to all three signals.

## Overview

This project will use computational-neuroscience methods to examine how SpikeGPT's native spike
trains and membrane potentials encode linguistic information. We hypothesize that (H1) membrane
potentials retain graded contextual information that is lost in binary spikes, (H2) spikes
preferentially mark salient events such as surprising or decision-relevant tokens, (H3) combining
spikes and voltages will decode linguistic variables better than either signal alone, and (H4) ISI
and ISI-CV will reveal condition- and layer-dependent changes in firing irregularity, particularly
during ambiguity, prediction uncertainty, and semantic drift. The goal is to determine **when
spike-only analysis is sufficient and when subthreshold voltage dynamics provide essential
additional information.**

---

## Scientific hypotheses

- **H1 — voltage retains graded context lost in spikes.** Membrane potentials carry graded
  contextual information discarded by binarization. *Tests:* higher participation ratio /
  intrinsic dimensionality and higher linear decodability of linguistic variables (token id, POS,
  position) from `V` than from `S` at matched layers; the per-layer **V−S gap** is the
  "information lost to spiking" curve, expected to widen where firing is sparse (deeper layers).
  Trajectories in `V` are smoother (lower curvature) and, because the WKV time-mix integrates
  history, retain *past-token* information over more lags than `S`.
- **H2 — spikes preferentially mark salient events.** Spiking is elevated at surprising or
  decision-relevant tokens. *Tests:* correlate per-token population spike count (and per-neuron
  spike probability) with token **surprisal** `−log p(token)` and next-token predictive
  **entropy** from the model's own logits; spike-triggered analysis of high- vs low-surprisal
  tokens; test whether spikes (vs voltage) are the signal that better *flags* these events even
  where they carry less identity information.
- **H3 — spikes + voltage decode better than either alone.** A joint `[V ‖ S]` (and `[V ‖ S ‖ I]`)
  decoder beats the best single-signal decoder. *Tests:* compare decoding accuracy of
  voltage-only, spike-only, and concatenated features per layer; quantify the super-additive gain
  and where in depth it appears. Establishes when spike-only analysis is *sufficient*.
- **H4 — ISI / ISI-CV reveal condition- and layer-dependent irregularity.** Inter-spike-interval
  statistics change with linguistic condition. *Tests:* per-neuron/per-layer ISI distributions and
  **CV of ISI** (irregularity) computed over token time, compared across conditions —
  **ambiguity, prediction uncertainty (high entropy), and semantic drift** (embedding-trajectory
  change) — vs low-uncertainty baselines. Expect elevated irregularity during uncertain/ambiguous
  spans and a systematic depth profile.

---

## Data model — one canonical tensor

Every downstream analysis reads a single exported archive so capture runs once:

```
resp[signal][branch]  ->  ndarray  (n_stim, T, n_layer, n_embd)
  signal  ∈ {"spikes", "voltage", "current"}     # + "rate" derived later
  branch  ∈ {"attn", "ffn"}                        # mem1 / mem2
labels                ->  dict of (n_stim, T) arrays:
    linguistic  : token_id, pos_tag, position, is_content
    uncertainty : surprisal = −log p(token), entropy = H[p(next)]   # from the model's own logits (H2/H4)
    drift       : semantic_drift = ‖Δ context-embedding‖ per token   # semantic-drift condition (H4)
meta                  ->  n_layer, n_embd, model_name, stimulus text, tokenizer
```

Surprisal/entropy come *for free* during capture: the forward pass already produces `out` (logits)
per token, so we log `softmax(out)` alongside the hooked activations — no extra passes.

`spikes` and `current` come straight from forward hooks; `voltage` is `_reconstruct_V(current)`
(closed form, exact for the LIF defaults). Keeping all three co-registered on the same
`(stim, token, layer, neuron)` grid is what makes the V-vs-S comparison a slice, not a re-run.

---

## Two activity regimes: driven vs spontaneous

Every pipeline runs on two conditions, the artificial analogue of stimulus-evoked vs
spontaneous neural activity:
- **driven** — teacher-forced on real text (`DEFAULT_STIMULI`); the network is clamped to
  ground-truth input each step (`capture.capture`).
- **spontaneous** — the model rolls out its own continuation from a short prompt and we record
  activity during the self-generated tokens only (`capture.capture_rollout`); the network runs
  on its own predictions.

`conditions.py` contrasts them per layer — firing rate, mean |V|, CV-ISI, participation ratio,
and a **subspace-alignment index** (fraction of spontaneous variance captured by the driven
condition's top-k PCs, relative to spontaneous's own; ~1 = shared subspace, ≪1 = distinct
regimes). Question: does the network occupy the same population geometry when driven by external
text as when free-running, or does self-rollout explore a different part of state space?

## Pipelines

### (1) Population geometry  — `geometry.py`
For a chosen `(signal, branch, layer)` build the `(neuron × token)` matrix and compute:
- **PCA** — variance explained, low-D trajectories through token time, per stimulus.
- **Participation ratio / intrinsic dimensionality** — `(Σλ)² / Σλ²`; the primary H1 metric.
- **jPCA** — rotational structure in the population trajectory during generation.
- **UMAP** (optional) — nonlinear embedding for visualization only.
- **Trajectory metrics** — path length, curvature, velocity autocorrelation (smoothness → H1).
Run the *same* code on `V`, `S`, `I` and overlay. Sweep across all 18 layers → depth curves.

### (4) Neural decoding — `decoding.py`
Cross-validated linear read-outs from the population vector at each `(signal, branch, layer)`:
- **Targets:** current token id (coarse: content vs function word, POS tag, position bucket),
  and *lagged* targets (token at t-k) for the temporal-memory analysis (H3).
- **Models:** ridge / logistic regression (linear = "what's linearly available"); optional MLP
  as a nonlinear ceiling.
- **Metrics:** balanced accuracy / R², with a shuffled-label null and CI over CV folds.
- **Curves:** accuracy vs layer, and the **V−S accuracy gap** vs layer (H2); accuracy vs lag (H3).

Additionally decode voltage-only, spike-only, and **joint `[V ‖ S]`** feature sets per layer to
test **H3** (super-additive gain), and report the joint − best-single accuracy curve.

### Salience analysis — `salience.py` (H2)
Relate spiking to token-level uncertainty from the model's own logits:
- Correlate per-token population spike count with **surprisal** and next-token **entropy**
  (per layer/branch); regress spike count on surprisal with a shuffled-token null.
- Spike-triggered comparison: high- vs low-surprisal tokens — is elevated spiking the *marker*
  of salient events even where voltage carries more identity information?
- Contrast with voltage: does `V` track uncertainty gradually while `S` flags it discretely?

### Firing irregularity — `isi.py` (H4)
- Per-neuron **ISI** (in token steps) and **CV of ISI** across layers.
- Compare irregularity across conditions — **ambiguity, high predictive entropy, semantic drift**
  (embedding-trajectory change) — vs matched low-uncertainty spans; depth profile of CV.

### Comparison orchestration — `compare.py`
Assembles the H1–H4 figures and a per-layer summary table
(PR_V, PR_S, acc_V, acc_S, acc_joint, V−S gap, joint gain, spike–surprisal r, mean CV-ISI).

---

## Critical correctness constraints (do not skip)

1. **Reset between stimuli.** LIF membrane state persists. On the RNN path, reset by (a) calling
   `node.reset()` on every `mem1[i]`/`mem2[i]` and (b) passing a zeroed recurrent `state` — do
   **not** pass `state=None`, which recreates the nodes and silently drops the forward hooks.
   Leaking state across stimuli corrupts every analysis (see CLAUDE.md).
2. **Input is continuous for the 216M pile model.** The network input is the raw embedding, not
   spikes. So the decoding "stimulus"/label is the token/embedding; the analyzed spikes are the
   *internal* LIF activations. (Only the char model, `vocab_size==77`, binarizes the input.)
3. **Voltage reconstruction is defaults-locked.** `_reconstruct_V` is exact only for
   `tau=2, v_th=1, v_reset=0, decay_input=True, hard reset`. If the checkpoint's LIF params
   differ, reconstruct from the node config, don't hard-code.
4. **Rescaling.** `model_run` rescales `att.output`/`ffn.value` by `2^(block//6)`. The captured
   current already reflects the rescaled forward pass, so V/S are internally consistent; just
   don't compare raw current magnitudes across the rescale boundary without noting it.

---

## Milestones

- **M0 — capture harness** (`capture.py`): hooks record spikes+current for *both* branches; log
  per-token logits (→ surprisal/entropy); reset-safe stimulus loop; export the canonical archive.
- **M1 — representations** (`representations.py`): `(neuron×token)` matrices + `rate` smoothing;
  sanity plots reproducing `spike_vis` firing-rate/V numbers.
- **M2 — geometry** (`geometry.py`): PCA/PR/jPCA + trajectory smoothness + lagged-decoding memory
  curves for V/S/I → **H1** figures.
- **M3 — decoding** (`decoding.py`): grouped-CV decoders + null; V-only / S-only / joint `[V‖S]`;
  accuracy-vs-layer, V−S gap (**H1**) and joint-gain (**H3**) figures.
- **M4 — salience** (`salience.py`): spike↔surprisal/entropy correlation, spike-triggered
  high/low-surprisal contrast → **H2** figure.
- **M5 — irregularity** (`isi.py`): ISI + CV-ISI by layer and condition (ambiguity / uncertainty /
  semantic drift) → **H4** figure.
- **M6 — writeup** (`compare.py`): combined summary table + figure panel; the
  "when is spike-only sufficient?" verdict per layer.

## How to run
```
python -m neuro_analysis.run_analysis --config neuro_analysis/configs/default.yaml
```
Outputs (tensors, figures, tables) go to `neuro_analysis/outputs/` (git-ignored).
Follow the repo `plot-style` conventions for all figures (Tol palette already used in spike_vis).
