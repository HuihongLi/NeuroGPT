# Project Plan — Driven vs spontaneous dynamics of SpikeGPT (with a voltage-vs-spike lens)

**Goal.** Treat SpikeGPT's internal LIF activations as a neural recording and characterize how
its population dynamics differ between two regimes, using standard neural-data-analysis pipelines
(population geometry, decoding, ISI statistics, rollout dynamics).

> **Primary axis — driven vs spontaneous.** Does the network behave differently when *driven* by
> ground-truth text (teacher-forced) versus running *spontaneously* on its own sampled tokens
> (autoregressive rollout)? This is the artificial analogue of stimulus-evoked vs spontaneous
> neural activity. First results show the two regimes **share a linguistic code but differ in
> dynamics** (see Findings) — the rollout-dynamics pipeline is built to explain *why*.

> **Secondary lens — what does binarization cost?** For every layer/branch/token we have three
> co-registered signals from the *same* neurons — graded current `I`, reconstructed membrane
> voltage `V`, and spike `S ∈ {0,1}`. Comparing `V` vs `S` (vs `I`) quantifies how much structure
> the spike-generating nonlinearity discards. This lens is applied *within each regime*.

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

## Findings so far (2026-07-22, SpikeGPT-216M, attn branch) & updated directions

First real two-condition run: **driven = 597 tokens / 41 passages**, **spontaneous = 480 tokens /
8 prompts × 60**. Decoding target = content vs function word. What held up:

**Robust (consistent across all 18 layers):**
- **Voltage trajectories are smoother than spikes** (curvature V < S at every layer, both
  conditions). *H1 supported.*
- **Voltage out-decodes spikes when driven** — the V−S gap is **positive at all 18 driven
  layers** (+0.01…+0.15, acc_V 0.78–0.89 vs acc_S 0.67–0.82). *H1 supported.*
- **Joint [V‖S] adds nothing over voltage alone** (joint_gain ≈ 0 everywhere). Spikes are a
  thresholded function of V, so V subsumes S for linear decoding. *H3 not supported — a clean
  negative: for read-out, voltage is sufficient.*
- **Spontaneous firing is far more irregular than driven**: CV-ISI 0.53–0.91 (spont) vs
  0.29–0.54 (driven) at **every** layer. The biggest, most reliable regime difference. *New.*

**Structured / regime-dependent (now interpretable, worth chasing):**
- **The V>S decoding advantage is gated by the regime.** In spontaneous rollout the gap
  collapses (negative at L0, L3, L15–17) and overall accuracy drops (0.62–0.78). Voltage's edge
  is a property of being *externally driven*, not intrinsic. *New — reframes H1.*
- **Participation ratio: spikes ≫ voltage, and voltage is U-shaped**, tiny mid-network (~1–2)
  but exploding at the readout layers (L15–17: 47–67, driven). Binarization *spreads* variance
  across many weak dimensions; voltage concentrates in a few modes then re-expands near output.
  *H1's "voltage higher-dimensional" prediction is wrong in the PR sense — reframe as variance
  concentration, not information.*
- **Subspace alignment (voltage): shared core, divergent ends.** Spontaneous voltage variance
  is almost fully captured by the driven subspace in **middle layers (align ≈ 0.94–1.0)** but
  poorly at the **deep output layers (L15–17: 0.27–0.48)**. The two regimes share a common
  mid-network geometry and diverge near the readout. *New.*
- **Spike surprise-coding is layer-structured**, not uniform: spikes track surprisal positively
  at the input (L0) and output (L16–17) layers, negatively in the middle. *Refines H2 — surprise
  is marked at the ends, not the core.*
- **The linguistic code is regime-invariant even though the dynamics are not.** A voltage decoder
  trained on driven activity reads spontaneous activity at ~0.62–0.78 balanced acc, and
  spont→driven at ~0.68–0.81 — both near the within-condition ceiling and well above 0.5. So the
  *content/function representation transfers across regimes*, while firing irregularity (CV-ISI)
  and deep-layer geometry differ sharply. **Dissociation: shared representation, different
  dynamics.** *Key new result.*

### Updated directions (priority order)
1. **Promote driven-vs-spontaneous to the primary axis.** New hypotheses:
   - **H5 — spontaneous activity is more irregular / more Poisson-like** than driven (CV-ISI,
     and follow up with avalanche/criticality and temporal autocorrelation).
   - **H6 — a shared "core" representational subspace in middle layers, divergence at
     input/output layers** between the two regimes (principal angles, not just variance capture).
   - **H7 — linguistic decodability (and the V>S gap) is regime-gated**: present when driven,
     attenuated when free-running.
2. **Cross-condition decoding** (train driven → test spontaneous and vice versa): does the code
   transfer? Being added to `conditions.py` / the contrast figure set.
3. **Rollout dynamics:** does self-rollout collapse toward attractors/repetition? Measure state
   recurrence, trajectory speed, velocity autocorrelation; relate to sampling temperature (sweep).
4. **Robustness fixes:** more spontaneous prompts (only 8 CV groups now → wide CIs); add
   shuffled-label nulls to every decoder; clip/reinterpret the alignment index (cross/self can
   exceed 1 from mean-centering — e.g. L0 = 1.30); compare attn vs ffn branch.
5. **Synthetic LFP/EEG** (`lfp.py`, H8): field-potential analogues + spectral slope / timescale /
   cross-layer synchrony, contrasted driven vs spontaneous. Connects to the aperiodic-exponent
   (FOOOF) and intrinsic-timescale literature. Future: proper Welch coherence, band definitions,
   and mapping token-rate spectra to a nominal sampling assumption.

## Two activity regimes: driven vs spontaneous

Every pipeline runs on two conditions, the artificial analogue of stimulus-evoked vs
spontaneous neural activity:
- **driven** — teacher-forced on real text (`DEFAULT_STIMULI`); the network is clamped to
  ground-truth input each step (`capture.capture`).
- **spontaneous** — the model rolls out its own continuation from a short prompt and we record
  activity during the self-generated tokens only (`capture.capture_rollout`); the network runs
  on its own predictions.

`conditions.py` contrasts them per layer — firing rate, mean |V|, CV-ISI, participation ratio,
a **subspace-alignment index** (fraction of spontaneous variance captured by the driven
condition's top-k PC directions, relative to spontaneous's own; ∈[0,1], ~1 = shared subspace),
and **cross-condition decoding** (train in one regime, test in the other — does the code transfer?).

### Rollout dynamics — `rollout_dynamics.py` (PRIMARY)
Explains *why* the spontaneous regime differs, by profiling how the population trajectory moves,
per layer, driven vs spontaneous:
- **speed** — mean ‖ΔV[t]‖ per token (how fast the state moves).
- **velocity autocorrelation** — lag-1 cosine of the velocity vector: directed flow (→1) vs
  diffusive jitter (→0).
- **recurrence rate** — fraction of non-adjacent token pairs whose states are within a
  per-sequence distance threshold; high ⇒ the trajectory revisits states (attractor/looping).
- **token repetition** — fraction of generated tokens repeating within a window; a direct index
  of self-rollout degeneration (spontaneous only).
Hypothesis (H5): spontaneous rollout is more recurrent / lower-velocity-autocorrelation (drifts
toward attractors and loops), which also drives its higher CV-ISI.

### Synthetic LFP / EEG — `lfp.py`
Build field-potential analogues from the same activations and analyze them with standard
electrophysiology spectral tools:
- **LFP_layer[t]** = population-mean pre-neuron **current** of a layer (∼ summed synaptic
  currents); **EEG_global[t]** = sum of the per-layer LFPs (spatially-pooled proxy).
- Per layer: **aperiodic 1/f slope** (log-log spectral slope / "aperiodic exponent"),
  **low-band power fraction**, **intrinsic timescale** (lag-1 autocorrelation), plus
  **cross-layer synchrony** (mean pairwise correlation of layer LFPs ∼ inter-electrode coherence).
- Caveat baked into the code: time is the **token step**, so frequencies are in **cycles/token
  (Nyquist 0.5), not Hz**, and sequences are short → report robust low-resolution summaries, not
  sharp peaks.
Hypothesis (**H8**): the spontaneous regime has a **steeper 1/f slope / longer intrinsic
timescale** (more low-frequency drift, consistent with looping) and different cross-layer
synchrony than the driven regime — an LFP-level signature of the same dynamics difference H5
measures at the trajectory level.

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

- **M0–M6 — done.** Capture harness (both branches + logits), representations, geometry
  (PR/jPCA/smoothness), decoding (grouped-CV + null + joint `[V‖S]` + lagged), salience, ISI,
  and the `compare.run` battery — all implemented and run on the 216M model.
- **M7 — two-regime capture — done.** `capture.capture_rollout` (spontaneous) + driven; per-regime
  battery + `compare.run_conditions` contrast (rate, |V|, CV-ISI, PR, subspace alignment,
  cross-condition decoding).
- **M8 — rollout dynamics (PRIMARY, current)** (`rollout_dynamics.py`): speed, velocity
  autocorrelation, recurrence rate, token repetition → **H5** (driven vs spontaneous dynamics).
- **M9 — robustness**: shuffled-label nulls on every decoder (done for identity), bounded
  subspace-alignment metric (fixed), more spontaneous prompts + longer rollout (`n_generate`),
  optional external corpora (`stimuli_file`/`prompts_file`); TODO: attn-vs-ffn sweep, principal
  angles, temperature sweep of the rollout.
- **M10 — writeup**: the "shared code, different dynamics" story across regimes.

## How to run
```
python -m neuro_analysis.run_analysis --config neuro_analysis/configs/default.yaml
```
Outputs (tensors, figures, tables) go to `neuro_analysis/outputs/` (git-ignored).
Follow the repo `plot-style` conventions for all figures (Tol palette already used in spike_vis).
