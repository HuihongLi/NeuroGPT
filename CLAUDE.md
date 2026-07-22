# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SpikeGPT is a generative language model built on the RWKV architecture where the standard
non-linearities are replaced with binary, event-driven **spiking neuron** activations (Leaky
Integrate-and-Fire). It is a fork/derivative of [RWKV-LM](https://github.com/BlinkDL/RWKV-LM);
much of the RWKV plumbing (WKV CUDA kernel, time-mixing/channel-mixing) is inherited, and the SNN
layer is added on top via the vendored `spikingjelly` library. Paper: arXiv:2302.13939.

## Commands

There is no build system, test suite, or linter. Everything runs as top-level scripts whose
configuration is edited **in the source file itself** (there is no CLI argument parsing beyond an
optional GPU index).

- **Inference / text generation:** `python run.py [GPU_INDEX]`
  Edit the `context` prompt, `MODEL_NAME`, `n_layer`, `n_embd`, `ctx_len`, `RUN_DEVICE`
  (`cpu`/`cuda`), and `FLOAT_MODE` near the top of `run.py`. Requires a `.pth` checkpoint under
  `model/` (e.g. `model/SpikeGPT-216M.pth`). CPU + `fp32` works and is reasonably fast.
  With `VISUALIZE = True` (default, top of `run.py`) it also saves a spiking-activity figure to
  `figs/spikegpt_activity.png` — see the Visualization section below.
- **Training (single GPU):** `python train.py`
  Edit `datafile_train`, `ctx_len`, `n_layer`, `n_embd`, `batch_size`, and learning-rate settings
  at the top of `train.py`. For binidx (pre-tokenized) data, uncomment the `MMapIndexedDataset`
  line and drop the `.bin`/`.idx` extension from the filename.
- **Training (multi-GPU):** launched via HuggingFace Accelerate — `accelerate launch train.py`
  (the trainer wraps everything in `Accelerator`).
- **NLU fine-tuning:** `python NLU/<task>.py` (sst-2, sst-5, mr, subj). Edit the checkpoint path
  inside the script before running. Note these scripts import `src.model_classify`, which is **not
  present** in this repo — it must be supplied before they will run.

## Runtime dependencies (beyond requirements.txt)

- **CUDA toolkit + `nvcc`** for training: `src/model.py` JIT-compiles the WKV kernel at import time
  via `torch.utils.cpp_extension.load(...)` from `cuda/wkv_op.cpp` + `cuda/wkv_cuda.cu`. Importing
  `src.model` on a machine without a CUDA compiler will fail. `run.py` uses `src.model_run`
  instead, which is a pure-PyTorch RNN implementation and does **not** need the kernel.
- **`cupy`** is required for training: the LIF neurons in `src/model.py` are created with
  `backend='cupy'`. It is not in `requirements.txt`. `src.model_run` does not use cupy.
- **`deepspeed`** is optional — `FusedAdam` is used if available, otherwise it falls back to
  `torch.optim.Adam`.

## Architecture

Two parallel model implementations exist and must be kept conceptually in sync:

- **`src/model.py` (`GPT`)** — the trainable model. Uses the fused WKV CUDA kernel (`RUN_CUDA`) for
  parallel time-mixing over the full sequence, and `spikingjelly` `MultiStepLIFNode` spiking
  neurons. This is what `train.py` and `NLU/*` build.
- **`src/model_run.py` (`RWKV_RNN`)** — the inference model used by `run.py`. Runs the *same*
  weights sequentially as an RNN (one token at a time, carrying hidden `state` + `mem` tensors),
  so no CUDA kernel is needed. On load it rescales `att.output.weight` and `ffn.value.weight` by
  `2^(block_id // RWKV_RESCALE_LAYER)` to keep fp16 activations in range — this rescaling is
  intrinsic to how the RNN path reproduces the trained model's output.

Both share the RWKV block structure: each `Block` = LayerNorm → time-mixing (`RWKV_TimeMix`, the
attention-replacement using time_decay/time_first/WKV) or channel-mixing (`RWKV_ChannelMix`),
wrapped by a spiking LIF neuron with a residual add. Key SNN detail in `src/model.py`: the LIF
node runs in multi-step mode, so tensors are `permute(1,0,2)`'d into `(T, B, C)` before the neuron
and back after.

### Spiking-network state must be reset between forward passes

`spikingjelly` neurons are stateful (membrane potential persists). The trainer calls
`functional.reset_net(model)` after each step (`src/trainer.py`). **Any new training/eval loop you
write must reset the net between sequences**, or membrane state leaks across batches and corrupts
results.

### Config plumbing

`GPTConfig`/`TrainerConfig` (in `src/model.py` / `src/trainer.py`) just absorb arbitrary kwargs via
`setattr`. `run.py` instead uses a `types.SimpleNamespace` (`args`). There is no shared config
schema — the scripts and the model classes agree by convention on field names
(`n_layer`, `n_embd`, `ctx_len`, `vocab_size`, `model_type`).

- `model_type`: `'RWKV'` (default) or `'RWKV-ffnPre'` (channel-mix before attention in block 0).
- `T_MAX` in `src/model.py` must be `>= ctx_len` (it sizes the CUDA kernel; larger costs VRAM).
- Tokenization: `TOKEN_MODE = 'pile'` uses the HF tokenizer `20B_tokenizer.json`
  (vocab 50277); `'char'` mode uses `vocab_book.json` for char-level BookCorpus models.

## Data

`src/utils.Dataset` takes either a raw text string or an `MMapIndexedDataset` (binidx). Custom
corpora are converted to binidx with the external
[json2binidx_tool](https://github.com/Abel2076/json2binidx_tool). Mini-epochs are fixed-length
(`ctx_len * epoch_length_fixed` tokens), not full passes over the data.

## Repository layout / conventions

Scripts read and write through fixed directories (paths are hardcoded in the scripts, relative to
the repo root — run from the root):

- `model/` — checkpoints. Inference loads `model/SpikeGPT-216M.pth`; `train.py` writes checkpoints
  here (`epoch_save_path`, final `torch.save`).
- `data/` — tokenizers (`20B_tokenizer.json`, `vocab_book.json`, tracked in git) and training
  corpora (`enwik8`, binidx files — git-ignored).
- `figs/` — generated figures (git-ignored except `.gitkeep`).
- `results/` — training logs (`trainer.py` appends to `results/wik8-0.01.txt`; git-ignored).

`.pth`/`.pt` checkpoints are git-ignored (the 216M model is downloaded, not committed).

## Visualization

`src/spike_vis.py` renders spiking activity captured during generation on the RNN inference path.
It attaches forward hooks to the `mem1`/`mem2` `LIFNode`s (attention- and FFN-branch spiking
neurons) to record spikes (0/1) and the pre-neuron current, and reconstructs the membrane
potential `V[t]` in closed form (exact for the `LIFNode` defaults `tau=2, v_th=1, v_reset=0,
decay_input=True`, hard reset). `run.py` calls it when `VISUALIZE = True`. One figure combines:
per-layer spike rasters, per-layer firing rate over tokens (attn + FFN), membrane-voltage traces,
and an input-embedding panel. Note on the input: for the 216M pile model the network input is the
**raw continuous embedding, not spikes** — only the char model (`vocab_size == 77`) applies an
`ATan` heaviside to the embedding in `model_run.py`.
