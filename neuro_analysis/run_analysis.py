"""End-to-end driver: load SpikeGPT (RNN path) -> capture -> analyze -> figures.

    python -m neuro_analysis.run_analysis --config neuro_analysis/configs/default.yaml

Stages are toggled in the config so capture (slow, GPU/CPU forward passes) can be run
once and the analysis iterated cheaply off the saved archive.
"""

import argparse
import os
import sys
import types

import numpy as np
import yaml

# Work whether launched as a module (`python -m neuro_analysis.run_analysis`, preferred) or
# directly (`python neuro_analysis/run_analysis.py`). The latter has no package context, so
# put the repo root on sys.path and import absolutely.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from neuro_analysis import capture, datasets, compare
else:
    from . import capture, datasets, compare


def build_model(cfg):
    """Instantiate RWKV_RNN exactly as run.py does, from the config's model fields."""
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ["RWKV_RUN_DEVICE"] = cfg["run_device"]
    args = types.SimpleNamespace(
        MODEL_NAME=cfg["model_name"], n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
        ctx_len=cfg["ctx_len"], vocab_size=cfg["vocab_size"], RUN_DEVICE=cfg["run_device"],
        FLOAT_MODE=cfg["float_mode"], head_qk=0, pre_ffn=0, grad_cp=0, my_pos_emb=0)
    from src.model_run import RWKV_RNN
    from src.utils import TOKENIZER
    model = RWKV_RNN(args)
    tok = TOKENIZER(cfg["word_name"], UNKNOWN_CHAR=cfg.get("unknown_char"))
    return model, tok, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="neuro_analysis/configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, "responses.npz")

    if cfg["stages"]["capture"]:
        model, tok, args = build_model(cfg)
        stimuli = datasets.DEFAULT_STIMULI
        toks = datasets.tokenize_stimuli(tok, stimuli, cfg["token_mode"])
        resp, lengths, aux = capture.capture(model, toks, cfg["ctx_len"], cfg["n_layer"])
        labels = datasets.build_labels(toks, tok, cfg["token_mode"], aux=aux)
        meta = {"n_layer": cfg["n_layer"], "n_embd": cfg["n_embd"],
                "model_name": cfg["model_name"], "stimuli": stimuli, "lengths": lengths}
        capture.save_archive(archive, resp, labels, meta)
        print(f"captured -> {archive}")

    if cfg["stages"]["compare"]:
        compare.run(archive, out_dir, branch=cfg["branch"], target=cfg["target"])


if __name__ == "__main__":
    main()
