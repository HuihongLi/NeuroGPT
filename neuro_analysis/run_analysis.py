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


def _meta(cfg, extra):
    m = {"n_layer": cfg["n_layer"], "n_embd": cfg["n_embd"], "model_name": cfg["model_name"]}
    m.update(extra)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="neuro_analysis/configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    conds = cfg.get("conditions", {"driven": True, "spontaneous": False})
    stages = cfg["stages"]
    driven_arch = os.path.join(out_dir, "responses_driven.npz")
    spont_arch = os.path.join(out_dir, "responses_spont.npz")

    if stages["capture"]:
        model, tok, args = build_model(cfg)

        if conds.get("driven"):
            stimuli = (datasets.load_lines(cfg["stimuli_file"])
                       if cfg.get("stimuli_file") else datasets.DEFAULT_STIMULI)
            toks = datasets.tokenize_stimuli(tok, stimuli, cfg["token_mode"])
            resp, lengths, aux = capture.capture(model, toks, cfg["ctx_len"], cfg["n_layer"])
            labels = datasets.build_labels(toks, tok, cfg["token_mode"], aux=aux)
            capture.save_archive(driven_arch, resp, labels,
                                 _meta(cfg, {"stimuli": stimuli, "lengths": lengths,
                                             "condition": "driven"}))
            print(f"captured driven -> {driven_arch}")

        if conds.get("spontaneous"):
            ro = cfg.get("rollout", {})
            prompts = (datasets.load_lines(cfg["prompts_file"])
                       if cfg.get("prompts_file") else datasets.DEFAULT_PROMPTS)
            ptoks = datasets.tokenize_stimuli(tok, prompts, cfg["token_mode"])
            resp, lengths, aux, gen = capture.capture_rollout(
                model, tok, ptoks, ro.get("n_generate", 60), cfg["ctx_len"], cfg["n_layer"],
                temperature=ro.get("temperature", 0.8), top_p=ro.get("top_p", 0.7))
            labels = datasets.build_labels(gen, tok, cfg["token_mode"], aux=aux)
            capture.save_archive(spont_arch, resp, labels,
                                 _meta(cfg, {"prompts": prompts, "lengths": lengths,
                                             "condition": "spontaneous"}))
            print(f"captured spontaneous -> {spont_arch}")

    if stages["compare"]:
        if conds.get("driven"):
            compare.run(driven_arch, os.path.join(out_dir, "driven"),
                        branch=cfg["branch"], target=cfg["target"])
        if conds.get("spontaneous"):
            compare.run(spont_arch, os.path.join(out_dir, "spont"),
                        branch=cfg["branch"], target=cfg["target"])

    if stages.get("contrast") and conds.get("driven") and conds.get("spontaneous"):
        compare.run_conditions(driven_arch, spont_arch,
                               os.path.join(out_dir, "contrast"), branch=cfg["branch"])


if __name__ == "__main__":
    main()
