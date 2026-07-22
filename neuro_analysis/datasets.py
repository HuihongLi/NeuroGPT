"""Stimulus sets + label extraction.

A "stimulus" is a short text passage; each token is a time step with labels we later
try to decode. Keep passages length-matched-ish so the (stim, token) grid pads cleanly.

Decoding targets (see PROJECT_PLAN.md pipeline #4):
  token_id   : raw next-token id (high-cardinality; usually decode a coarser view)
  is_content : content word (noun/verb/adj/adv) vs function word  -> clean binary target
  pos_tag    : coarse POS via a lightweight tagger
  position   : within-sequence position bucket (tests positional coding / H3 memory)
"""

# A small, topically varied default probe set. Swap/extend for a real study; larger and
# more balanced sets give tighter decoding CIs.
DEFAULT_STIMULI = [
    "The development of renewable energy has accelerated over the past decade.",
    "Prehistoric humans painted animals on the limestone walls of a deep cave.",
    "The central bank raised interest rates to slow rising consumer prices.",
    "A small herd of dragons was discovered in a remote Tibetan valley.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into sugar.",
    "The defendant was found guilty and sentenced by the presiding judge.",
]


def tokenize_stimuli(tokenizer, stimuli, token_mode="pile"):
    """Return list[list[int]] token ids for each stimulus using the run.py tokenizer."""
    if token_mode == "pile":
        return [tokenizer.tokenizer.encode(s) for s in stimuli]
    # char mode
    return [[tokenizer.stoi.get(c, tokenizer.UNKNOWN_CHAR) for c in s] for s in stimuli]


# Closed-class English function words: everything else is treated as a content word.
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at",
    "by", "for", "with", "as", "is", "are", "was", "were", "be", "been", "being", "am",
    "it", "its", "this", "that", "these", "those", "he", "she", "they", "we", "you", "i",
    "his", "her", "their", "our", "your", "my", "me", "him", "them", "us", "who", "whom",
    "which", "what", "from", "into", "over", "under", "up", "down", "out", "so", "than",
    "not", "no", "do", "does", "did", "has", "have", "had", "will", "would", "can",
    "could", "should", "may", "might", "must", "there", "here", "about", "up", "just",
}


def _is_content(text):
    w = text.strip().lower().strip(".,;:!?\"'()[]")
    return 0 if (w == "" or w in FUNCTION_WORDS) else 1


def build_labels(stimuli_tokens, tokenizer, token_mode="pile", aux=None):
    """Build the per-(stim, token) label dict (ragged: list-of-lists, one list per stimulus).

    Linguistic (from tokens/text):
        token_id, position, is_content   (content vs function word, via FUNCTION_WORDS)
    Uncertainty + drift (from `aux` produced by capture.capture):
        surprisal, entropy, semantic_drift = ‖emb[t] - emb[t-1]‖
    `pos_tag` is left out by default (needs a tagger); is_content is the clean binary target.
    """
    labels = {k: [] for k in
              ("token_id", "position", "is_content", "surprisal", "entropy", "semantic_drift")}
    for s, toks in enumerate(stimuli_tokens):
        labels["token_id"].append(list(toks))
        labels["position"].append(list(range(len(toks))))
        if token_mode == "pile":
            texts = [tokenizer.tokenizer.decode([t]) for t in toks]
        else:
            texts = [tokenizer.itos.get(t, "") for t in toks]
        labels["is_content"].append([_is_content(t) for t in texts])

        if aux is not None:
            labels["surprisal"].append(list(aux["surprisal"][s]))
            labels["entropy"].append(list(aux["entropy"][s]))
            emb = aux["embedding"][s]
            drift = [0.0] + [float(((emb[i] - emb[i - 1]) ** 2).sum() ** 0.5)
                             for i in range(1, len(emb))]
            labels["semantic_drift"].append(drift)
    if aux is None:
        for k in ("surprisal", "entropy", "semantic_drift"):
            del labels[k]
    return labels
