"""Stimulus sets + label extraction.

A "stimulus" is a short text passage; each token is a time step with labels we later
try to decode. Keep passages length-matched-ish so the (stim, token) grid pads cleanly.

Decoding targets (see PROJECT_PLAN.md pipeline #4):
  token_id   : raw next-token id (high-cardinality; usually decode a coarser view)
  is_content : content word (noun/verb/adj/adv) vs function word  -> clean binary target
  pos_tag    : coarse POS via a lightweight tagger
  position   : within-sequence position bucket (tests positional coding / H3 memory)
"""

# Topically varied probe set for the INPUT-DRIVEN (teacher-forced) condition. Larger and
# more balanced than a handful of sentences -> tighter decoding CIs. Mix of registers
# (science, news, narrative, history, law, technical, everyday) so decoded structure is
# not tied to one domain. Extend freely; the CV is grouped by stimulus so it scales.
DEFAULT_STIMULI = [
    # science
    "The development of renewable energy has accelerated over the past decade.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The mitochondria supply most of the chemical energy that a living cell needs.",
    "Gravity bends the path of light as it passes close to a very massive object.",
    "Vaccines train the immune system to recognize and destroy specific pathogens.",
    "The greenhouse effect traps heat in the atmosphere and warms the planet slowly.",
    "Neurons communicate by releasing chemical signals across narrow synaptic gaps.",
    "The periodic table arranges the elements by their increasing atomic number.",
    # news / economics
    "The central bank raised interest rates again to slow rising consumer prices.",
    "Officials warned that the drought could reduce the national grain harvest sharply.",
    "The company reported record quarterly profits despite weak overseas demand.",
    "Negotiators reached a fragile ceasefire after months of difficult diplomacy.",
    "Voters headed to the polls under tight security on a cold election morning.",
    "The airline canceled hundreds of flights as the winter storm moved inland.",
    # narrative
    "She opened the rusty gate and stepped into the overgrown, silent garden.",
    "The old sailor told stories of storms that no one in the tavern believed.",
    "A single lamp burned in the window long after the rest of the town slept.",
    "He hesitated at the door, then knocked twice and waited for an answer.",
    "The children chased the kite until it vanished behind the distant hills.",
    "Rain hammered the tin roof while the travelers waited for the bus to arrive.",
    # history
    "Prehistoric humans painted animals on the limestone walls of a deep cave.",
    "The empire collapsed slowly as trade routes shifted and cities lost their wealth.",
    "Explorers mapped the coastline long before anyone ventured into the interior.",
    "The treaty redrew the borders and left several old grievances unresolved.",
    # law / formal
    "The defendant was found guilty and sentenced by the presiding judge.",
    "The contract clearly states that either party may terminate with written notice.",
    "The committee reviewed the evidence carefully before issuing its final report.",
    "Under the new statute, applicants must disclose any prior financial interest.",
    # technical / instructional
    "Restart the server, clear the cache, and then run the migration script again.",
    "The algorithm sorts the list in place and returns the number of comparisons.",
    "Tighten each bolt in a star pattern to spread the load across the plate evenly.",
    "Back up the database before applying any update to the production environment.",
    # everyday
    "I forgot my umbrella at home, so I got completely soaked on the way to work.",
    "We should leave early to avoid the traffic and still make the afternoon train.",
    "The recipe calls for two eggs, a cup of flour, and a pinch of salt.",
    "Could you turn down the music a little while I finish this phone call?",
    "The dog barked at every passing car until the neighbors began to complain.",
    # abstract / opinion
    "Freedom without responsibility eventually undermines the very trust it depends on.",
    "Curiosity, more than talent, seems to separate great scientists from the rest.",
    "A small kindness offered at the right moment can change the whole day.",
    "Progress is rarely a straight line; it advances, stalls, and doubles back.",
]

# Short seed prompts for the SPONTANEOUS (self-rollout) condition — the model continues
# each on its own, and we record activity during its self-generated tokens. Kept generic
# so the rollout is genuinely driven by the model's dynamics, not a strong topical cue.
DEFAULT_PROMPTS = [
    "The most important thing to remember is",
    "In the beginning, there was",
    "Scientists have recently discovered that",
    "She looked out the window and thought about",
    "The problem with modern cities is that",
    "Once upon a time, in a distant land,",
    "According to the report, the main cause was",
    "He picked up the letter and began to read:",
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
