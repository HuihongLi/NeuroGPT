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
    # more science
    "Bacteria can share genes directly, spreading resistance faster than mutation alone.",
    "The telescope collected light that had traveled for billions of years to reach us.",
    "Enzymes lower the activation energy required for a biochemical reaction to proceed.",
    "Tectonic plates grind past one another, storing stress that is released as earthquakes.",
    "A superconductor carries current with no resistance below its critical temperature.",
    "The genome encodes instructions, but the environment shapes which genes are expressed.",
    # more news / economics
    "Investors grew nervous as bond yields climbed to their highest level in years.",
    "The city council approved funding for a new subway line after a long debate.",
    "Aid agencies struggled to reach the flooded region before the next storm arrived.",
    "Regulators fined the bank for failing to report suspicious transactions promptly.",
    "The startup doubled its workforce and opened a second office overseas this spring.",
    # more narrative
    "The letter had no return address, only a faded stamp from a foreign country.",
    "Morning fog rolled off the river as the fishermen pushed their boats from shore.",
    "He counted the coins twice, then slid them across the counter without a word.",
    "The house at the end of the lane had been empty for as long as anyone remembered.",
    "She memorized the map, burned it, and walked into the desert before dawn.",
    "The orchestra fell silent, and for a moment no one in the hall dared to breathe.",
    # more history / society
    "The printing press spread ideas faster than any authority could suppress them.",
    "Farmers abandoned the drought-stricken plains and moved west in search of work.",
    "The revolution promised equality but soon devoured many of its own leaders.",
    "Trade along the ancient road carried silk, spices, religion, and disease alike.",
    # more technical / instructional
    "Allocate the buffer before the loop, and free it once the last worker returns.",
    "Measure twice, cut once, and always clamp the board before the blade spins up.",
    "Cache the query result so repeated requests do not hit the database every time.",
    "Label each wire at both ends before you disconnect anything from the panel.",
    # more everyday / dialogue
    "Do you want to split the last slice, or should we just order another pizza?",
    "The bus was late again, so we stood in the rain arguing about the schedule.",
    "I can't find my keys anywhere, and we were supposed to leave ten minutes ago.",
    "She whispered that the surprise party was tonight, so act normal at dinner.",
    # more abstract / opinion
    "Memory is less a recording than a story we quietly rewrite each time we recall it.",
    "The market rewards confidence, which is why it so often mistakes luck for skill.",
    "Language shapes thought, but thought, straining against it, reshapes language in turn.",
    "We overestimate what we can do in a day and underestimate what a decade can hold.",
    "Trust is built in drops and lost in buckets, and rarely refilled at the old rate.",
]

# Short seed prompts for the SPONTANEOUS (self-rollout) condition — the model continues
# each on its own, and we record activity during its self-generated tokens. Kept generic
# so the rollout is genuinely driven by the model's dynamics, not a strong topical cue.
# More prompts = more CV groups for the spontaneous condition (tighter CIs).
DEFAULT_PROMPTS = [
    "The most important thing to remember is",
    "In the beginning, there was",
    "Scientists have recently discovered that",
    "She looked out the window and thought about",
    "The problem with modern cities is that",
    "Once upon a time, in a distant land,",
    "According to the report, the main cause was",
    "He picked up the letter and began to read:",
    "The future of technology depends on",
    "After the long journey, they finally",
    "Nobody expected that the experiment would",
    "The history of the region is marked by",
    "When the storm finally passed, the village",
    "The key difference between the two theories is",
    "In her speech, the president argued that",
    "The recipe had been passed down for",
    "Deep in the forest, something was",
    "Economists disagree about whether",
    "The old machine still worked, but",
    "To understand the problem, we must first",
    "It was a cold morning when the",
    "The committee concluded that the evidence",
    "Every great discovery begins with",
    "The letter arrived three weeks after",
]


def load_lines(path):
    """Read one stimulus/prompt per non-empty line — lets you scale the corpus without
    editing this file (set stimuli_file / prompts_file in the config)."""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


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
