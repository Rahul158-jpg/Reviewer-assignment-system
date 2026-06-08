import re
from collections import Counter

# Expanded stopwords to aggressively remove junk tokens
STOPWORDS = {
    "the","is","are","was","were","this","that","these","those",
    "and","or","for","with","from","into","onto","of","to",
    "in","on","by","an","a","as","at","it","its","be",
    "we","our","their","they","them","such","using",
    "based","approach","method","model","system","data"
}

# Additional noisy tokens (verbs, adverbs, generic adjectives)
STOPWORDS.update({
    "provides", "provide", "providing", "provided",
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "used", "use", "uses", "available", "recent", "recently", "currently",
    "highly", "however", "towards", "toward", "shown", "show", "shows", "showed",
    "these", "such", "result", "results", "findings", "paper", "study", "studies",
    "present", "presented", "introduce", "introduces", "introducing"
})


# Generic, unhelpful tokens to filter out
GENERIC_TERMS = {
    "method", "approach", "model", "system", "data",
    "analysis", "study", "work", "paper",
    "learn", "learning", "analyze", "opinion",
    "task", "result"
}

CONCEPT_MAP = {
    "nlp": ["natural language processing", "text classification"],
    "prompt": ["prompt engineering", "few-shot learning"],
    "statistics": ["statistical extraction", "information extraction"],
    "graph": ["knowledge graph", "graph modeling"],
    "transformer": ["transformer models", "attention mechanism"],
    "classification": ["text classification", "document classification"],
}

# Add method-level concept mappings for stronger, research-grade phrasing
CONCEPT_MAP.update({
    "tabular": ["structured data querying", "semantic table understanding"],
    "chatbot": ["conversational interface", "dialog systems"],
    "conversation": ["conversational interface", "dialog systems"],
    "data": ["data analysis", "data exploration"],
})


# Improve concept coverage for prompt-related research terminology
CONCEPT_MAP.update({
    "template": [
        "prompt templates",
        "prompt design"
    ],
    "verbalizer": [
        "prompt verbalizers",
        "label mapping in prompts"
    ],
    "resume": [
        "information extraction",
        "document parsing"
    ],
    "prompt": [
        "prompt learning",
        "few-shot learning",
        "prompt engineering"
    ]
})


# Domain-level concept synonyms (high-value research phrases)
CONCEPT_MAP.update({
    "program synthesis": ["code generation", "program generation"],
    "code generation": ["program synthesis"],
    "decoding": ["generation", "inference", "transformer decoding"],
    "optimization": ["efficiency", "performance"],
    "transformer": ["transformer models", "attention mechanism", "attention model", "transformer decoding"]
})

# Simple normalization (lightweight lemmatization) + synonyms
SYNONYMS = {
    "nlp": "natural language processing",
    "transformers": "transformer",
    "learning": "learn",
    "models": "model",
    "networks": "network",
    "analysis": "analyze",
    "sentiment": "opinion",
}


def normalize(word):
    word = word.lower()

    # synonym mapping
    if word in SYNONYMS:
        word = SYNONYMS[word]

    # safer suffix handling: handle common 'ies' -> 'y', and light past stripping
    if word.endswith("ies") and len(word) > 5:
        word = word[:-3] + "y"
    elif word.endswith("ed") and len(word) > 5:
        word = word[:-2]
    # prefer singular where safe (avoid trimming short words or words ending with 'ss')
    

    # remove any remaining non-alphanumerics
    word = re.sub(r'[^a-z0-9\s]', '', word)

    return word


def extract_keyphrases(text, top_k=10):

    text = text.lower()

    # preserve multi-word concept occurrences from CONCEPT_MAP (boost phrases)
    phrase_matches = set()
    try:
        for key in CONCEPT_MAP.keys():
            if " " in key:
                # match phrase boundaries, robust to punctuation
                pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
                if re.search(pattern, text):
                    phrase_matches.add(key)
    except Exception:
        phrase_matches = set()

    # remove special chars
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    words = text.split()

    # normalize once and filter aggressively (use normalized token for checks)
    norm_words = []
    for w in words:
        nw = normalize(w)
        if not nw:
            continue
        if nw in STOPWORDS:
            continue
        if nw in GENERIC_TERMS:
            continue
        if len(nw) <= 3:
            continue
        norm_words.append(nw)

    # --- PHRASES (prefer richer bigrams where both tokens are informative) ---
    def extract_phrases(norm_words_list):
        phrases = []
        for i in range(len(norm_words_list) - 1):
            a = norm_words_list[i]
            b = norm_words_list[i + 1]
            if len(a) > 4 and len(b) > 4:
                phrases.append(f"{a} {b}")
        return phrases

    bigrams = extract_phrases(norm_words)

    all_terms = norm_words + bigrams

    # include any detected multi-word phrases so they are not lost
    if phrase_matches:
        all_terms = all_terms + list(phrase_matches)

    freq = Counter(all_terms)
    terms = [w for w, _ in freq.most_common(top_k)]

    # quick dedupe while preserving order
    seen = set()
    dedup = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            dedup.append(t)

    # enrich and return broader research-level concept phrases
    return enrich_keywords(dedup)


def clean_keywords(words):
    return [
        w for w in words
        if len(w) > 4 and w not in STOPWORDS and w not in GENERIC_TERMS
    ]


def enrich_keywords(keywords):
    """Expand short keywords into richer concept phrases using CONCEPT_MAP.

    Returns a list with original keywords plus mapped concept phrases.
    """
    enriched = set()
    for k in (keywords or []):
        key = k.lower()
        # keep original token
        enriched.add(k)
        # direct mapping
        if key in CONCEPT_MAP:
            enriched.update(CONCEPT_MAP[key])
        # try simple singular/plural and gerund variants to match map keys
        if key.endswith("s"):
            sing = key[:-1]
            if sing in CONCEPT_MAP:
                enriched.update(CONCEPT_MAP[sing])
        if key.endswith("ing"):
            base = key[:-3]
            if base in CONCEPT_MAP:
                enriched.update(CONCEPT_MAP[base])
        if key.endswith("ies") and len(key) > 4:
            alt = key[:-3] + "y"
            if alt in CONCEPT_MAP:
                enriched.update(CONCEPT_MAP[alt])
    return list(enriched)