# ==========================================
# HYBRID SCORER (FINAL RESEARCH VERSION)
# ==========================================

import re

# -------------------------------
# CONFIG (SEMANTIC-PROTECTED)
# -------------------------------
# Semantic similarity is the primary ranking signal. KG/keyword evidence only
# breaks close ties; COI is handled as a hard filter.
SIM_WEIGHT = 0.85
KG_WEIGHT  = 0.02
KEY_WEIGHT = 0.08
COI_THRESHOLD = 0.5

TOP_K = 20
RERANK_TOP = 10


# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text


# -------------------------------
# KEYWORD OVERLAP
# -------------------------------
def keyword_overlap(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)

    set1 = set(text1.split())
    set2 = set(text2.split())

    if not set1 or not set2:
        return 0.0

    return len(set1 & set2) / (len(set1 | set2) + 1e-8)


# -------------------------------
# KG SCORE
# -------------------------------
def compute_kg_score(pid, rid, cso_ms, cso_rev, gliner_ms, gliner_rev):

    paper_topics = set()
    reviewer_topics = set()

    if cso_ms and pid in cso_ms:
        paper_topics.update(map(str.lower, cso_ms[pid]))

    if gliner_ms and pid in gliner_ms:
        paper_topics.update(map(str.lower, gliner_ms[pid].get("all_labels", [])))

    if cso_rev and rid in cso_rev:
        reviewer_topics.update(map(str.lower, cso_rev[rid]))

    if gliner_rev and rid in gliner_rev:
        reviewer_topics.update(map(str.lower, gliner_rev[rid].get("all_labels", [])))

    if not paper_topics or not reviewer_topics:
        return 0.0

    inter = len(paper_topics & reviewer_topics)
    union = len(paper_topics | reviewer_topics)

    return inter / (union + 1e-8)


# -------------------------------
# BUILD TEXT MAPS (FIXED)
# -------------------------------
def build_text_maps(manuscripts, reviewers):

    paper_texts = {}
    reviewer_texts = {}

    # Papers → keep title boost + abstract
    for m in manuscripts.values():
        pid = str(m["paper_id"])
        title = m.get("title", "")
        abstract = m.get("abstract", "")

        paper_texts[pid] = (title + " ") * 3 + abstract

    # Reviewers → area + keywords + top titles (stronger signal)
    for r in reviewers.values():
        rid = str(r["reviewer_id"]) 

        area = r.get("area", "") or r.get("areas", "") or ""
        # keywords may be list or string
        kws = r.get("keywords", "")
        if isinstance(kws, list):
            kws = " ".join(kws)

        titles = []
        for pub in r.get("publications", [])[:5]:
            titles.append(pub.get("title", ""))

        reviewer_texts[rid] = " ".join([area, kws, " ".join(titles)]).strip()

    return paper_texts, reviewer_texts


# -------------------------------
# NORMALIZATION
# -------------------------------
def normalize(scores):
    if not scores:
        return scores

    min_v = min(scores)
    max_v = max(scores)

    if max_v - min_v < 1e-8:
        return [0.0 for _ in scores]

    return [(s - min_v) / (max_v - min_v + 1e-8) for s in scores]


# -------------------------------
# HYBRID SCORING
# -------------------------------
def hybrid_score(
    sim_data,
    manuscripts,
    reviewers,
    coi_dict,
    cso_ms,
    cso_rev,
    gliner_ms,
    gliner_rev
):

    print("\n=== HYBRID SCORING (FINAL) ===")

    paper_texts, reviewer_texts = build_text_maps(manuscripts, reviewers)

    fusion = {}

    for pid, ranked in sim_data.items():

        ranked = ranked[:TOP_K]
        temp = []

        for rid, sim_score in ranked:

            sim_score = float(sim_score)

            # KG
            kg_score = compute_kg_score(
                pid, rid, cso_ms, cso_rev, gliner_ms, gliner_rev
            )

            # Soft penalty
            if kg_score < 0.03:
                kg_score *= 0.2

            # Keyword
            key_score = keyword_overlap(
                paper_texts.get(pid, ""),
                reviewer_texts.get(rid, "")
            )

            # 🔥 HARD FILTER (SAFE)
            # COI is a filter, not a scoring penalty.
            coi_entry = coi_dict.get((pid, rid), {})
            coi = float(coi_entry.get("coi_score", 0) or 0)
            if coi_entry.get("flagged") or coi >= COI_THRESHOLD:
                continue

            temp.append({
                "reviewer_id": rid,
                "sim": sim_score,
                "kg": kg_score,
                "key": key_score
            })

        # Skip empty cases
        if not temp:
            continue

        # Normalize
        sim_vals = normalize([x["sim"] for x in temp])
        kg_vals  = normalize([x["kg"] for x in temp])
        key_vals = normalize([x["key"] for x in temp])

        results = []

        for i, item in enumerate(temp):

            score = (
                SIM_WEIGHT * sim_vals[i] +
                KG_WEIGHT  * kg_vals[i] +
                KEY_WEIGHT * key_vals[i]
            )

            # HARD FILTER: penalize weak raw similarity to improve precision
            # raw sim is item["sim"] (cosine), apply multiplicative down-weight
            try:
                raw_sim = float(item.get("sim", 0))
            except Exception:
                raw_sim = 0.0

            if raw_sim < 0.3:
                score *= 0.5

            results.append({
                "reviewer_id": item["reviewer_id"],
                "score": score
            })

        # -------------------------------
        # SMART RERANK
        # -------------------------------
        results.sort(key=lambda x: x["score"], reverse=True)

        for i in range(min(RERANK_TOP, len(results))):
            results[i]["score"] += 0.08 / (i + 1)

        results.sort(key=lambda x: x["score"], reverse=True)

        fusion[pid] = results

    return fusion
