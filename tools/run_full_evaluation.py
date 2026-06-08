import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
from scipy.stats import ttest_rel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

from assignment.assigner import (
    build_paper_text,
    build_reviewer_text,
    get_citations,
    get_h_index,
    get_reviewer_cache_info,
    prepare_reviewers_for_ranking,
    rank_reviewers_for_paper,
    reviewer_capacity_status,
)
import assignment.assigner as assigner_module
from config import RANKING_SCORING_MODE


DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

K = 3
RANK_DEPTH = int(os.getenv("RANK_DEPTH", "100"))
RETRIEVAL_DEPTH = int(os.getenv("RETRIEVAL_DEPTH", "100"))
RERANK_DEPTH = int(os.getenv("RERANK_DEPTH", "10"))
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0"))
PROGRESS_EVERY = int(os.getenv("EVAL_PROGRESS_EVERY", "25"))
CROSS_MAX_EVAL_PAIRS = int(os.getenv("CROSS_MAX_EVAL_PAIRS", "80"))


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def embedding_cache_path(name):
    return os.path.join(BASE_DIR, "cache", name)


def cached_encode_texts(texts, cache_name, batch_size=32):
    model_name = getattr(assigner_module, "_ACTIVE_EMBEDDING_MODEL", None)
    dim = getattr(assigner_module, "MODEL_EMBEDDING_DIM", None)
    payload = {
        "model": model_name,
        "dim": dim,
        "count": len(texts),
        "texts": texts,
    }
    signature = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    os.makedirs(os.path.join(BASE_DIR, "cache"), exist_ok=True)
    emb_path = embedding_cache_path(f"{cache_name}.npy")
    meta_path = embedding_cache_path(f"{cache_name}_meta.json")

    if os.path.exists(emb_path) and os.path.exists(meta_path):
        try:
            meta = load_json(meta_path)
            embeddings = np.load(emb_path)
            if (
                meta.get("signature") == signature
                and meta.get("count") == len(texts)
                and len(embeddings) == len(texts)
            ):
                return np.asarray(embeddings, dtype="float32")
        except Exception:
            pass

    embeddings = assigner_module.bi_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    try:
        np.save(emb_path, embeddings)
        save_json(
            meta_path,
            {
                "signature": signature,
                "count": len(texts),
                "dim": dim,
                "model": model_name,
            },
        )
    except Exception:
        pass
    return embeddings


def text_to_tokens(text):
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split()
        if len(token) > 3
    }


def minmax(values):
    values = list(values)
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def should_cross_rerank(row):
    if len(row) < 5:
        return True
    margin = float(row[0].get("score", 0.0) or 0.0) - float(row[4].get("score", 0.0) or 0.0)
    threshold = float(getattr(assigner_module, "CROSS_MARGIN_THRESHOLD", 0.035))
    return margin < threshold


def gini(values):
    values = sorted(float(v) for v in values)
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    weighted = sum((2 * idx - n - 1) * value for idx, value in enumerate(values, start=1))
    return weighted / (n * sum(values))


def average_precision(ranking, truth):
    truth = set(map(str, truth))
    if not truth:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, reviewer_id in enumerate(ranking, start=1):
        if str(reviewer_id) in truth:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / len(truth)


def ndcg_at_k(ranking, truth, k=K):
    truth = set(map(str, truth))
    gains = [1.0 if str(reviewer_id) in truth else 0.0 for reviewer_id in ranking[:k]]
    dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(ranking, truth):
    truth = set(map(str, truth))
    for idx, reviewer_id in enumerate(ranking, start=1):
        if str(reviewer_id) in truth:
            return 1.0 / idx
    return 0.0


def per_paper_scores(rankings, ground_truth, k=K):
    rows = []
    for paper_id, truth in ground_truth.items():
        if paper_id not in rankings:
            continue
        ranking = [str(item["reviewer_id"]) for item in rankings[paper_id]]
        top_k = ranking[:k]
        hits = len(set(top_k) & set(map(str, truth)))
        rows.append(
            {
                "paper_id": paper_id,
                f"precision_at_{k}": hits / k,
                f"recall_at_{k}": hits / len(truth) if truth else 0.0,
                "mrr": reciprocal_rank(ranking, truth),
                f"ndcg_at_{k}": ndcg_at_k(ranking, truth, k),
                "average_precision": average_precision(ranking, truth),
            }
        )
    return rows


def aggregate_metrics(rankings, ground_truth, k=K):
    rows = per_paper_scores(rankings, ground_truth, k)
    if not rows:
        return {
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            "mrr": 0.0,
            f"ndcg_at_{k}": 0.0,
            "map": 0.0,
            "evaluated_papers": 0,
        }
    return {
        f"precision_at_{k}": statistics.mean(row[f"precision_at_{k}"] for row in rows),
        f"recall_at_{k}": statistics.mean(row[f"recall_at_{k}"] for row in rows),
        "mrr": statistics.mean(row["mrr"] for row in rows),
        f"ndcg_at_{k}": statistics.mean(row[f"ndcg_at_{k}"] for row in rows),
        "map": statistics.mean(row["average_precision"] for row in rows),
        "evaluated_papers": len(rows),
    }


def to_ranking(scores):
    return [
        {"reviewer_id": reviewer_id, "score": float(score)}
        for reviewer_id, score in sorted(scores, key=lambda item: item[1], reverse=True)[:RANK_DEPTH]
    ]


def rank_positions(items, key):
    ranked = sorted(items, key=lambda row: float(key(row) or 0.0), reverse=True)
    return {str(row["reviewer_id"]): rank for rank, row in enumerate(ranked, start=1)}


def normalized_rrf(*ranks, k=60):
    usable = [int(rank) for rank in ranks if rank]
    if not usable:
        return 0.0
    raw = sum(1.0 / (k + rank) for rank in usable)
    best = len(usable) * (1.0 / (k + 1))
    return raw / max(best, 1e-12)


def rank_tfidf(papers, reviewers):
    paper_ids = list(papers.keys())
    reviewer_ids = [str(reviewer["reviewer_id"]) for reviewer in reviewers]
    paper_texts = [build_paper_text(papers[paper_id]) for paper_id in paper_ids]
    reviewer_texts = [build_reviewer_text(reviewer) for reviewer in reviewers]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(paper_texts + reviewer_texts)
    paper_matrix = matrix[: len(paper_texts)]
    reviewer_matrix = matrix[len(paper_texts) :]
    similarities = cosine_similarity(paper_matrix, reviewer_matrix)

    rankings = {}
    for row_idx, paper_id in enumerate(paper_ids):
        scores = zip(reviewer_ids, similarities[row_idx])
        rankings[paper_id] = to_ranking(scores)
    return rankings


def rank_embedding(papers, reviewers):
    rankings = {}
    for paper_id, paper in papers.items():
        ranked = rank_reviewers_for_paper(paper, reviewers, candidate_k=RANK_DEPTH, scoring_mode="semantic")
        rankings[paper_id] = [
            {"reviewer_id": str(item["reviewer_id"]), "score": float(item.get("similarity_score", 0.0))}
            for item in sorted(ranked, key=lambda row: row.get("similarity_score", 0.0), reverse=True)
        ][:RANK_DEPTH]
    return rankings


def rank_live_once(papers, reviewers):
    rankings = {}
    total = len(papers)
    stage_start = time.perf_counter()
    for idx, (paper_id, paper) in enumerate(papers.items(), start=1):
        ranked = rank_reviewers_for_paper(
            paper,
            reviewers,
            candidate_k=RANK_DEPTH,
            scoring_mode=RANKING_SCORING_MODE,
        )
        rankings[paper_id] = [
            {
                "reviewer_id": str(item["reviewer_id"]),
                "score": float(item.get("final_score", 0.0) or 0.0),
                "similarity_score": float(item.get("similarity_score", 0.0) or 0.0),
                "kg_score": float(item.get("kg_score", 0.0) or 0.0),
                "kg_effective": float(item.get("score_debug", {}).get("kg_effective", 0.0) or 0.0),
                "kg_confidence": float(item.get("kg_confidence", 0.0) or 0.0),
                "authority_score": float(item.get("authority_score", 0.0) or 0.0),
                "llm_suitability_score": float(item.get("llm_suitability_score", item.get("llm_match_score", 0.0)) or 0.0),
                "llm_match_score": float(item.get("llm_match_score", 0.0) or 0.0),
                "keyword_overlap_capped": float(item.get("keyword_overlap_capped", 0.0) or 0.0),
                "cross_score": float(item.get("cross_score", 0.0) or 0.0),
                "rrf_score": float(item.get("rrf_score", 0.0) or 0.0),
            }
            for item in ranked[:RANK_DEPTH]
        ]
        if PROGRESS_EVERY > 0 and (idx == 1 or idx % PROGRESS_EVERY == 0 or idx == total):
            elapsed = time.perf_counter() - stage_start
            per_paper = elapsed / max(1, idx)
            remaining = per_paper * max(0, total - idx)
            print(
                f"Live ranking {idx}/{total} papers "
                f"elapsed={elapsed/60:.1f}m remaining~{remaining/60:.1f}m",
                flush=True,
            )
    return rankings


def rank_embedding_from_live(live_rankings):
    rankings = {}
    for paper_id, ranked in live_rankings.items():
        rankings[paper_id] = [
            {"reviewer_id": item["reviewer_id"], "score": item.get("similarity_score", 0.0)}
            for item in sorted(ranked, key=lambda row: row.get("similarity_score", 0.0), reverse=True)
        ]
    return rankings


def citation_similarity_score(paper_tokens, reviewer):
    weighted = 0.0
    total_weight = 0.0
    for pub_tokens, citations in reviewer.get("_citation_features", []):
        if not pub_tokens:
            continue
        overlap = len(paper_tokens & pub_tokens) / len(paper_tokens | pub_tokens) if paper_tokens else 0.0
        weight = math.log1p(float(citations or 0))
        weighted += overlap * max(weight, 1.0)
        total_weight += max(weight, 1.0)
    return weighted / total_weight if total_weight else 0.0


def rank_citation_similarity(papers, reviewers):
    rankings = {}
    reviewer_features = []
    for reviewer in reviewers:
        if "_citation_features" not in reviewer:
            reviewer["_citation_features"] = [
                (
                    text_to_tokens(" ".join([publication.get("title", ""), publication.get("abstract", "")])),
                    publication.get("citations", 0),
                )
                for publication in reviewer.get("publications", [])[:20]
            ]
        reviewer_features.append((str(reviewer["reviewer_id"]), reviewer))

    for paper_id, paper in papers.items():
        paper_tokens = text_to_tokens(build_paper_text(paper))
        scores = []
        for reviewer_id, reviewer in reviewer_features:
            scores.append((reviewer_id, citation_similarity_score(paper_tokens, reviewer)))
        rankings[paper_id] = to_ranking(scores)
    return rankings


def rank_ablation_from_live(papers, live_rankings, mode, coi_lookup):
    rankings = {}
    for paper_id, ranked in live_rankings.items():
        semantic_rank = rank_positions(ranked, lambda row: row.get("similarity_score", 0.0))
        authority_rank = rank_positions(ranked, lambda row: row.get("authority_score", 0.0))
        kg_rank = rank_positions(ranked, lambda row: row.get("kg_effective", row.get("kg_score", 0.0)))
        final_rank = rank_positions(ranked, lambda row: row.get("score", 0.0))
        scores = []
        for item in ranked:
            reviewer_id = str(item["reviewer_id"])
            semantic = float(item.get("similarity_score", 0.0) or 0.0)
            authority = float(item.get("authority_score", 0.0) or 0.0)
            kg = float(item.get("kg_effective", item.get("kg_score", 0.0)) or 0.0)
            llm = float(item.get("llm_suitability_score", item.get("llm_match_score", 0.0)) or 0.0)
            final_score = float(item.get("score", 0.0) or 0.0)
            coi = coi_lookup.get((paper_id, reviewer_id), {})
            coi_flagged = bool(coi.get("flagged")) or float(coi.get("coi_score", 0.0) or 0.0) >= 0.5
            has_cross = "cross_score" in item
            cross = float(item.get("cross_score", 0.0) or 0.0)

            if mode == "semantic_only":
                score = semantic
            elif mode == "semantic_authority":
                score = 0.70 * semantic + 0.30 * authority
            elif mode == "semantic_kg":
                score = 0.50 * semantic + 0.50 * kg
            elif mode == "semantic_llm":
                score = 0.50 * semantic + 0.50 * llm
            elif mode == "semantic_kg_llm":
                score = 0.34 * semantic + 0.33 * kg + 0.33 * llm
            elif mode == "semantic_kg_authority":
                score = 0.75 * semantic + 0.20 * kg + 0.05 * authority
            elif mode == "rrf_fusion":
                score = normalized_rrf(
                    semantic_rank.get(reviewer_id),
                    authority_rank.get(reviewer_id),
                    kg_rank.get(reviewer_id),
                    final_rank.get(reviewer_id),
                )
            elif mode == "coi_filtering":
                if coi_flagged:
                    continue
                score = semantic
            else:
                semantic_kg_score = 0.34 * semantic + 0.33 * kg + 0.33 * llm
                if has_cross:
                    score = semantic_kg_score if cross <= 0 else 0.3 * semantic_kg_score + 0.7 * cross
                else:
                    score = semantic_kg_score
            scores.append((reviewer_id, score))

        rankings[paper_id] = to_ranking(scores)
    return rankings


def selected_assignments_from_rankings(rankings, reviewers=None, top_n=K, load_penalty=0.02):
    loads = Counter()
    assignments = {}
    reviewer_lookup = {
        str(reviewer.get("reviewer_id")): reviewer
        for reviewer in reviewers or []
    }
    for paper_id, ranked in rankings.items():
        selected = []
        selected_set = set()
        for _ in range(top_n):
            best = None
            best_score = float("-inf")
            for item in ranked:
                reviewer_id = str(item["reviewer_id"])
                if reviewer_id in selected_set:
                    continue
                reviewer = reviewer_lookup.get(reviewer_id, item)
                status = reviewer_capacity_status(reviewer, loads[reviewer_id])
                if status["hard_excluded"]:
                    continue
                availability = max(0.0, 1.0 - float(status.get("load_ratio", 0.0) or 0.0))
                score = float(item.get("score", 0.0) or 0.0) * availability
                if score > best_score:
                    best = reviewer_id
                    best_score = score
            if best is None:
                break
            selected.append(best)
            selected_set.add(best)
            loads[best] += 1
        assignments[paper_id] = selected
    return assignments


def workload_fairness(assignments, all_reviewer_ids, reviewers=None):
    reviewer_lookup = {
        str(reviewer.get("reviewer_id")): reviewer
        for reviewer in reviewers or []
    }
    counts = Counter()
    for reviewer_ids in assignments.values():
        counts.update(map(str, reviewer_ids))
    loads = [counts.get(str(reviewer_id), 0) for reviewer_id in all_reviewer_ids]
    capacities = {
        str(reviewer_id): reviewer_capacity_status(reviewer_lookup.get(str(reviewer_id), {"reviewer_id": reviewer_id}), counts.get(str(reviewer_id), 0))
        for reviewer_id in all_reviewer_ids
    }
    mean = statistics.mean(loads) if loads else 0.0
    variance = statistics.pvariance(loads) if loads else 0.0
    return {
        "total_reviewers": len(loads),
        "assigned_reviewers": sum(1 for load in loads if load > 0),
        "total_assignments": sum(loads),
        "avg_load": mean,
        "variance": variance,
        "std_load": math.sqrt(variance),
        "min_load": min(loads) if loads else 0,
        "max_load": max(loads) if loads else 0,
        "gini": gini(loads),
        "near_capacity_reviewers": sum(1 for status in capacities.values() if status["capacity_status"] == "near_capacity"),
        "at_capacity_reviewers": sum(1 for status in capacities.values() if status["capacity_status"] == "at_capacity"),
        "capacity_policy": "junior=6, mid=10, senior=12-15",
        "capacity_by_reviewer": {reviewer_id: status["capacity"] for reviewer_id, status in capacities.items()},
        "assignment_counts": dict(sorted(counts.items())),
    }


def conflict_detection_eval(coi_rows, threshold=0.5):
    tp = fp = tn = fn = 0
    labeled = []
    for row in coi_rows:
        label = 1 if bool(row.get("flagged")) else 0
        pred = 1 if float(row.get("coi_score", 0.0) or 0.0) >= threshold else 0
        labeled.append({"paper_id": str(row.get("paper_id")), "reviewer_id": str(row.get("reviewer_id")), "label": label, "prediction": pred})
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "labels_source": "data/coi_graph.json flagged field",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "evaluated_pairs": len(labeled),
    }, labeled


def format_float(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_markdown(summary_path, metric_rows, ablation_rows, runtime_rows, fairness, conflict, significance):
    lines = [
        "# Evaluation Results",
        "",
        "## Final Metric Table",
        "",
        "| Method | P@3 | Recall@3 | MRR | NDCG@3 | MAP | Papers |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['method']} | {row['precision_at_3']:.4f} | {row['recall_at_3']:.4f} | "
            f"{row['mrr']:.4f} | {row['ndcg_at_3']:.4f} | {row['map']:.4f} | {row['evaluated_papers']} |"
        )

    lines += [
        "",
        "## Ablation Results",
        "",
        "| Version | P@3 | MRR | NDCG@3 | MAP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['version']} | {row['precision_at_3']:.4f} | {row['mrr']:.4f} | "
            f"{row['ndcg_at_3']:.4f} | {row['map']:.4f} |"
        )

    lines += [
        "",
        "## Runtime Results",
        "",
        "| Stage | Time seconds |",
        "|---|---:|",
    ]
    for row in runtime_rows:
        lines.append(f"| {row['stage']} | {row['time_seconds']:.4f} |")

    lines += [
        "",
        "## Fairness Values",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Avg load | {fairness['avg_load']:.4f} |",
        f"| Variance | {fairness['variance']:.4f} |",
        f"| Gini | {fairness['gini']:.4f} |",
        f"| Min load | {fairness['min_load']} |",
        f"| Max load | {fairness['max_load']} |",
        "",
        "## Conflict Detection Evaluation",
        "",
        "| Precision | Recall | F1 | TP | FP | TN | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {conflict['precision']:.4f} | {conflict['recall']:.4f} | {conflict['f1']:.4f} | "
            f"{conflict['tp']} | {conflict['fp']} | {conflict['tn']} | {conflict['fn']} |"
        ),
        "",
        "## Statistical Significance",
        "",
        "| Comparison | t statistic | p value | Significant p<0.05 |",
        "|---|---:|---:|---|",
    ]
    for item in significance:
        lines.append(
            f"| {item['comparison']} | {item['t_statistic']:.4f} | {item['p_value']:.6f} | {item['significant_p_lt_0_05']} |"
        )

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    manuscripts = load_json(os.path.join(DATA_DIR, "manuscripts.json"))
    reviewers = load_json(os.path.join(DATA_DIR, "reviewers.json"))
    ground_truth_rows = load_json(os.path.join(DATA_DIR, "ground_truth.json"))
    coi_rows = load_json(os.path.join(DATA_DIR, "coi_graph.json"))

    manuscript_map = {str(item["paper_id"]): item for item in manuscripts}
    ground_truth = {
        str(item["paper_id"]): [str(reviewer_id) for reviewer_id in item.get("assigned_reviewers", [])]
        for item in ground_truth_rows
        if str(item.get("paper_id")) in manuscript_map and item.get("assigned_reviewers")
    }
    eval_papers = {paper_id: manuscript_map[paper_id] for paper_id in ground_truth}
    if EVAL_LIMIT > 0:
        limited_ids = list(eval_papers.keys())[:EVAL_LIMIT]
        eval_papers = {paper_id: eval_papers[paper_id] for paper_id in limited_ids}
        ground_truth = {paper_id: ground_truth[paper_id] for paper_id in limited_ids}
    coi_lookup = {(str(row["paper_id"]), str(row["reviewer_id"])): row for row in coi_rows}
    print(
        f"Evaluating {len(eval_papers)} papers "
        f"(RANK_DEPTH={RANK_DEPTH}, scoring={RANKING_SCORING_MODE}, "
        f"USE_KG={assigner_module.USE_KG}, USE_NODE2VEC_KG={assigner_module.USE_NODE2VEC_KG}, "
        f"USE_CROSS={assigner_module.USE_CROSS})",
        flush=True,
    )

    timings = []
    start = time.perf_counter()
    prepare_reviewers_for_ranking(reviewers)
    timings.append({"stage": "Reviewer preparation", "time_seconds": time.perf_counter() - start})
    save_json(os.path.join(RESULTS_DIR, "retrieval_model_info.json"), get_reviewer_cache_info())

    start = time.perf_counter()
    tfidf_rankings = rank_tfidf(eval_papers, reviewers)
    timings.append({"stage": "TF-IDF similarity", "time_seconds": time.perf_counter() - start})

    start = time.perf_counter()
    live_rankings = rank_live_once(eval_papers, reviewers)
    save_json(os.path.join(RESULTS_DIR, "live_rankings_with_features.json"), live_rankings)
    timings.append({"stage": "Live scoring/ranking", "time_seconds": time.perf_counter() - start})

    start = time.perf_counter()
    embedding_rankings = rank_embedding_from_live(live_rankings)
    semantic_rankings = rank_ablation_from_live(eval_papers, live_rankings, "semantic_only", coi_lookup)
    semantic_authority_rankings = rank_ablation_from_live(eval_papers, live_rankings, "semantic_authority", coi_lookup)
    semantic_kg_rankings = rank_ablation_from_live(eval_papers, live_rankings, "semantic_kg", coi_lookup)
    semantic_llm_rankings = rank_ablation_from_live(eval_papers, live_rankings, "semantic_llm", coi_lookup)
    semantic_kg_llm_rankings = rank_ablation_from_live(eval_papers, live_rankings, "semantic_kg_llm", coi_lookup)
    semantic_kg_authority_rankings = rank_ablation_from_live(eval_papers, live_rankings, "semantic_kg_authority", coi_lookup)
    rrf_rankings = rank_ablation_from_live(eval_papers, live_rankings, "rrf_fusion", coi_lookup)
    coi_rankings = rank_ablation_from_live(eval_papers, live_rankings, "coi_filtering", coi_lookup)
    proposed_rankings = rank_ablation_from_live(eval_papers, live_rankings, "full_system", coi_lookup)
    timings.append({"stage": "Ablation and proposed ranking", "time_seconds": time.perf_counter() - start})

    full_pipeline_time = sum(row["time_seconds"] for row in timings)
    timings.append({"stage": "Full pipeline", "time_seconds": full_pipeline_time})

    ranking_export = []
    for paper_id, ranked in proposed_rankings.items():
        ranking_export.append(
            {
                "paper_id": paper_id,
                "predicted_rankings": [item["reviewer_id"] for item in ranked],
                "predicted_scores": [item["score"] for item in ranked],
                "ground_truth": ground_truth[paper_id],
            }
        )
    save_json(os.path.join(RESULTS_DIR, "results.json"), ranking_export)

    all_rankings = {
        "tfidf": tfidf_rankings,
        "embedding": embedding_rankings,
        "proposed": proposed_rankings,
        "semantic_only": semantic_rankings,
        "semantic_authority": semantic_authority_rankings,
        "semantic_kg": semantic_kg_rankings,
        "semantic_llm": semantic_llm_rankings,
        "semantic_kg_llm": semantic_kg_llm_rankings,
        "semantic_kg_authority": semantic_kg_authority_rankings,
        "rrf_fusion": rrf_rankings,
        "coi_filtering": coi_rankings,
    }
    save_json(os.path.join(RESULTS_DIR, "rankings_by_method.json"), all_rankings)

    metric_rows = []
    for method, rankings in [
        ("TF-IDF", tfidf_rankings),
        ("Embedding", embedding_rankings),
        ("Proposed", proposed_rankings),
    ]:
        metrics = aggregate_metrics(rankings, ground_truth, K)
        metric_rows.append({"method": method, **metrics})

    ablation_rows = []
    for version, rankings in [
        ("Base semantic only", semantic_rankings),
        ("Semantic + authority", semantic_authority_rankings),
        ("Semantic + KG", semantic_kg_rankings),
        ("Semantic + LLM Profile", semantic_llm_rankings),
        ("Semantic + KG + LLM Profile", semantic_kg_llm_rankings),
        ("Semantic + KG + light authority", semantic_kg_authority_rankings),
        ("Reciprocal rank fusion", rrf_rankings),
        ("COI filtering", coi_rankings),
        ("Full system", proposed_rankings),
    ]:
        metrics = aggregate_metrics(rankings, ground_truth, K)
        ablation_rows.append({"version": version, **metrics})

    proposed_assignments = selected_assignments_from_rankings(proposed_rankings, reviewers, K)
    fairness = workload_fairness(proposed_assignments, [str(reviewer["reviewer_id"]) for reviewer in reviewers], reviewers)
    conflict, conflict_labels = conflict_detection_eval(coi_rows, threshold=0.5)

    method_score_rows = {
        "Proposed": {row["paper_id"]: row for row in per_paper_scores(proposed_rankings, ground_truth, K)},
        "TF-IDF": {row["paper_id"]: row for row in per_paper_scores(tfidf_rankings, ground_truth, K)},
        "Embedding": {row["paper_id"]: row for row in per_paper_scores(embedding_rankings, ground_truth, K)},
        "Semantic only": {row["paper_id"]: row for row in per_paper_scores(semantic_rankings, ground_truth, K)},
        "Semantic + KG": {row["paper_id"]: row for row in per_paper_scores(semantic_kg_rankings, ground_truth, K)},
    }
    significance = []
    comparisons = [
        ("Proposed vs TF-IDF", "Proposed", "TF-IDF"),
        ("Proposed vs Embedding", "Proposed", "Embedding"),
        ("Semantic + KG vs Semantic only", "Semantic + KG", "Semantic only"),
        ("Proposed vs Semantic + KG", "Proposed", "Semantic + KG"),
    ]
    for name, left_name, right_name in comparisons:
        left_by_pid = method_score_rows[left_name]
        right_by_pid = method_score_rows[right_name]
        common_ids = sorted(set(left_by_pid) & set(right_by_pid))
        left_ap = [left_by_pid[paper_id]["average_precision"] for paper_id in common_ids]
        right_ap = [right_by_pid[paper_id]["average_precision"] for paper_id in common_ids]
        if len(common_ids) >= 2:
            test = ttest_rel(left_ap, right_ap)
            t_stat = float(test.statistic) if not math.isnan(float(test.statistic)) else 0.0
            p_value = float(test.pvalue) if not math.isnan(float(test.pvalue)) else 1.0
        else:
            t_stat = 0.0
            p_value = 1.0
        significance.append(
            {
                "comparison": name,
                "metric": "per-paper average precision",
                "paired_samples": len(common_ids),
                "left_method": left_name,
                "right_method": right_name,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_p_lt_0_05": p_value < 0.05,
            }
        )

    save_json(os.path.join(RESULTS_DIR, "final_metric_table.json"), metric_rows)
    save_json(os.path.join(RESULTS_DIR, "ablation_results.json"), ablation_rows)
    save_json(os.path.join(RESULTS_DIR, "runtime_results.json"), timings)
    save_json(os.path.join(RESULTS_DIR, "fairness_values.json"), fairness)
    save_json(os.path.join(RESULTS_DIR, "conflict_detection_evaluation.json"), conflict)
    save_json(os.path.join(RESULTS_DIR, "conflict_detection_labels.json"), conflict_labels)
    save_json(os.path.join(RESULTS_DIR, "statistical_significance.json"), significance)

    write_csv(
        os.path.join(RESULTS_DIR, "final_metric_table.csv"),
        metric_rows,
        ["method", "precision_at_3", "recall_at_3", "mrr", "ndcg_at_3", "map", "evaluated_papers"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "ablation_results.csv"),
        ablation_rows,
        ["version", "precision_at_3", "recall_at_3", "mrr", "ndcg_at_3", "map", "evaluated_papers"],
    )
    write_csv(os.path.join(RESULTS_DIR, "runtime_results.csv"), timings, ["stage", "time_seconds"])

    fairness_rows = [
        {"metric": key, "value": value}
        for key, value in fairness.items()
        if key != "assignment_counts"
    ]
    write_csv(os.path.join(RESULTS_DIR, "fairness_values.csv"), fairness_rows, ["metric", "value"])
    write_csv(
        os.path.join(RESULTS_DIR, "conflict_detection_evaluation.csv"),
        [{key: value for key, value in conflict.items() if key != "labels_source"}],
        ["threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "evaluated_pairs"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "statistical_significance.csv"),
        significance,
        [
            "comparison",
            "metric",
            "paired_samples",
            "left_method",
            "right_method",
            "t_statistic",
            "p_value",
            "significant_p_lt_0_05",
        ],
    )

    write_markdown(
        os.path.join(RESULTS_DIR, "evaluation_summary.md"),
        metric_rows,
        ablation_rows,
        timings,
        fairness,
        conflict,
        significance,
    )

    print(f"Saved evaluation artifacts to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
