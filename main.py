# ==========================================
# FINAL Reviewer Recommendation PIPELINE (CLEAN VERSION)
# ==========================================

import time
import json
import os
import re
import random

try:
    import numpy as np
except Exception:
    np = None

from config import (
    MANUSCRIPTS_PATH,
    REVIEWERS_PATH,
    GROUND_TRUTH,
    COI_PATH,
    CSO_MS_PATH,
    CSO_REV_PATH,
    GLINER_MS_PATH,
    GLINER_REV_PATH,
    EVAL_K,
    ASSIGN_K,
    TOP_K,
    TEST_MODE,
    TEST_PAPERS,
    USE_CROSS,
    USE_KG,
    RANKING_SCORING_MODE,
)

from assignment.assigner import get_kg_cache_info, get_reviewer_cache_info, prepare_reviewers_for_ranking, rank_reviewers_for_paper
from assignment.assigner import classify_reviewer, compute_stats, reviewer_capacity_status, select_role_balanced_reviewers


# -------------------------------
# LOAD JSON
# -------------------------------
def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# -------------------------------
# LOAD DATA
# -------------------------------
def load_all():
    manuscripts = load_json(MANUSCRIPTS_PATH)
    reviewers   = load_json(REVIEWERS_PATH)
    gt          = load_json(GROUND_TRUTH)
    coi         = load_json(COI_PATH)

    ms_dict = {m["paper_id"]: m for m in manuscripts}
    rev_dict = {r["reviewer_id"]: r for r in reviewers}
    gt_dict = {g["paper_id"]: g["assigned_reviewers"] for g in gt}
    coi_dict = {(c["paper_id"], c["reviewer_id"]): c for c in coi}

    return ms_dict, rev_dict, gt_dict, coi_dict


# -------------------------------
# METRICS
# -------------------------------
def compute_mrr(fusion, gt):

    total = 0
    count = 0

    for pid, gt_reviewers in gt.items():

        if pid not in fusion:
            continue

        ranked = fusion[pid]

        for i, r in enumerate(ranked[:10], 1):
            if r["reviewer_id"] in gt_reviewers:
                total += 1 / i
                break

        count += 1

    return total / count if count else 0


def precision_at_k(fusion, gt, k):
    total = 0
    count = 0

    for pid, gt_reviewers in gt.items():

        if pid not in fusion:
            continue

        top_k = [r["reviewer_id"] for r in fusion[pid][:k]]

        hits = len(set(top_k) & set(gt_reviewers))
        total += hits / k
        count += 1

    return total / count if count else 0


def recall_at_k(fusion, gt, k):
    total = 0
    count = 0

    for pid, gt_reviewers in gt.items():

        if pid not in fusion:
            continue

        top_k = [r["reviewer_id"] for r in fusion[pid][:k]]

        hits = len(set(top_k) & set(gt_reviewers))
        total += hits / len(gt_reviewers)
        count += 1

    return total / count if count else 0


def coverage(fusion, total_reviewers):
    unique = set()

    for ranked in fusion.values():
        for r in ranked:
            unique.add(str(r["reviewer_id"]))

    return len(unique) / total_reviewers if total_reviewers else 0


def diversity(fusion):
    unique = set()
    total = 0

    for lst in fusion.values():
        total += len(lst)
        for r in lst:
            unique.add(str(r["reviewer_id"]))

    return len(unique) / total if total else 0


def recall_at_100(paper_id, ranked_reviewers, gt):
    top_100 = [str(r["reviewer_id"]) for r in ranked_reviewers[:100]]
    gt_ids = [str(x) for x in gt.get(str(paper_id), [])]
    return len(set(top_100) & set(gt_ids)) > 0


def hit_rate_at_k(fusion, gt, k):
    hits = 0
    count = 0

    for pid, gt_reviewers in gt.items():
        if pid not in fusion:
            continue

        top_k = [str(r["reviewer_id"]) for r in fusion[pid][:k]]
        gt_ids = [str(x) for x in gt_reviewers]
        hits += 1 if set(top_k) & set(gt_ids) else 0
        count += 1

    return hits / count if count else 0.0


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    random.seed(42)
    if np is not None:
        np.random.seed(42)

    start = time.time()

    print("\n=== FINAL Reviewer Recommendation PIPELINE (CLEAN + HYBRID) ===")
    print("Fast settings:", {
        "USE_CROSS": USE_CROSS,
        "USE_KG": USE_KG,
        "SCORING_MODE": RANKING_SCORING_MODE,
        "TOP_K": TOP_K,
        "TEST_MODE": TEST_MODE,
    })

    # Load data
    ms_dict, rev_dict, gt, coi = load_all()
    print("Ignoring cached similarity_combined.json for ranking; computing live similarities this run.")

    # Load KG enrichment
    cso_ms = load_json(CSO_MS_PATH)
    cso_rev = load_json(CSO_REV_PATH)
    gliner_ms = load_json(GLINER_MS_PATH)
    gliner_rev = load_json(GLINER_REV_PATH)

    reviewers = list(rev_dict.values()) if isinstance(rev_dict, dict) else list(rev_dict or [])
    prep_start = time.time()
    prepare_reviewers_for_ranking(reviewers)
    print("Reviewer preparation time:", round(time.time() - prep_start, 2), "sec")
    cache_info = get_reviewer_cache_info()
    print("Reviewer embedding cache:", cache_info.get("status"), cache_info.get("path", ""))
    kg_cache_info = get_kg_cache_info()
    print("KG Node2Vec cache before ranking:", kg_cache_info.get("status"), kg_cache_info)

    fusion = {}
    ranking_pids = [str(pid) for pid in gt.keys() if str(pid) in ms_dict]
    if TEST_MODE:
        ranking_pids = ranking_pids[:TEST_PAPERS]
    sample_pids = set(ranking_pids[:50])
    hits = []
    sample_ranking = None
    base_rankings = {}
    ranking_start = time.time()

    for idx, pid in enumerate(ranking_pids):
        ranked = rank_reviewers_for_paper(ms_dict[pid], reviewers, candidate_k=TOP_K, scoring_mode=RANKING_SCORING_MODE)
        base_rankings[pid] = ranked
        fusion[pid] = [
            {
                "reviewer_id": str(r["reviewer_id"]),
                "score": float(r.get("final_score", 0.0)),
                "similarity_score": float(r.get("similarity_score", 0.0)),
                "semantic_score": float(r.get("semantic_score", r.get("similarity_score", 0.0)) or 0.0),
                "kg_score": float(r.get("kg_score", 0.0) or 0.0),
                "cross_score": float(r.get("cross_score", r.get("similarity_score", 0.0)) or 0.0),
                "authority_score": float(r.get("authority_score", 0.0)),
                "rank_signal": float(r.get("rank_signal", 0.0) or 0.0),
            }
            for r in ranked
        ]

        if pid in sample_pids:
            hits.append(recall_at_100(pid, ranked, gt))
            if sample_ranking is None:
                sample_ranking = ranked

        if idx == 0:
            print("Top10:", [r["reviewer_id"] for r in fusion[pid][:10]])
            print("GT:", gt.get(pid, []))

    print("Ranking time:", round(time.time() - ranking_start, 2), "sec")
    kg_cache_info = get_kg_cache_info()
    print("KG Node2Vec cache after ranking:", kg_cache_info.get("status"), kg_cache_info)
    recall100 = (sum(hits) / len(hits)) if hits else 0.0
    print("Recall@100 (hit rate):", round(recall100, 4))
    if sample_ranking:
        print("SIM sample:", sample_ranking[0]["similarity_score"])

    # -------------------------------
    # Create diversity-aware assignments across all papers
    # -------------------------------
    ASSIGN_PATH = os.path.join(os.path.dirname(__file__), "outputs", "assignments.json")
    METRICS_PATH = os.path.join(os.path.dirname(__file__), "outputs", "metrics.json")

    def _reviewer_signature_tokens(reviewer):
        text = str(reviewer.get("reviewer_text", "") or "").lower()
        tokens = {
            token for token in re.findall(r"[a-z0-9]{4,}", text)
            if token not in {"with", "from", "using", "based", "their", "these", "this"}
        }
        return tokens

    def _conflicts_with_selected(candidate, selected, min_overlap=0.6):
        candidate_tokens = _reviewer_signature_tokens(candidate)
        if not candidate_tokens:
            return False

        for chosen in selected:
            chosen_tokens = _reviewer_signature_tokens(chosen)
            if not chosen_tokens:
                continue

            overlap_ratio = len(candidate_tokens & chosen_tokens) / max(
                1,
                min(len(candidate_tokens), len(chosen_tokens)),
            )
            if overlap_ratio >= min_overlap:
                return True

        return False

    def create_assignments_with_diversity(paper_ids, ranked_map, top_n=3, penalty_step=0.02):
        reviewer_usage = {}
        reviewer_capacity_map = {}
        assignments = {}

        for pid in paper_ids:
            ranked = ranked_map.get(str(pid))
            if not ranked:
                continue

            scored_ranked = []
            for reviewer in ranked:
                candidate = dict(reviewer)
                rid = str(candidate.get("reviewer_id"))
                status = reviewer_capacity_status(candidate, reviewer_usage.get(rid, 0))
                reviewer_capacity_map.setdefault(rid, status["capacity"])
                if status["hard_excluded"]:
                    continue
                load_ratio = float(status.get("load_ratio", 0.0) or 0.0)
                penalty = penalty_step * reviewer_usage.get(rid, 0) * (1.0 + load_ratio)
                candidate["final_score"] = float(candidate.get("final_score", 0.0)) - penalty
                candidate["current_load"] = status["current_load"]
                candidate["capacity"] = status["capacity"]
                candidate["load_ratio"] = status["load_ratio"]
                candidate["capacity_status"] = status["capacity_status"]
                scored_ranked.append(candidate)

            if not scored_ranked:
                scored_ranked = [dict(reviewer) for reviewer in ranked[:top_n]]

            scored_ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            role_stats = compute_stats(scored_ranked)
            diversity_filtered = []
            for candidate in scored_ranked:
                if _conflicts_with_selected(candidate, diversity_filtered):
                    continue
                diversity_filtered.append(candidate)
                if len(diversity_filtered) == top_n:
                    break

            selection_pool = diversity_filtered if len(diversity_filtered) >= top_n else scored_ranked
            chosen = select_role_balanced_reviewers(selection_pool, role_stats, top_n=top_n)
            if len(chosen) < top_n:
                chosen = selection_pool[:top_n]

            assigned_reviewers = []
            for reviewer in chosen:
                rid = str(reviewer.get("reviewer_id"))
                reviewer_usage[rid] = reviewer_usage.get(rid, 0) + 1
                status = reviewer_capacity_status(reviewer, reviewer_usage.get(rid, 0))
                reviewer_capacity_map[rid] = status["capacity"]
                assigned_reviewers.append(
                    {
                        "reviewer_id": rid,
                        "score": float(reviewer.get("final_score", 0.0)),
                        "h_index": reviewer.get("h_index", reviewer.get("h_index_proxy", reviewer.get("hindex", 0))),
                        "role": classify_reviewer(reviewer),
                        "capacity": status["capacity"],
                        "current_load": status["current_load"],
                        "load_ratio": round(status["load_ratio"], 4),
                        "capacity_status": status["capacity_status"],
                    }
                )

            assignments[str(pid)] = assigned_reviewers
            if len(assignments) == 1:
                print("Assignment sample:", assignments[str(pid)])
                print([classify_reviewer(reviewer) for reviewer in assignments[str(pid)]])
            print(pid, len(assignments[str(pid)]))

        overloaded = sum(1 for rid, load in reviewer_usage.items() if load >= reviewer_capacity_map.get(rid, 1))
        near_capacity = sum(
            1
            for rid, load in reviewer_usage.items()
            if 0.0 < load / max(1, reviewer_capacity_map.get(rid, 1)) >= 0.90
        )
        print(
            "Dynamic capacity summary:",
            {
                "assigned_reviewers": len(reviewer_usage),
                "near_capacity_reviewers": near_capacity,
                "at_capacity_reviewers": overloaded,
                "capacity_policy": "junior=6, mid=10, senior=12-15",
            },
        )

        return assignments

    assignments = create_assignments_with_diversity(ranking_pids, base_rankings, top_n=ASSIGN_K, penalty_step=0.05)

    # save assignments for downstream evaluation and UI
    os.makedirs(os.path.dirname(ASSIGN_PATH), exist_ok=True)
    with open(ASSIGN_PATH, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2)

    print(f"Saved assignments -> {ASSIGN_PATH}")

    # -------------------------------
    # Assignment-level evaluation (precision/recall/coverage/diversity)
    # -------------------------------
    assigned_set = set()
    total_assigned = 0
    precision_sum = 0.0
    recall_sum = 0.0
    count_with_gt = 0

    for pid, roles in assignments.items():
        assigned_list = [str(item.get("reviewer_id")) for item in roles if item.get("reviewer_id")]
        total_assigned += len(assigned_list)
        assigned_set.update(assigned_list)

        gt_reviewers = gt.get(str(pid), []) if gt else []
        if gt_reviewers:
            hits = len(set(assigned_list) & set(map(str, gt_reviewers)))
            precision_sum += (hits / len(assigned_list)) if assigned_list else 0.0
            recall_sum += (hits / len(gt_reviewers)) if gt_reviewers else 0.0
            count_with_gt += 1

    assign_diversity = (len(assigned_set) / total_assigned) if total_assigned else 0.0
    assign_precision = (precision_sum / count_with_gt) if count_with_gt else 0.0
    assign_recall = (recall_sum / count_with_gt) if count_with_gt else 0.0
    total_reviewers_count = len(rev_dict) if isinstance(rev_dict, dict) else len(rev_dict or [])
    assign_coverage = (len(assigned_set) / total_reviewers_count) if total_reviewers_count else 0.0

    print("\nASSIGNMENT-LEVEL RESULTS")
    print("Assignment Precision:", round(assign_precision, 4))
    print("Assignment Recall:", round(assign_recall, 4))
    print("Assignment Coverage:", round(assign_coverage, 4))
    print("Assignment Diversity:", round(assign_diversity, 4))

    # -------------------------------
    # EVALUATION
    # -------------------------------
    mrr = compute_mrr(fusion, gt)
    p_k = precision_at_k(fusion, gt, EVAL_K)
    r_k = recall_at_k(fusion, gt, EVAL_K)
    recall10 = hit_rate_at_k(fusion, gt, 10)
    total_reviewers = len(rev_dict) if isinstance(rev_dict, dict) else len(rev_dict or [])
    cov = coverage(fusion, total_reviewers)
    div = diversity(fusion)

    print("\nRESULTS")
    print("MRR:", round(mrr, 4))
    print(f"Precision@{EVAL_K}:", round(p_k, 4))
    print(f"Recall@{EVAL_K}:", round(r_k, 4))
    print("Recall@10:", round(recall10, 4))
    print("Coverage:", cov)
    print("Diversity:", round(div, 4))

    metrics = {
        "mrr": round(mrr, 4),
        "precision_at_5": round(p_k, 4),
        "recall_at_5": round(r_k, 4),
        "recall_at_10": round(recall10, 4),
        "recall_at_100": round(recall100, 4),
        "coverage": round(cov, 4),
        "diversity": round(div, 4),
        "assignment_precision": round(assign_precision, 4),
        "assignment_recall": round(assign_recall, 4),
        "assignment_coverage": round(assign_coverage, 4),
        "assignment_diversity": round(assign_diversity, 4),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTime:", round(time.time() - start, 2), "sec")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()
