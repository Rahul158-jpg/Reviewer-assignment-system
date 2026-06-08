import csv
import json
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

LIVE_RANKINGS_PATH = os.path.join(RESULTS_DIR, "live_rankings_with_features.json")
RANKINGS_PATH = os.path.join(RESULTS_DIR, "rankings_by_method.json")
GT_PATH = os.path.join(BASE_DIR, "data", "ground_truth.json")
OUT_JSON = os.path.join(RESULTS_DIR, "fusion_weight_grid.json")
OUT_CSV = os.path.join(RESULTS_DIR, "fusion_weight_grid.csv")
K = int(os.getenv("FUSION_TUNE_K", "5"))
RANK_DEPTH = int(os.getenv("RANK_DEPTH", "100"))


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def normalize_ground_truth(raw):
    if isinstance(raw, list):
        return {
            str(row.get("paper_id")): [str(rid) for rid in row.get("assigned_reviewers", [])]
            for row in raw
            if row.get("paper_id") is not None and row.get("assigned_reviewers")
        }
    return {str(pid): [str(rid) for rid in reviewers] for pid, reviewers in raw.items()}


def reciprocal_rank(ranking, truth):
    truth = set(map(str, truth))
    for idx, reviewer_id in enumerate(ranking, start=1):
        if str(reviewer_id) in truth:
            return 1.0 / idx
    return 0.0


def average_precision(ranking, truth):
    truth = set(map(str, truth))
    if not truth:
        return 0.0
    hits = 0
    total = 0.0
    for idx, reviewer_id in enumerate(ranking, start=1):
        if str(reviewer_id) in truth:
            hits += 1
            total += hits / idx
    return total / len(truth)


def ndcg_at_k(ranking, truth, k):
    truth = set(map(str, truth))
    gains = [1.0 if str(reviewer_id) in truth else 0.0 for reviewer_id in ranking[:k]]
    dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def aggregate_metrics(rankings, ground_truth, k):
    rows = []
    for paper_id, truth in ground_truth.items():
        ranked = rankings.get(str(paper_id), [])
        if not ranked:
            continue
        ranking = [str(item["reviewer_id"]) for item in ranked]
        top_k = ranking[:k]
        truth_set = set(map(str, truth))
        hits = len(set(top_k) & truth_set)
        rows.append({
            f"precision_at_{k}": hits / k,
            f"recall_at_{k}": hits / len(truth_set) if truth_set else 0.0,
            "mrr": reciprocal_rank(ranking, truth_set),
            f"ndcg_at_{k}": ndcg_at_k(ranking, truth_set, k),
            "map": average_precision(ranking, truth_set),
        })
    if not rows:
        return {f"precision_at_{k}": 0.0, f"recall_at_{k}": 0.0, "mrr": 0.0, f"ndcg_at_{k}": 0.0, "map": 0.0}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    } | {"evaluated_papers": len(rows)}


def candidate_weight_grid():
    return [
        (0.90, 0.10, 0.00),
        (0.85, 0.15, 0.00),
        (0.80, 0.20, 0.00),
        (0.75, 0.20, 0.05),
        (0.70, 0.25, 0.05),
        (0.70, 0.20, 0.10),
        (0.65, 0.25, 0.10),
    ]


def load_component_rankings():
    if os.path.exists(LIVE_RANKINGS_PATH):
        return load_json(LIVE_RANKINGS_PATH)
    rankings = load_json(RANKINGS_PATH)
    semantic = rankings.get("semantic_only") or rankings.get("embedding") or {}
    semantic_kg = rankings.get("semantic_kg") or semantic
    component_rankings = {}
    for paper_id, ranked in semantic.items():
        kg_scores = {
            str(item["reviewer_id"]): float(item.get("score", 0.0) or 0.0)
            for item in semantic_kg.get(paper_id, [])
        }
        component_rankings[paper_id] = [
            {
                "reviewer_id": str(item["reviewer_id"]),
                "similarity_score": float(item.get("score", 0.0) or 0.0),
                "kg_effective": max(0.0, kg_scores.get(str(item["reviewer_id"]), 0.0) - float(item.get("score", 0.0) or 0.0)),
                "authority_score": 0.0,
            }
            for item in ranked
        ]
    return component_rankings


def fused_rankings(live_rankings, sem_w, kg_w, auth_w):
    fused = {}
    for paper_id, ranked in live_rankings.items():
        scores = []
        for row in ranked:
            semantic = float(row.get("similarity_score", 0.0) or 0.0)
            kg = float(row.get("kg_effective", row.get("kg_score", 0.0)) or 0.0)
            authority = float(row.get("authority_score", 0.0) or 0.0)
            score = sem_w * semantic + kg_w * kg + auth_w * authority
            scores.append({"reviewer_id": str(row["reviewer_id"]), "score": score})
        fused[paper_id] = sorted(scores, key=lambda item: item["score"], reverse=True)[:RANK_DEPTH]
    return fused


def main():
    ground_truth = normalize_ground_truth(load_json(GT_PATH))
    live_rankings = load_component_rankings()
    if not live_rankings:
        raise SystemExit("Missing rankings. Run tools\\run_full_evaluation.py first.")

    rows = []
    for sem_w, kg_w, auth_w in candidate_weight_grid():
        fused = fused_rankings(live_rankings, sem_w, kg_w, auth_w)
        metrics = aggregate_metrics(fused, ground_truth, K)
        rows.append({
            "semantic_weight": sem_w,
            "kg_weight": kg_w,
            "authority_weight": auth_w,
            **metrics,
        })

    rows.sort(key=lambda row: (row.get(f"ndcg_at_{K}", 0.0), row.get("map", 0.0), row.get("mrr", 0.0)), reverse=True)
    save_json(OUT_JSON, rows)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Best fusion weights:")
    for key, value in rows[0].items():
        print(f"  {key}: {value}")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
