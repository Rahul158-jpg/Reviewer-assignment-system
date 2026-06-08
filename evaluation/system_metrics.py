import csv
import json
import math
import os
import statistics
from collections import Counter

import numpy as np

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover - scipy is optional for this report
    ttest_rel = None
    wilcoxon = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SIM_PATH = os.path.join(OUTPUT_DIR, "similarity_combined.json")
GT_PATH = os.path.join(DATA_DIR, "ground_truth.json")
REVIEWERS_PATH = os.path.join(DATA_DIR, "reviewers.json")
ASSIGN_PATH = os.path.join(OUTPUT_DIR, "assignments.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
RANKINGS_PATH = os.path.join(RESULTS_DIR, "rankings_by_method.json")
FAIRNESS_PATH = os.path.join(RESULTS_DIR, "fairness_values.json")
SIGNIFICANCE_PATH = os.path.join(RESULTS_DIR, "statistical_significance.json")
REPORT_JSON = os.path.join(RESULTS_DIR, "system_metrics_report.json")
REPORT_CSV = os.path.join(RESULTS_DIR, "system_metrics_report.csv")

KS = (3, 5, 10)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_ground_truth(raw):
    if isinstance(raw, list):
        return {
            str(row.get("paper_id")): [str(rid) for rid in row.get("assigned_reviewers", [])]
            for row in raw
            if row.get("paper_id") is not None and row.get("assigned_reviewers")
        }
    if isinstance(raw, dict):
        return {str(pid): [str(rid) for rid in reviewers] for pid, reviewers in raw.items() if reviewers}
    return {}


def normalize_ranked(ranked):
    if not ranked:
        return []
    normalized = []
    for item in ranked:
        if isinstance(item, dict):
            rid = item.get("reviewer_id") or item.get("rid") or item.get("id")
            score = item.get("score", item.get("final", item.get("similarity", 0.0)))
        elif isinstance(item, (list, tuple)) and item:
            rid = item[0]
            score = item[1] if len(item) > 1 else 0.0
        else:
            rid = item
            score = 0.0
        if rid is not None:
            normalized.append({"reviewer_id": str(rid), "score": float(score or 0.0)})
    return normalized


def rankings_from_similarity(sim_data):
    return {str(pid): normalize_ranked(ranked) for pid, ranked in (sim_data or {}).items()}


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


def per_paper_scores(rankings, ground_truth, k):
    rows = []
    for paper_id, truth in ground_truth.items():
        ranked = normalize_ranked(rankings.get(str(paper_id), []))
        if not ranked:
            continue
        ranking = [row["reviewer_id"] for row in ranked]
        top_k = ranking[:k]
        truth_set = set(map(str, truth))
        hits = len(set(top_k) & truth_set)
        rows.append({
            "paper_id": str(paper_id),
            f"precision_at_{k}": hits / k,
            f"recall_at_{k}": hits / len(truth_set) if truth_set else 0.0,
            f"ndcg_at_{k}": ndcg_at_k(ranking, truth_set, k),
            "mrr": reciprocal_rank(ranking, truth_set),
            "average_precision": average_precision(ranking, truth_set),
        })
    return rows


def aggregate_ranking_metrics(rankings, ground_truth, k):
    rows = per_paper_scores(rankings, ground_truth, k)
    if not rows:
        return {
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"ndcg_at_{k}": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "evaluated_papers": 0,
        }
    return {
        f"precision_at_{k}": statistics.mean(row[f"precision_at_{k}"] for row in rows),
        f"recall_at_{k}": statistics.mean(row[f"recall_at_{k}"] for row in rows),
        f"ndcg_at_{k}": statistics.mean(row[f"ndcg_at_{k}"] for row in rows),
        "mrr": statistics.mean(row["mrr"] for row in rows),
        "map": statistics.mean(row["average_precision"] for row in rows),
        "evaluated_papers": len(rows),
    }


def compute_mrr(sim_data, gt, k=5):
    """Compatibility wrapper used by the Flask dashboard."""
    return aggregate_ranking_metrics(rankings_from_similarity(sim_data), normalize_ground_truth(gt), k)["mrr"]


def compute_precision_at_k(sim_data, gt, k=5):
    """Compatibility wrapper used by the Flask dashboard."""
    return aggregate_ranking_metrics(rankings_from_similarity(sim_data), normalize_ground_truth(gt), k)[f"precision_at_{k}"]


def compute_recall_at_k(sim_data, gt, k=5):
    """Compatibility wrapper used by the Flask dashboard."""
    return aggregate_ranking_metrics(rankings_from_similarity(sim_data), normalize_ground_truth(gt), k)[f"recall_at_{k}"]


def compute_coverage(sim_data, total_reviewers, k=5):
    """Return reviewer coverage across the top-k recommendations for each paper."""
    return coverage_at_k(rankings_from_similarity(sim_data), total_reviewers, k)


def compute_diversity(sim_data, k=3):
    """Estimate panel diversity from unique reviewers represented in each top-k panel."""
    rankings = rankings_from_similarity(sim_data)
    values = []
    for ranked in rankings.values():
        reviewer_ids = [row["reviewer_id"] for row in normalize_ranked(ranked)[:k]]
        if len(reviewer_ids) <= 1:
            continue
        unique_count = len(set(reviewer_ids))
        values.append((unique_count - 1) / (len(reviewer_ids) - 1))
    return statistics.mean(values) if values else 0.0


def compute_load_balance(assignments):
    """Return Jain's fairness index over reviewer assignment counts."""
    counts = Counter()
    for assigned in (assignments or {}).values():
        if isinstance(assigned, dict):
            for value in assigned.values():
                if isinstance(value, list):
                    counts.update(str(item) for item in value if item)
                elif value:
                    counts[str(value)] += 1
        else:
            for item in assigned or []:
                if isinstance(item, dict) and item.get("reviewer_id"):
                    counts[str(item["reviewer_id"])] += 1
                elif item:
                    counts[str(item)] += 1
    return jain_index(counts.values()) if counts else 0.0


def token_set(text):
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or "")).split()
        if len(token) > 3
    }


def reviewer_profile_tokens(reviewer):
    parts = []
    for key in ("area", "areas", "expertise", "topics", "keywords"):
        value = reviewer.get(key)
        if isinstance(value, list):
            parts.extend(map(str, value))
        elif value:
            parts.append(str(value))
    for publication in reviewer.get("publications", [])[:10]:
        parts.append(str(publication.get("title", "")))
        parts.append(str(publication.get("abstract", ""))[:500])
    return token_set(" ".join(parts))


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def panel_diversity(rankings, reviewer_lookup, k=3):
    values = []
    entropies = []
    for ranked in rankings.values():
        reviewer_ids = [row["reviewer_id"] for row in normalize_ranked(ranked)[:k]]
        token_sets = [reviewer_lookup.get(rid, set()) for rid in reviewer_ids]
        pairwise = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                pairwise.append(1.0 - jaccard(token_sets[i], token_sets[j]))
        if pairwise:
            values.append(statistics.mean(pairwise))
        topic_counts = Counter(token for tokens in token_sets for token in tokens)
        total = sum(topic_counts.values())
        if total:
            entropy = -sum((count / total) * math.log2(count / total) for count in topic_counts.values())
            entropies.append(entropy / math.log2(max(2, len(topic_counts))))
    return {
        f"panel_dissimilarity_at_{k}": statistics.mean(values) if values else 0.0,
        f"topic_entropy_at_{k}": statistics.mean(entropies) if entropies else 0.0,
    }


def coverage_at_k(rankings, total_reviewers, k=5):
    used = set()
    for ranked in rankings.values():
        used.update(row["reviewer_id"] for row in normalize_ranked(ranked)[:k])
    return len(used) / total_reviewers if total_reviewers else 0.0


def gini(values):
    values = sorted(float(value) for value in values)
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    weighted = sum((2 * idx - n - 1) * value for idx, value in enumerate(values, start=1))
    return weighted / (n * sum(values))


def jain_index(values):
    values = [float(value) for value in values]
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value * value for value in values)
    return numerator / denominator if denominator else 0.0


def load_fairness(assignments, reviewer_ids, saved_fairness=None):
    if saved_fairness and saved_fairness.get("assignment_counts"):
        counts = Counter({str(k): int(v or 0) for k, v in saved_fairness["assignment_counts"].items()})
    else:
        counts = Counter()
        for assigned in (assignments or {}).values():
            if isinstance(assigned, dict):
                counts.update(str(rid) for rid in assigned.values() if rid)
            else:
                for item in assigned or []:
                    if isinstance(item, dict) and item.get("reviewer_id"):
                        counts[str(item["reviewer_id"])] += 1
                    elif item:
                        counts[str(item)] += 1
    loads = [counts.get(str(rid), 0) for rid in reviewer_ids]
    return {
        "assigned_reviewers": sum(1 for value in loads if value > 0),
        "total_reviewers": len(loads),
        "coverage": (sum(1 for value in loads if value > 0) / len(loads)) if loads else 0.0,
        "avg_load": statistics.mean(loads) if loads else 0.0,
        "std_load": statistics.pstdev(loads) if loads else 0.0,
        "max_load": max(loads) if loads else 0,
        "gini": gini(loads),
        "jain_fairness": jain_index(loads),
    }


def paired_significance(method_rows, proposed_name="proposed", baseline_names=("tfidf", "embedding")):
    if proposed_name not in method_rows:
        return []
    proposed = {row["paper_id"]: row["average_precision"] for row in method_rows[proposed_name]}
    results = []
    for baseline_name in baseline_names:
        if baseline_name not in method_rows:
            continue
        baseline = {row["paper_id"]: row["average_precision"] for row in method_rows[baseline_name]}
        common_ids = sorted(set(proposed) & set(baseline))
        proposed_values = [proposed[pid] for pid in common_ids]
        baseline_values = [baseline[pid] for pid in common_ids]
        item = {
            "comparison": f"{proposed_name} vs {baseline_name}",
            "metric": "per-paper average precision",
            "paired_samples": len(common_ids),
            "mean_delta": statistics.mean(p - b for p, b in zip(proposed_values, baseline_values)) if common_ids else 0.0,
        }
        if len(common_ids) >= 2 and ttest_rel:
            test = ttest_rel(proposed_values, baseline_values)
            item["paired_t_p_value"] = float(test.pvalue) if not math.isnan(float(test.pvalue)) else 1.0
            item["paired_t_statistic"] = float(test.statistic) if not math.isnan(float(test.statistic)) else 0.0
        else:
            item["paired_t_p_value"] = None
            item["paired_t_statistic"] = None
        if len(common_ids) >= 2 and wilcoxon:
            try:
                test = wilcoxon(proposed_values, baseline_values, zero_method="zsplit")
                item["wilcoxon_p_value"] = float(test.pvalue)
                item["wilcoxon_statistic"] = float(test.statistic)
            except ValueError:
                item["wilcoxon_p_value"] = None
                item["wilcoxon_statistic"] = None
        else:
            item["wilcoxon_p_value"] = None
            item["wilcoxon_statistic"] = None
        results.append(item)
    return results


def safe_delta(current, baseline):
    return current - baseline if current is not None and baseline is not None else None


def percent_delta(current, baseline):
    if current is None or baseline in (None, 0):
        return None
    return ((current - baseline) / abs(baseline)) * 100


def lookup_p_value(significance, comparison_name):
    for item in significance or []:
        if item.get("comparison") == comparison_name:
            return item.get("p_value", item.get("paired_t_p_value"))
    return None


def build_interpretation(report):
    ranking = report.get("ranking", {})
    proposed = ranking.get("proposed", {})
    embedding = ranking.get("embedding") or ranking.get("semantic_only") or {}
    tfidf = ranking.get("tfidf", {})
    fairness = report.get("fairness", {})
    diversity = report.get("diversity", {}).get("proposed", {})

    p_embedding = lookup_p_value(report.get("significance"), "Proposed vs Embedding")
    p_tfidf = lookup_p_value(report.get("significance"), "Proposed vs TF-IDF")
    ndcg_delta_embedding = safe_delta(proposed.get("ndcg_at_5"), embedding.get("ndcg_at_5"))
    mrr_delta_embedding = safe_delta(proposed.get("mrr"), embedding.get("mrr"))
    map_delta_embedding = safe_delta(proposed.get("map"), embedding.get("map"))

    if p_embedding is not None and p_embedding >= 0.05:
        ranking_status = "competitive_not_significantly_better_than_embedding"
    elif ndcg_delta_embedding is not None and ndcg_delta_embedding > 0:
        ranking_status = "improved_over_embedding"
    else:
        ranking_status = "does_not_outperform_embedding"

    if p_tfidf is not None and p_tfidf < 0.05:
        tfidf_status = "significantly_better_than_tfidf"
    elif proposed.get("ndcg_at_5", 0) > tfidf.get("ndcg_at_5", 0):
        tfidf_status = "better_than_tfidf_but_not_statistically_confirmed"
    else:
        tfidf_status = "not_better_than_tfidf"

    return {
        "ranking_status": ranking_status,
        "tfidf_status": tfidf_status,
        "recommended_positioning": (
            "Frame the system as hybrid multi-signal reviewer panel optimization that preserves "
            "competitive retrieval quality while improving operational assignment properties such "
            "as coverage, workload fairness, explainability, and topical panel diversity."
        ),
        "avoid_claim": "Do not claim a large or statistically significant retrieval gain over embedding-only retrieval.",
        "key_comparison_to_embedding": {
            "mrr_delta": mrr_delta_embedding,
            "map_delta": map_delta_embedding,
            "ndcg_at_5_delta": ndcg_delta_embedding,
            "ndcg_at_5_relative_percent": percent_delta(proposed.get("ndcg_at_5"), embedding.get("ndcg_at_5")),
            "p_value": p_embedding,
        },
        "strengths_to_emphasize": {
            "reviewer_coverage": fairness.get("coverage"),
            "jain_fairness": fairness.get("jain_fairness"),
            "gini": fairness.get("gini"),
            "panel_dissimilarity_at_3": diversity.get("panel_dissimilarity_at_3"),
            "topic_entropy_at_3": diversity.get("topic_entropy_at_3"),
        },
        "next_priority": [
            "Increase measurable KG and authority contribution in ablation results.",
            "Tune panel optimization so the proposed method does not lose Precision@3 or Recall@10 to embedding-only retrieval.",
            "Report fairness/diversity gains as primary assignment-quality outcomes when ranking gains are small.",
        ],
    }


def load_rankings():
    rankings = load_json(RANKINGS_PATH)
    if rankings:
        return {method: rankings_from_similarity(data) for method, data in rankings.items()}
    sim_data = load_json(SIM_PATH, {})
    return {"similarity_combined": rankings_from_similarity(sim_data)}


def print_metric_block(name, metrics):
    print(f"\n{name}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    print("\n=== SYSTEM METRICS REPORT ===")

    ground_truth = normalize_ground_truth(load_json(GT_PATH, {}))
    reviewers = load_json(REVIEWERS_PATH, []) or []
    reviewer_ids = [str(row.get("reviewer_id")) for row in reviewers if row.get("reviewer_id") is not None]
    reviewer_lookup = {str(row.get("reviewer_id")): reviewer_profile_tokens(row) for row in reviewers}
    rankings_by_method = load_rankings()
    assignments = load_json(ASSIGN_PATH, {}) or {}
    saved_fairness = load_json(FAIRNESS_PATH, {}) or {}

    if not ground_truth or not rankings_by_method:
        print("Missing ground truth or ranking data. Run tools\\run_full_evaluation.py first.")
        return

    report = {
        "ranking": {},
        "diversity": {},
        "coverage": {},
        "fairness": {},
        "significance": [],
        "notes": [
            "Reviewer-assignment labels are sparse; absolute precision can be low even when recommendations are plausible.",
            "Use relative improvements, NDCG@K, MAP, diversity, and fairness together rather than relying on one metric.",
        ],
    }
    per_method_rows = {}

    for method, rankings in rankings_by_method.items():
        method_report = {}
        for k in KS:
            method_report.update(aggregate_ranking_metrics(rankings, ground_truth, k))
        report["ranking"][method] = method_report
        report["coverage"][method] = {
            f"coverage_at_{k}": coverage_at_k(rankings, len(reviewer_ids), k) for k in KS
        }
        report["diversity"][method] = {}
        for k in (3, 5):
            report["diversity"][method].update(panel_diversity(rankings, reviewer_lookup, k))
        per_method_rows[method] = per_paper_scores(rankings, ground_truth, 5)

    report["fairness"] = load_fairness(assignments, reviewer_ids, saved_fairness)
    report["significance"] = load_json(SIGNIFICANCE_PATH, None) or paired_significance(per_method_rows)
    report["interpretation"] = build_interpretation(report)

    save_json(REPORT_JSON, report)

    csv_rows = []
    for method, metrics in report["ranking"].items():
        row = {"method": method}
        row.update({key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()})
        row.update({key: round(value, 6) for key, value in report["coverage"].get(method, {}).items()})
        row.update({key: round(value, 6) for key, value in report["diversity"].get(method, {}).items()})
        if method == "proposed" and "embedding" in report["ranking"]:
            embedding_metrics = report["ranking"]["embedding"]
            row["delta_vs_embedding_mrr"] = round(metrics.get("mrr", 0.0) - embedding_metrics.get("mrr", 0.0), 6)
            row["delta_vs_embedding_map"] = round(metrics.get("map", 0.0) - embedding_metrics.get("map", 0.0), 6)
            row["delta_vs_embedding_ndcg_at_5"] = round(
                metrics.get("ndcg_at_5", 0.0) - embedding_metrics.get("ndcg_at_5", 0.0),
                6,
            )
        csv_rows.append(row)
    write_csv(REPORT_CSV, csv_rows)

    for method in sorted(report["ranking"]):
        print_metric_block(f"Ranking: {method}", report["ranking"][method])
        print_metric_block(f"Coverage: {method}", report["coverage"][method])
        print_metric_block(f"Diversity: {method}", report["diversity"][method])

    print_metric_block("Fairness", report["fairness"])
    print("\nSignificance")
    for item in report["significance"]:
        print(
            f"  {item.get('comparison')}: "
            f"p={item.get('p_value', item.get('paired_t_p_value'))}, "
            f"samples={item.get('paired_samples')}"
        )
    print_metric_block("Interpretation", report["interpretation"])

    print(f"\nSaved: {REPORT_JSON}")
    print(f"Saved: {REPORT_CSV}")


if __name__ == "__main__":
    main()
