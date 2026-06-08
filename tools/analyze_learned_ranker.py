import csv
import json
import math
import os

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:
    ttest_rel = None
    wilcoxon = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.json")
METHOD_RANKINGS_PATH = os.path.join(RESULTS_DIR, "rankings_by_method.json")
LEARNED_RANKINGS_PATH = os.path.join(RESULTS_DIR, "learned_ranker_rankings.json")
LEARNED_FEATURES_PATH = os.path.join(RESULTS_DIR, "learned_ranker_features.json")
MIN_COMMON_PAPERS = int(os.getenv("MIN_LEARNED_SIGNIFICANCE_PAPERS", "100"))

OUT_SIG_JSON = os.path.join(RESULTS_DIR, "learned_ranker_significance.json")
OUT_SIG_CSV = os.path.join(RESULTS_DIR, "learned_ranker_significance.csv")
OUT_IMPORTANCE_JSON = os.path.join(RESULTS_DIR, "learned_ranker_feature_importance.json")
OUT_IMPORTANCE_CSV = os.path.join(RESULTS_DIR, "learned_ranker_feature_importance.csv")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_ground_truth(raw):
    if isinstance(raw, list):
        return {
            str(row.get("paper_id")): {str(rid) for rid in row.get("assigned_reviewers", [])}
            for row in raw
            if row.get("paper_id") is not None and row.get("assigned_reviewers")
        }
    return {str(pid): {str(rid) for rid in reviewers} for pid, reviewers in raw.items()}


def average_precision(ranking, truth):
    if not truth:
        return 0.0
    hits = 0
    total = 0.0
    for idx, reviewer_id in enumerate(ranking, start=1):
        if str(reviewer_id) in truth:
            hits += 1
            total += hits / idx
    return total / len(truth)


def reciprocal_rank(ranking, truth):
    for idx, reviewer_id in enumerate(ranking, start=1):
        if str(reviewer_id) in truth:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranking, truth, k):
    gains = [1.0 if str(reviewer_id) in truth else 0.0 for reviewer_id in ranking[:k]]
    dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def per_paper_scores(rankings, ground_truth):
    rows = {}
    for paper_id, truth in ground_truth.items():
        ranked = rankings.get(str(paper_id), [])
        if not ranked:
            continue
        ranking = [str(item["reviewer_id"]) for item in ranked]
        rows[str(paper_id)] = {
            "average_precision": average_precision(ranking, truth),
            "mrr": reciprocal_rank(ranking, truth),
            "ndcg_at_5": ndcg_at_k(ranking, truth, 5),
        }
    return rows


def paired_test(left_values, right_values):
    if len(left_values) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "wilcoxon_p_value": None}
    if ttest_rel:
        test = ttest_rel(left_values, right_values)
        t_stat = float(test.statistic) if not math.isnan(float(test.statistic)) else 0.0
        p_value = float(test.pvalue) if not math.isnan(float(test.pvalue)) else 1.0
    else:
        t_stat = 0.0
        p_value = None
    wilcoxon_p = None
    if wilcoxon:
        try:
            wilcoxon_p = float(wilcoxon(left_values, right_values, zero_method="zsplit").pvalue)
        except ValueError:
            wilcoxon_p = None
    return {"t_statistic": t_stat, "p_value": p_value, "wilcoxon_p_value": wilcoxon_p}


def compare_methods(learned_scores, baseline_scores, baseline_name, metric):
    common_ids = sorted(set(learned_scores) & set(baseline_scores))
    left = [learned_scores[pid][metric] for pid in common_ids]
    right = [baseline_scores[pid][metric] for pid in common_ids]
    test = paired_test(left, right)
    mean_delta = (sum(l - r for l, r in zip(left, right)) / len(left)) if left else 0.0
    return {
        "comparison": f"Learned ranker vs {baseline_name}",
        "metric": metric,
        "paired_samples": len(common_ids),
        "mean_delta": mean_delta,
        **test,
        "significant_p_lt_0_05": (test["p_value"] is not None and test["p_value"] < 0.05),
        "is_full_evaluation": len(common_ids) >= MIN_COMMON_PAPERS,
        "warning": "" if len(common_ids) >= MIN_COMMON_PAPERS else "limited common-paper count; rerun full evaluation before paper use",
    }


def feature_group(feature_name):
    if feature_name.startswith("kg"):
        return "KG evidence"
    if feature_name.startswith("semantic") or feature_name == "similarity_score":
        return "Semantic relevance"
    if feature_name.startswith("authority"):
        return "Authority evidence"
    if "keyword" in feature_name:
        return "Topic overlap"
    if "hybrid" in feature_name:
        return "Frozen hybrid rank"
    return "Other"


def build_feature_importance():
    features = load_json(LEARNED_FEATURES_PATH)
    coefficients = features.get("logistic_probe_coefficients", {})
    total_abs = sum(abs(float(value)) for value in coefficients.values()) or 1.0
    rows = []
    group_totals = {}
    for feature, value in sorted(coefficients.items(), key=lambda item: abs(float(item[1])), reverse=True):
        coefficient = float(value)
        group = feature_group(feature)
        normalized_abs = abs(coefficient) / total_abs
        rows.append({
            "feature": feature,
            "group": group,
            "coefficient": coefficient,
            "direction": "positive" if coefficient > 0 else "negative",
            "normalized_abs_importance": normalized_abs,
        })
        group_totals[group] = group_totals.get(group, 0.0) + normalized_abs
    return {
        "feature_importance": rows,
        "group_importance": [
            {"group": group, "normalized_abs_importance": value}
            for group, value in sorted(group_totals.items(), key=lambda item: item[1], reverse=True)
        ],
        "notes": [
            "Feature importances come from a balanced logistic probe trained on the learned-ranker feature matrix.",
            "Positive coefficients increase assignment likelihood; negative coefficients indicate a feature was not useful in its current form.",
        ],
    }


def main():
    ground_truth = normalize_ground_truth(load_json(GROUND_TRUTH_PATH))
    method_rankings = load_json(METHOD_RANKINGS_PATH)
    learned_rankings = load_json(LEARNED_RANKINGS_PATH)

    learned_scores = per_paper_scores(learned_rankings, ground_truth)
    baselines = {
        "TF-IDF": method_rankings.get("tfidf", {}),
        "Embedding": method_rankings.get("embedding", {}),
        "Proposed stable v1": method_rankings.get("proposed", {}),
        "Semantic + KG": method_rankings.get("semantic_kg", {}),
    }
    significance_rows = []
    for baseline_name, rankings in baselines.items():
        baseline_scores = per_paper_scores(rankings, ground_truth)
        for metric in ("average_precision", "mrr", "ndcg_at_5"):
            significance_rows.append(compare_methods(learned_scores, baseline_scores, baseline_name, metric))

    importance = build_feature_importance()
    save_json(OUT_SIG_JSON, significance_rows)
    write_csv(OUT_SIG_CSV, significance_rows)
    save_json(OUT_IMPORTANCE_JSON, importance)
    write_csv(OUT_IMPORTANCE_CSV, importance["feature_importance"])

    print("Learned-ranker significance:")
    for row in significance_rows:
        if row["metric"] == "average_precision":
            print(f"  {row['comparison']}: p={row['p_value']:.6g}, delta={row['mean_delta']:.6f}")
    print(f"Saved: {OUT_SIG_JSON}")
    print(f"Saved: {OUT_SIG_CSV}")
    print(f"Saved: {OUT_IMPORTANCE_JSON}")
    print(f"Saved: {OUT_IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()
