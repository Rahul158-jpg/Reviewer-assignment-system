import csv
import json
import math
import os
import statistics


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
GROUND_TRUTH_PATH = os.path.join(ROOT_DIR, "data", "ground_truth.json")
LIVE_RANKINGS_PATH = os.path.join(RESULTS_DIR, "live_rankings_with_features.json")
PAIR_OUTPUT_PATH = os.path.join(RESULTS_DIR, "llm_signal_top10_pairs.csv")
FALSE_NEGATIVE_OUTPUT_PATH = os.path.join(RESULTS_DIR, "llm_signal_false_negatives.csv")
HIGH_FALSE_POSITIVE_OUTPUT_PATH = os.path.join(RESULTS_DIR, "llm_signal_high_false_positives.csv")
SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "llm_signal_analysis.json")
SUMMARY_MD_PATH = os.path.join(RESULTS_DIR, "llm_signal_analysis.md")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ground_truth(rows):
    lookup = {}
    if isinstance(rows, dict):
        for paper_id, reviewers in rows.items():
            lookup[str(paper_id)] = {str(rid) for rid in reviewers or []}
        return lookup

    for row in rows:
        paper_id = str(row.get("paper_id", ""))
        reviewers = (
            row.get("assigned_reviewers")
            or row.get("ground_truth_reviewer_ids")
            or row.get("reviewer_ids")
            or []
        )
        if paper_id:
            lookup[paper_id] = {str(rid) for rid in reviewers}
    return lookup


def mean(values):
    return sum(values) / len(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def zero_rate(values):
    return sum(1 for value in values if abs(value) <= 1e-12) / len(values) if values else 0.0


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def safe_float(value):
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def roc_auc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None

    wins = 0.0
    total = len(positives) * len(negatives)
    for pos_score in positives:
        for neg_score in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def point_biserial(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives or len(scores) < 2:
        return None

    std_all = statistics.pstdev(scores)
    if std_all == 0:
        return None

    p = len(positives) / len(scores)
    q = len(negatives) / len(scores)
    return (mean(positives) - mean(negatives)) / std_all * math.sqrt(p * q)


def summarize_group(values):
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "zero_rate": zero_rate(values),
    }


def main():
    ground_truth = normalize_ground_truth(load_json(GROUND_TRUTH_PATH))
    live_rankings = load_json(LIVE_RANKINGS_PATH)

    pair_rows = []
    for paper_id in sorted(live_rankings.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))[:100]:
        relevant_reviewers = ground_truth.get(str(paper_id), set())
        for rank, item in enumerate(live_rankings.get(paper_id, [])[:10], start=1):
            reviewer_id = str(item.get("reviewer_id", ""))
            llm_score = safe_float(item.get("llm_suitability_score", item.get("llm_match_score", 0.0)))
            pair_rows.append({
                "paper_id": str(paper_id),
                "rank": rank,
                "reviewer_id": reviewer_id,
                "relevant": int(reviewer_id in relevant_reviewers),
                "semantic_score": safe_float(item.get("similarity_score", 0.0)),
                "kg_score": safe_float(item.get("kg_score", 0.0)),
                "kg_effective": safe_float(item.get("kg_effective", item.get("kg_score", 0.0))),
                "llm_score": llm_score,
                "authority_score": safe_float(item.get("authority_score", 0.0)),
                "final_score": safe_float(item.get("score", 0.0)),
            })

    labels = [bool(row["relevant"]) for row in pair_rows]
    llm_scores = [row["llm_score"] for row in pair_rows]
    semantic_scores = [row["semantic_score"] for row in pair_rows]
    kg_scores = [row["kg_effective"] for row in pair_rows]
    relevant_llm = [row["llm_score"] for row in pair_rows if row["relevant"]]
    nonrelevant_llm = [row["llm_score"] for row in pair_rows if not row["relevant"]]

    summary = {
        "sample": {
            "papers": len({row["paper_id"] for row in pair_rows}),
            "pairs": len(pair_rows),
            "relevant_pairs": sum(row["relevant"] for row in pair_rows),
            "top_k_per_paper": 10,
        },
        "llm_score": {
            "all": summarize_group(llm_scores),
            "relevant": summarize_group(relevant_llm),
            "nonrelevant": summarize_group(nonrelevant_llm),
            "mean_difference_relevant_minus_nonrelevant": mean(relevant_llm) - mean(nonrelevant_llm),
            "auc": roc_auc(labels, llm_scores),
            "point_biserial": point_biserial(labels, llm_scores),
        },
        "semantic_score": {
            "auc": roc_auc(labels, semantic_scores),
            "point_biserial": point_biserial(labels, semantic_scores),
        },
        "kg_effective": {
            "auc": roc_auc(labels, kg_scores),
            "point_biserial": point_biserial(labels, kg_scores),
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = list(pair_rows[0].keys()) if pair_rows else []
    with open(PAIR_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    false_negatives = [
        row for row in pair_rows
        if row["relevant"] and row["llm_score"] <= 1e-12
    ]
    high_false_positives = [
        row for row in pair_rows
        if not row["relevant"] and row["llm_score"] > 0.7
    ]
    with open(FALSE_NEGATIVE_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(false_negatives)
    with open(HIGH_FALSE_POSITIVE_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(high_false_positives)

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    verdict = "informative"
    llm_auc = summary["llm_score"]["auc"]
    llm_delta = summary["llm_score"]["mean_difference_relevant_minus_nonrelevant"]
    if llm_auc is None:
        verdict = "inconclusive: not enough positive/negative labels in the sampled top-10 pairs"
    elif llm_auc < 0.52 or llm_delta <= 0:
        verdict = "weak or noisy: improve LLM profile scoring before increasing its weight"
    elif llm_auc < 0.60:
        verdict = "mildly informative: tune cautiously and keep LLM weight small"

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# LLM Profile Signal Analysis\n\n")
        f.write("This diagnostic uses the first 100 papers from `results/live_rankings_with_features.json` ")
        f.write("and labels the top-10 reviewers using `data/ground_truth.json`.\n\n")
        f.write(f"## Verdict\n\n{verdict}\n\n")
        f.write("## Sample\n\n")
        f.write(f"- Papers: {summary['sample']['papers']}\n")
        f.write(f"- Pairs: {summary['sample']['pairs']}\n")
        f.write(f"- Relevant pairs in top 10: {summary['sample']['relevant_pairs']}\n\n")
        f.write("## LLM Score\n\n")
        for label in ("all", "relevant", "nonrelevant"):
            group = summary["llm_score"][label]
            f.write(
                f"- {label}: count={group['count']}, mean={group['mean']:.4f}, "
                f"median={group['median']:.4f}, p90={group['p90']:.4f}, "
                f"zero_rate={group['zero_rate']:.4f}\n"
            )
        f.write(
            f"- Mean difference relevant - nonrelevant: "
            f"{summary['llm_score']['mean_difference_relevant_minus_nonrelevant']:.4f}\n"
        )
        f.write(f"- AUC: {summary['llm_score']['auc']}\n")
        f.write(f"- Point-biserial correlation: {summary['llm_score']['point_biserial']}\n\n")
        f.write("## Comparator AUC\n\n")
        f.write(f"- Semantic score AUC: {summary['semantic_score']['auc']}\n")
        f.write(f"- KG effective AUC: {summary['kg_effective']['auc']}\n\n")
        f.write("## Pair-Level Output\n\n")
        f.write("See `results/llm_signal_top10_pairs.csv` for manual inspection.\n")
        f.write("See `results/llm_signal_false_negatives.csv` for relevant reviewers with zero LLM score.\n")
        f.write("See `results/llm_signal_high_false_positives.csv` for non-relevant reviewers with LLM score above 0.7.\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {PAIR_OUTPUT_PATH}")
    print(f"Wrote {FALSE_NEGATIVE_OUTPUT_PATH}")
    print(f"Wrote {HIGH_FALSE_POSITIVE_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_JSON_PATH}")
    print(f"Wrote {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
