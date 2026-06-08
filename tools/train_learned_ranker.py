import csv
import importlib
import json
import math
import os
import inspect
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

LIVE_RANKINGS_PATH = os.path.join(RESULTS_DIR, "live_rankings_with_features.json")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.json")
OUT_RANKINGS = os.path.join(RESULTS_DIR, "learned_ranker_rankings.json")
OUT_METRICS = os.path.join(RESULTS_DIR, "learned_ranker_metrics.json")
OUT_METRICS_CSV = os.path.join(RESULTS_DIR, "learned_ranker_metrics.csv")
OUT_FEATURES = os.path.join(RESULTS_DIR, "learned_ranker_features.json")
USE_HARD_NEGATIVES = os.getenv("USE_HARD_NEGATIVES", "false").lower() in ("1", "true", "yes")
HARD_NEGATIVES_PER_PAPER = int(os.getenv("HARD_NEGATIVES_PER_PAPER", "30"))
HARD_NEGATIVE_WEIGHT = float(os.getenv("HARD_NEGATIVE_WEIGHT", "2.0"))
LEARNED_RANKER_MODEL = os.getenv("LEARNED_RANKER_MODEL", "hist_gbdt").lower()
MIN_PAPERS_FOR_OUTPUT = int(os.getenv("MIN_LEARNED_RANKER_PAPERS", "100"))
ALLOW_SMALL_LEARNED_RANKER = os.getenv("ALLOW_SMALL_LEARNED_RANKER", "false").lower() in ("1", "true", "yes")

ALL_FEATURE_NAMES = [
    "similarity_score",
    "kg_effective",
    "kg_score",
    "kg_confidence",
    "authority_score",
    "keyword_overlap_capped",
    "semantic_rank_recip",
    "kg_rank_recip",
    "authority_rank_recip",
    "hybrid_rank_recip",
]

DEFAULT_EXCLUDED_FEATURES = {
    "authority_score",
    "authority_rank_recip",
    "hybrid_rank_recip",
}
ENV_EXCLUDED_FEATURES = {
    item.strip()
    for item in os.getenv("LEARNED_RANKER_EXCLUDE_FEATURES", ",".join(sorted(DEFAULT_EXCLUDED_FEATURES))).split(",")
    if item.strip()
}
FEATURE_NAMES = [name for name in ALL_FEATURE_NAMES if name not in ENV_EXCLUDED_FEATURES]
_XGB_RANKER_CLASS = None


def get_xgb_ranker_class():
    global _XGB_RANKER_CLASS
    if _XGB_RANKER_CLASS is not None:
        return _XGB_RANKER_CLASS
    try:
        module = importlib.import_module("xgboost")
        _XGB_RANKER_CLASS = getattr(module, "XGBRanker")
    except Exception:
        _XGB_RANKER_CLASS = False
    return _XGB_RANKER_CLASS or None


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def rank_map(rows, key):
    ordered = sorted(rows, key=lambda row: float(row.get(key, 0.0) or 0.0), reverse=True)
    return {str(row.get("reviewer_id")): idx for idx, row in enumerate(ordered, start=1)}


def build_dataset(live_rankings, ground_truth):
    x_rows = []
    y_rows = []
    groups = []
    meta = []

    for paper_id, rows in live_rankings.items():
        truth = ground_truth.get(str(paper_id), set())
        if not rows:
            continue

        semantic_ranks = rank_map(rows, "similarity_score")
        kg_ranks = rank_map(rows, "kg_effective")
        authority_ranks = rank_map(rows, "authority_score")
        hybrid_ranks = rank_map(rows, "score")

        for row in rows:
            reviewer_id = str(row.get("reviewer_id"))
            semantic_rank = semantic_ranks.get(reviewer_id, len(rows))
            kg_rank = kg_ranks.get(reviewer_id, len(rows))
            authority_rank = authority_ranks.get(reviewer_id, len(rows))
            hybrid_rank = hybrid_ranks.get(reviewer_id, len(rows))
            features = {
                "similarity_score": float(row.get("similarity_score", 0.0) or 0.0),
                "kg_effective": float(row.get("kg_effective", row.get("kg_score", 0.0)) or 0.0),
                "kg_score": float(row.get("kg_score", 0.0) or 0.0),
                "kg_confidence": float(row.get("kg_confidence", 0.0) or 0.0),
                "authority_score": float(row.get("authority_score", 0.0) or 0.0),
                "keyword_overlap_capped": float(row.get("keyword_overlap_capped", 0.0) or 0.0),
                "semantic_rank_recip": 1.0 / max(1, semantic_rank),
                "kg_rank_recip": 1.0 / max(1, kg_rank),
                "authority_rank_recip": 1.0 / max(1, authority_rank),
                "hybrid_rank_recip": 1.0 / max(1, hybrid_rank),
            }
            x_rows.append([features[name] for name in FEATURE_NAMES])
            y_rows.append(1 if reviewer_id in truth else 0)
            groups.append(str(paper_id))
            meta.append({
                "paper_id": str(paper_id),
                "reviewer_id": reviewer_id,
                "semantic_rank": semantic_rank,
                "kg_rank": kg_rank,
                "authority_rank": authority_rank,
                "hybrid_rank": hybrid_rank,
            })

    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int), np.asarray(groups), meta


def make_model():
    xgb_ranker_class = get_xgb_ranker_class()
    if LEARNED_RANKER_MODEL == "xgboost" and xgb_ranker_class is not None:
        return xgb_ranker_class(
            objective="rank:ndcg",
            n_estimators=int(os.getenv("XGB_RANKER_ESTIMATORS", "120")),
            learning_rate=float(os.getenv("XGB_RANKER_LR", "0.05")),
            max_depth=int(os.getenv("XGB_RANKER_DEPTH", "4")),
            subsample=float(os.getenv("XGB_RANKER_SUBSAMPLE", "0.9")),
            colsample_bytree=float(os.getenv("XGB_RANKER_COLSAMPLE", "0.9")),
            random_state=42,
        )
    return HistGradientBoostingClassifier(
        max_iter=int(os.getenv("LEARNED_RANKER_MAX_ITER", "120")),
        learning_rate=float(os.getenv("LEARNED_RANKER_LR", "0.05")),
        max_leaf_nodes=int(os.getenv("LEARNED_RANKER_LEAVES", "15")),
        l2_regularization=float(os.getenv("LEARNED_RANKER_L2", "0.01")),
        random_state=42,
    )


def build_sample_weight(y_train, train_meta):
    positive = max(1, int(y_train.sum()))
    negative = max(1, len(y_train) - positive)
    pos_weight = min(50.0, negative / positive)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0).astype(float)
    if USE_HARD_NEGATIVES and HARD_NEGATIVES_PER_PAPER > 0:
        for idx, row in enumerate(train_meta):
            if y_train[idx] == 0 and int(row.get("semantic_rank", 10**9)) <= HARD_NEGATIVES_PER_PAPER:
                sample_weight[idx] *= HARD_NEGATIVE_WEIGHT
    return sample_weight


def xgb_group_sizes(train_meta):
    sizes = []
    current = None
    count = 0
    for row in train_meta:
        paper_id = row["paper_id"]
        if current is None:
            current = paper_id
        if paper_id != current:
            sizes.append(count)
            current = paper_id
            count = 0
        count += 1
    if count:
        sizes.append(count)
    return sizes


def fit_predict_fold(model, x_train, y_train, x_test, train_meta):
    if len(set(y_train.tolist())) < 2:
        return np.zeros(len(x_test), dtype=float)
    sample_weight = build_sample_weight(y_train, train_meta)
    xgb_ranker_class = get_xgb_ranker_class()
    if inspect.isclass(xgb_ranker_class) and isinstance(model, xgb_ranker_class):
        order = sorted(range(len(train_meta)), key=lambda idx: train_meta[idx]["paper_id"])
        x_ordered = x_train[order]
        y_ordered = y_train[order]
        weight_ordered = sample_weight[order]
        meta_ordered = [train_meta[idx] for idx in order]
        model.fit(x_ordered, y_ordered, group=xgb_group_sizes(meta_ordered), sample_weight=weight_ordered)
        return model.predict(x_test)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    return model.predict(x_test)


def cross_validated_predictions(x, y, groups, meta):
    predictions = np.zeros(len(y), dtype=float)
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise SystemExit("Need at least two papers for learned ranker cross-validation.")

    splitter = GroupKFold(n_splits=n_splits)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_model()
        train_meta = [meta[idx] for idx in train_idx]
        predictions[test_idx] = fit_predict_fold(model, x[train_idx], y[train_idx], x[test_idx], train_meta)
        print(f"Finished learned-ranker fold {fold_idx}/{n_splits}")
    return predictions


def build_rankings(meta, predictions):
    by_paper = defaultdict(list)
    for row, score in zip(meta, predictions):
        by_paper[row["paper_id"]].append({
            "reviewer_id": row["reviewer_id"],
            "score": float(score),
        })
    return {
        paper_id: sorted(rows, key=lambda item: item["score"], reverse=True)
        for paper_id, rows in by_paper.items()
    }


def reciprocal_rank(ranking, truth):
    for idx, reviewer_id in enumerate(ranking, start=1):
        if str(reviewer_id) in truth:
            return 1.0 / idx
    return 0.0


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


def ndcg_at_k(ranking, truth, k):
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
        hits = len(set(top_k) & truth)
        rows.append({
            f"precision_at_{k}": hits / k,
            f"recall_at_{k}": hits / len(truth) if truth else 0.0,
            f"ndcg_at_{k}": ndcg_at_k(ranking, truth, k),
            "mrr": reciprocal_rank(ranking, truth),
            "map": average_precision(ranking, truth),
        })
    if not rows:
        return {}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    } | {"evaluated_papers": len(rows)}


def train_interpretable_probe(x, y):
    if len(set(y.tolist())) < 2:
        return {}
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x_scaled, y)
    coefficients = model.coef_[0]
    return {
        name: float(coef)
        for name, coef in sorted(zip(FEATURE_NAMES, coefficients), key=lambda item: abs(item[1]), reverse=True)
    }


def main():
    if not os.path.exists(LIVE_RANKINGS_PATH):
        raise SystemExit("Missing results\\live_rankings_with_features.json. Run tools\\run_full_evaluation.py first.")

    live_rankings = load_json(LIVE_RANKINGS_PATH)
    ground_truth = normalize_ground_truth(load_json(GROUND_TRUTH_PATH))
    if len(live_rankings) < MIN_PAPERS_FOR_OUTPUT and not ALLOW_SMALL_LEARNED_RANKER:
        raise SystemExit(
            f"Only {len(live_rankings)} papers found in results\\live_rankings_with_features.json. "
            "This looks like a limited EVAL_LIMIT run. Rerun tools\\run_full_evaluation.py without EVAL_LIMIT, "
            "or set ALLOW_SMALL_LEARNED_RANKER=true for a debug-only run."
        )
    x, y, groups, meta = build_dataset(live_rankings, ground_truth)
    print(f"Training learned reranker on {len(y)} candidate pairs, positives={int(y.sum())}")

    predictions = cross_validated_predictions(x, y, groups, meta)
    rankings = build_rankings(meta, predictions)
    metrics = {
        "learned_ranker": {
            **aggregate_metrics(rankings, ground_truth, 3),
            **aggregate_metrics(rankings, ground_truth, 5),
            **aggregate_metrics(rankings, ground_truth, 10),
        }
    }
    features = {
        "feature_names": FEATURE_NAMES,
        "excluded_features": sorted(ENV_EXCLUDED_FEATURES),
        "model_type": LEARNED_RANKER_MODEL if LEARNED_RANKER_MODEL != "xgboost" or get_xgb_ranker_class() is not None else "hist_gbdt_fallback_xgboost_unavailable",
        "candidate_pairs": int(len(y)),
        "positive_pairs": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "training_strategy": "all_candidates_with_hard_negative_weighting" if USE_HARD_NEGATIVES else "all_candidate_pairs",
        "hard_negatives_per_paper": HARD_NEGATIVES_PER_PAPER if USE_HARD_NEGATIVES else None,
        "hard_negative_weight": HARD_NEGATIVE_WEIGHT if USE_HARD_NEGATIVES else None,
        "logistic_probe_coefficients": train_interpretable_probe(x, y),
    }

    save_json(OUT_RANKINGS, rankings)
    save_json(OUT_METRICS, metrics)
    save_json(OUT_FEATURES, features)
    write_csv(OUT_METRICS_CSV, [{"method": "learned_ranker", **metrics["learned_ranker"]}])

    print(json.dumps(metrics, indent=2))
    print(f"Saved: {OUT_RANKINGS}")
    print(f"Saved: {OUT_METRICS}")
    print(f"Saved: {OUT_FEATURES}")


if __name__ == "__main__":
    main()
