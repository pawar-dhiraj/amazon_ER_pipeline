"""
Step 6 (used throughout) -- Scoring.

Implements the CHALLENGE'S OWN metric exactly: F_0.5 computed per
Source-1 entity, then macro-averaged across all Source-1 entities,
singletons included (empty predicted list on a true singleton scores
1.0; any predicted match on a true singleton scores 0.0).

Use this -- not sklearn's generic f1_score -- for all threshold
tuning and validation. Tuning against the wrong metric (plain
accuracy/F1) is one of the easiest ways to silently under-perform on
leaderboard day given how precision-heavy F0.5 is.
"""
from collections import defaultdict


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def score_entity(true_matches: set, predicted_matches: set) -> float:
    if not true_matches and not predicted_matches:
        return 1.0  # correctly predicted singleton
    if not predicted_matches:
        precision = 0.0
        recall = 0.0
    else:
        tp = len(true_matches & predicted_matches)
        precision = tp / len(predicted_matches)
        recall = tp / len(true_matches) if true_matches else 0.0
    return _f_beta(precision, recall, beta=0.5)


def macro_f0_5(ground_truth: dict, predictions: dict) -> float:
    """ground_truth / predictions: {source1_entity_id: set(matched_ids)}.
    Every S1 entity in ground_truth must have an entry in predictions
    (missing entries are treated as an empty prediction -- i.e. a
    singleton guess -- matching how an incomplete submission would
    actually be scored)."""
    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        scores.append(score_entity(true_matches, pred_matches))
    return sum(scores) / len(scores) if scores else 0.0


def parse_ground_truth_tsv(path: str) -> dict:
    """Parses train_ground_truth.tsv into {source1_entity_id: set(ids)}."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    result = defaultdict(set)
    for _, row in df.iterrows():
        ids = row["matched_entity_ids"].strip()
        result[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()
    return dict(result)


def tune_threshold(val_scored_pairs, ground_truth: dict, thresholds=None) -> tuple:
    """val_scored_pairs: DataFrame with columns
    [source1_entity_id, entity_id, match_probability].
    Sweeps thresholds and returns (best_threshold, best_f0_5)."""
    import numpy as np
    if thresholds is None:
        thresholds = np.arange(0.30, 0.96, 0.02)

    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        preds = defaultdict(set)
        subset = val_scored_pairs[val_scored_pairs["match_probability"] >= t]
        for _, row in subset.iterrows():
            preds[row["source1_entity_id"]].add(row["entity_id"])
        score = macro_f0_5(ground_truth, dict(preds))
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score
