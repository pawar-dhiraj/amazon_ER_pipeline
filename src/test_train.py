"""
Run: python -m src.test_train

Checks Step 4 (src/train.py) on a small sample: label construction,
that positives actually exist, and that the model trains cleanly.
"""
import pandas as pd

from . import features
from . import train as train_mod
from .test_utils import build_small_candidates, load_small_sample


def main():
    print("Loading small sample...")
    s1c, s2c, s3c, ground_truth, suffix_map = load_small_sample()
    others = pd.concat([s2c, s3c], ignore_index=True)
    candidates = build_small_candidates(s1c, s2c, s3c)
    fm = features.build_feature_matrix(candidates, s1c, others)

    print("Building training labels from ground truth...")
    labeled = train_mod.build_training_labels(fm, ground_truth)
    n_pos = int(labeled["label"].sum())
    pos_rate = labeled["label"].mean()
    print(f"positive rate: {pos_rate:.4f}  ({n_pos} positive pairs out of {len(labeled)})")

    if n_pos == 0:
        print("\nSTOP -- zero positive pairs found. Two likely causes, check in this order:")
        print("  1. Sample too small: with only head(200)/head(2000) rows, the true matches")
        print("     for those specific S1 entities may simply not be included -- rerun with")
        print("     larger n_s1/n_cand in test_utils.load_small_sample() before assuming a bug.")
        print("  2. entity_id dtype/whitespace mismatch between your features and the ground")
        print("     truth file -- both should be loaded with dtype=str and no stray spaces.")
        return

    print("\nTraining LightGBM matcher...")
    model = train_mod.train_matcher(labeled)
    scored = train_mod.predict_probabilities(model, fm)
    print(scored["match_probability"].describe())

    print("\n--- Checks ---")
    prob_range_ok = scored["match_probability"].between(0, 1).all()
    print(f"  match_probability in [0,1]: {'OK' if prob_range_ok else 'CHECK THIS -- out of range'}")

    print("\nFeature importances (gain) -- nothing should sit at exactly 0 across the board,")
    print("that would suggest a broken feature rather than a genuinely unhelpful one:")
    importances = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=train_mod.FEATURE_COLUMNS,
    ).sort_values(ascending=False)
    print(importances.to_string())


if __name__ == "__main__":
    main()
