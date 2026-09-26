"""
Run: python -m src.test_postprocess

Checks Step 5 (src/postprocess.py) on a small sample, applying the
SAME structural rules the challenge's own utils/validate_submission.py
enforces -- catches format bugs on a tiny sample before you spend real
time (or a real submission) finding them on the full run.
"""
import pandas as pd

from . import features, postprocess
from . import train as train_mod
from .test_utils import build_small_candidates, load_small_sample


def main():
    s1c, s2c, s3c, ground_truth, suffix_map = load_small_sample()
    others = pd.concat([s2c, s3c], ignore_index=True)
    candidates = build_small_candidates(s1c, s2c, s3c)
    fm = features.build_feature_matrix(candidates, s1c, others)
    labeled = train_mod.build_training_labels(fm, ground_truth)

    if labeled["label"].sum() == 0:
        print("No positive pairs in this sample -- widen n_s1/n_cand in "
              "test_utils.load_small_sample() first (postprocess needs a trained "
              "model with real signal to test against meaningfully).")
        return

    model = train_mod.train_matcher(labeled)
    scored = train_mod.predict_probabilities(model, fm)

    matching_df, candidate_df = postprocess.build_output_frames(
        scored, s1c["entity_id"].tolist(), base_threshold=0.5
    )

    print("--- Checks (same rules as utils/validate_submission.py) ---")

    row_count_ok = len(matching_df) == len(s1c)
    print(f"matching_results rows: {len(matching_df)}  (expected: {len(s1c)})  "
          f"{'OK' if row_count_ok else 'STOP -- row count mismatch, entities are missing'}")

    dup_found = any(
        len(ids := [i for i in row.split(",") if i]) != len(set(ids))
        for row in matching_df["matched_entity_ids"]
    )
    print(f"duplicate IDs within any single row: {'FOUND -- bug!' if dup_found else 'none, OK'}")

    dup_s1_rows = matching_df["source1_entity_id"].duplicated().any()
    print(f"duplicate source1_entity_id rows: {'FOUND -- bug!' if dup_s1_rows else 'none, OK'}")

    cand_lookup = dict(zip(candidate_df["source1_entity_id"], candidate_df["candidate_entity_ids"]))
    subset_violations = 0
    for _, row in matching_df.iterrows():
        matched = {i for i in row["matched_entity_ids"].split(",") if i}
        cands = {i for i in cand_lookup.get(row["source1_entity_id"], "").split(",") if i}
        if not matched.issubset(cands):
            subset_violations += 1
    print(f"matches not present in that entity's own candidate list: {subset_violations}  "
          f"{'OK' if subset_violations == 0 else '-- STOP, matched_entity_ids must be a subset of candidates'}")

    print("\nNext: write this sample to disk and run the challenge's own validator directly:")
    print("  from src import postprocess")
    print("  postprocess.write_outputs(matching_df, candidate_df, 'output_sample')")
    print("Then from student_resource/:")
    print("  python3 utils/validate_submission.py --matching output_sample/matching_results.tsv "
          "\\\n      --candidate output_sample/candidate_pairs.tsv --test-dir dataset/train")


if __name__ == "__main__":
    main()
