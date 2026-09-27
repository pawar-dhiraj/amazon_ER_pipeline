"""Shared helpers for the standalone test_*.py scripts. NOT part of
the production pipeline (pipeline.py doesn't import this) -- these
exist purely so each test_*.py file doesn't repeat the same sample-
loading boilerplate.

Run any of the test scripts from the student_resource/ root with:
    python -m src.test_features
    python -m src.test_train
    python -m src.test_evaluate
    python -m src.test_postprocess
"""
import pandas as pd

from . import blocking, evaluate, preprocessing


def load_small_sample(data_dir="dataset", n_s1=200, n_extra_candidates=2000, random_state=0):
    """Loads a small slice of the TRAIN split (never test) and
    preprocesses it. Kept deliberately tiny so every test script here
    runs in seconds, not minutes -- these are for catching bugs in
    logic, not for measuring real recall/accuracy (use the full
    dataset for that, via test_blocking_recall.py).

    IMPORTANT: a plain independent .head(n)/.head(n) slice of S1 and
    S2/S3 almost never overlaps with the ground truth, since the files
    aren't row-aligned -- that was silently producing zero positive
    pairs in every downstream test. Fixed here by explicitly pulling in
    each sampled S1 entity's TRUE matching S2/S3 rows (via the ground
    truth file) on top of a random slice of extra rows to act as
    negatives/distractors."""
    s1_full = pd.read_csv(f"{data_dir}/train/train_source1.tsv", sep="\t", dtype=str)
    s2_full = pd.read_csv(f"{data_dir}/train/train_source2.tsv", sep="\t", dtype=str)
    s3_full = pd.read_csv(f"{data_dir}/train/train_source3.tsv", sep="\t", dtype=str)
    ground_truth = evaluate.parse_ground_truth_tsv(f"{data_dir}/train/train_ground_truth.tsv")

    s1 = s1_full.sample(min(n_s1, len(s1_full)), random_state=random_state)

    needed_s2, needed_s3 = set(), set()
    for s1_id in s1["entity_id"]:
        for matched_id in ground_truth.get(s1_id, set()):
            if matched_id.startswith("S2-"):
                needed_s2.add(matched_id)
            elif matched_id.startswith("S3-"):
                needed_s3.add(matched_id)

    s2_needed = s2_full[s2_full["entity_id"].isin(needed_s2)]
    s3_needed = s3_full[s3_full["entity_id"].isin(needed_s3)]
    s2_extra = s2_full.sample(min(n_extra_candidates, len(s2_full)), random_state=random_state)
    s3_extra = s3_full.sample(min(n_extra_candidates, len(s3_full)), random_state=random_state)

    s2 = pd.concat([s2_needed, s2_extra]).drop_duplicates(subset="entity_id")
    s3 = pd.concat([s3_needed, s3_extra]).drop_duplicates(subset="entity_id")

    suffix_map = preprocessing.build_suffix_dictionary(s1["business_name"])
    s1c = preprocessing.preprocess_dataframe(s1, suffix_map)
    s2c = preprocessing.preprocess_dataframe(s2, suffix_map)
    s3c = preprocessing.preprocess_dataframe(s3, suffix_map)
    return s1c, s2c, s3c, ground_truth, suffix_map


def build_small_candidates(s1c, s2c, s3c, top_k=15, use_embedding_blocking=False):
    """use_embedding_blocking defaults to False here -- these are quick
    logic checks, not a real recall measurement, so skip the slow
    FAISS/embedding path unless a specific test script needs it."""
    return blocking.generate_candidates(
        s1c, s2c, s3c, top_k=top_k, use_embedding_blocking=use_embedding_blocking
    )
