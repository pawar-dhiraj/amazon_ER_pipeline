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


def load_small_sample(data_dir="dataset", n_s1=200, n_cand=2000):
    """Loads a small slice of the TRAIN split (never test) and
    preprocesses it. Kept deliberately tiny so every test script here
    runs in seconds, not minutes -- these are for catching bugs in
    logic, not for measuring real recall/accuracy (use the full
    dataset for that, via a dedicated recall test like your existing
    test_blocking_recall.py)."""
    s1 = pd.read_csv(f"{data_dir}/train/train_source1.tsv", sep="\t", dtype=str).head(n_s1)
    s2 = pd.read_csv(f"{data_dir}/train/train_source2.tsv", sep="\t", dtype=str).head(n_cand)
    s3 = pd.read_csv(f"{data_dir}/train/train_source3.tsv", sep="\t", dtype=str).head(n_cand)
    ground_truth = evaluate.parse_ground_truth_tsv(f"{data_dir}/train/train_ground_truth.tsv")

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
