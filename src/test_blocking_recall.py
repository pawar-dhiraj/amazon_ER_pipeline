"""
Run: python -m src.test_blocking_recall [--n-s1 20000] [--top-k 15]
                                          [--no-embedding-blocking] [--n-jobs -1]

Checks Step 2 (src/blocking.py) recall against the TRAIN ground truth
on a configurable-size sample -- NOT the full 13.8M rows. This is the
single most important check in the whole pipeline: a true match that
never becomes a candidate here can never be recovered by the matcher,
no matter how good it is downstream.

Recommended sequence after any change to blocking.py:
  1. Run this with a small --n-s1 (2,000-20,000) first -- fast, catches
     crashes and gross recall problems in under a minute.
  2. Once that looks right, bump --n-s1 up (e.g. 100,000-200,000) for a
     more statistically meaningful recall number before committing to
     the full 13.8M-row run.
"""
import argparse
import time
from collections import defaultdict

import pandas as pd

from . import blocking, evaluate, preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--n-s1", type=int, default=20_000,
                         help="Number of Source-1 rows to sample. Keep this small "
                              "(a few thousand to tens of thousands) for a quick "
                              "check -- this script is NOT meant to run on the full "
                              "13.8M rows.")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--no-embedding-blocking", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print("Loading datasets...")
    s1_full = pd.read_csv(f"{args.data_dir}/train/train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{args.data_dir}/train/train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{args.data_dir}/train/train_source3.tsv", sep="\t", dtype=str)
    ground_truth = evaluate.parse_ground_truth_tsv(f"{args.data_dir}/train/train_ground_truth.tsv")
    print(f"S1 rows (full): {len(s1_full)}   S2 rows: {len(s2)}   S3 rows: {len(s3)}")

    s1 = s1_full.sample(min(args.n_s1, len(s1_full)), random_state=args.random_state)
    print(f"S1 sample for this run: {len(s1)}")

    print("\nBuilding suffix dictionary...")
    suffix_map = preprocessing.build_suffix_dictionary(s1["business_name"])

    print("Preprocessing...")
    s1c = preprocessing.preprocess_dataframe(s1, suffix_map)
    s2c = preprocessing.preprocess_dataframe(s2, suffix_map)
    s3c = preprocessing.preprocess_dataframe(s3, suffix_map)

    print(f"\nGenerating candidates "
          f"(embedding_blocking={not args.no_embedding_blocking}, n_jobs={args.n_jobs})...")
    start = time.time()
    candidates = blocking.generate_candidates(
        s1c, s2c, s3c, top_k=args.top_k,
        use_embedding_blocking=not args.no_embedding_blocking,
        n_jobs=args.n_jobs,
    )
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s ({elapsed / max(len(s1c), 1):.4f}s per S1 entity)")

    sizes = [len(v) for v in candidates.values()]
    zero_count = sum(1 for s in sizes if s == 0)
    print(f"\navg candidates per S1 entity: {sum(sizes) / len(sizes):.1f}")
    print(f"entities with zero candidates: {zero_count} / {len(sizes)}")

    print("\n--- Recall against ground truth (this sample's S1 entities only) ---")
    hits, total = 0, 0
    misses = []
    for s1_id in s1c["entity_id"]:
        true_ids = ground_truth.get(s1_id, set())
        if not true_ids:
            continue  # singleton -- nothing to recall
        total += len(true_ids)
        found = true_ids & candidates.get(s1_id, set())
        hits += len(found)
        if len(found) < len(true_ids):
            misses.append((s1_id, true_ids - found))

    if total == 0:
        print("No S1 entities with true matches in this sample -- increase --n-s1, "
              "or this sample happened to draw only singletons.")
        return

    recall = hits / total
    print(f"blocking recall: {hits}/{total} = {recall:.3f}")

    if recall < 0.90:
        print("\nCHECK THIS -- recall below 0.90 means the matcher downstream can never "
              "recover these missed true matches, regardless of how well it's tuned.")
        print(f"First few missed cases (up to 5) for manual inspection:")
        for s1_id, missed_ids in misses[:5]:
            row = s1c[s1c["entity_id"] == s1_id].iloc[0]
            print(f"  {s1_id}  name='{row['business_name']}'  missed={missed_ids}")
    else:
        print("OK -- recall looks healthy for this sample size.")


if __name__ == "__main__":
    main()
