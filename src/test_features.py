"""
Run: python -m src.test_features

Checks Step 3 (src/features.py) on a small sample before you trust it
on the full dataset. See PIPELINE_HOWTO.md's `features.py` section for
what each check means.
"""
import pandas as pd

from . import features
from .test_utils import build_small_candidates, load_small_sample


def main():
    print("Loading small sample...")
    s1c, s2c, s3c, ground_truth, suffix_map = load_small_sample()
    others = pd.concat([s2c, s3c], ignore_index=True)

    print("Generating candidates (hashing blocker only -- fast)...")
    candidates = build_small_candidates(s1c, s2c, s3c)
    avg_cands = sum(len(v) for v in candidates.values()) / len(candidates)
    print(f"avg candidates per S1 entity: {avg_cands:.1f}")

    print("\nBuilding feature matrix...")
    fm = features.build_feature_matrix(candidates, s1c, others)
    print(fm.describe())

    print("\n--- Checks ---")
    similarity_columns = [
        "name_jaro_winkler", "name_levenshtein", "name_token_sort",
        "name_char_ngram_jaccard", "addr_jaro_winkler", "addr_levenshtein",
        "addr_char_ngram_jaccard",
    ]
    for col in similarity_columns:
        lo, hi = fm[col].min(), fm[col].max()
        ok = 0.0 <= lo and hi <= 1.0
        flag = "OK" if ok else "CHECK THIS -- outside [0,1], normalization bug upstream"
        print(f"  {col}: range [{lo:.3f}, {hi:.3f}]  {flag}")

    same_lang_rate = fm["same_language"].mean()
    print(f"  same_language non-zero rate: {same_lang_rate:.3f}  "
          f"(0.0 across the board is EXPECTED if lid.176.bin isn't installed -- not a bug)")

    emb_nan_rate = fm["emb_sim_name"].isna().mean()
    print(f"  emb_sim_name NaN rate: {emb_nan_rate:.3f}  "
          f"(currently always ~1.0 -- see PIPELINE_HOWTO.md's 'one thing to know' note)")

    if len(fm) == 0:
        print("\nSTOP: feature matrix is empty. Blocking produced zero candidates on this "
              "sample -- check test_blocking_recall.py output before debugging features.py.")


if __name__ == "__main__":
    main()
