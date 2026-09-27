"""
Run: python -m src.test_preprocess

Checks Step 1 (src/preprocessing.py) on a small sample: text
normalization, numeric-token extraction, and the frequency-mined
suffix dictionary. See PIPELINE_HOWTO.md's `preprocessing.py` section
for what each check means.
"""
import pandas as pd

from . import preprocessing


def main():
    print("Loading a small sample of train_source1.tsv...")
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str).sample(
        500, random_state=0
    )

    print("Building suffix dictionary...")
    suffix_map = preprocessing.build_suffix_dictionary(s1["business_name"])
    print(f"suffix_map size: {len(suffix_map)}")

    print("\nPreprocessing...")
    s1c = preprocessing.preprocess_dataframe(s1, suffix_map)

    print("\nSample before/after (10 rows):")
    print(s1c[["business_name", "name_norm", "business_address", "address_norm"]].head(10)
          .to_string())

    print("\n--- Checks ---")
    suffix_size_ok = 5 <= len(suffix_map) <= 2000
    print(f"suffix_map size in sane range (5-2000): "
          f"{'OK' if suffix_size_ok else 'CHECK THIS -- too small (rarely normalizing) or too big (over-matching real words)'}")

    empty_after_clean = ((s1c["business_name"].notna()) & (s1c["business_name"].str.strip() != "")
                          & (s1c["name_norm"].str.strip() == "")).sum()
    print(f"non-empty names that became empty after cleaning: {empty_after_clean}  "
          f"{'OK' if empty_after_clean == 0 else 'CHECK THIS -- likely an unhandled Unicode/script case'}")

    has_address_matches = (s1c["has_address"] == s1c["business_address"].notna()
                            & (s1c["business_address"].astype(str).str.strip() != "")).all()
    print(f"has_address flag consistent with raw column: {'OK' if has_address_matches else 'CHECK THIS'}")

    numeric_extracted = s1c["name_numeric_tokens"].apply(len).sum() + \
        s1c["address_numeric_tokens"].apply(len).sum()
    print(f"total numeric tokens extracted across sample: {numeric_extracted}  "
          f"(0 across a 500-row sample would be suspicious -- postal codes/street numbers "
          f"are common)")

    print("\nSpot-check a few known legal-suffix cases by eye above -- e.g. names ending in "
          "'Pvt Ltd', 'Corp', 'LLC' should normalize toward a consistent token.")


if __name__ == "__main__":
    main()
