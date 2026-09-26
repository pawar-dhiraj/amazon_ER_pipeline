import pandas as pd
from src import preprocessing, blocking, evaluate

# Load small samples
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
).head(100)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    dtype=str
).head(200)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    dtype=str
).head(200)

# Build suffix dictionary from S1
suffix_map = preprocessing.build_suffix_dictionary(
    s1["business_name"]
)

# Preprocess all three sources
s1_clean = preprocessing.preprocess_dataframe(
    s1,
    suffix_map
)

s2_clean = preprocessing.preprocess_dataframe(
    s2,
    suffix_map
)

s3_clean = preprocessing.preprocess_dataframe(
    s3,
    suffix_map
)

# Generate candidates
candidates = blocking.generate_candidates(
    s1_clean,
    s2_clean,
    s3_clean,
    top_k=15
)

# Candidate statistics
sizes = [len(v) for v in candidates.values()]

print("\n===== BLOCKING RESULTS =====")

print(
    "S1 entities:",
    len(s1_clean)
)

print(
    "Average candidates per S1 entity:",
    sum(sizes) / len(sizes)
)

print(
    "Maximum candidates:",
    max(sizes)
)

print(
    "Minimum candidates:",
    min(sizes)
)

print(
    "Entities with zero candidates:",
    sum(1 for s in sizes if s == 0)
)

print("\nFirst 5 candidate sets:")

for i, (s1_id, candidate_ids) in enumerate(candidates.items()):
    if i >= 5:
        break

    print(
        s1_id,
        "->",
        list(candidate_ids)[:10]
    )