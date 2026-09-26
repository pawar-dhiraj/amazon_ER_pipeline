import pandas as pd
from src import preprocessing

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
).head(100)

suffix_map = preprocessing.build_suffix_dictionary(
    s1["business_name"]
)

s1_clean = preprocessing.preprocess_dataframe(
    s1,
    suffix_map
)

print(
    s1_clean[
        [
            "business_name",
            "name_norm",
            "business_address",
            "address_norm"
        ]
    ].head(10)
)

print("suffix_map size:", len(suffix_map))