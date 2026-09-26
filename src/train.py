"""
Step 4 -- Matcher training.

Primary matcher: LightGBM on the Step-3 engineered features.
Deliberately NOT a neural net / LLM as the primary decision-maker --
GBDT-on-features repeatedly matches or beats deep matchers on
structured, moderate-sized entity-resolution benchmarks (see
Mudgal et al., SIGMOD 2018, and the Foursquare Location Matching
winning solutions), trains in minutes on a laptop, and sidesteps the
challenge's model-size/license constraint entirely since a GBDT
ensemble of tree splits isn't parameterized the way the "<=8B
parameters" rule is written for.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "name_jaro_winkler", "name_levenshtein", "name_token_sort",
    "name_char_ngram_jaccard", "addr_jaro_winkler", "addr_levenshtein",
    "addr_char_ngram_jaccard", "numeric_token_overlap", "s1_has_address",
    "cand_has_address", "both_have_address", "same_country",
    "name_len_ratio", "emb_sim_name", "emb_sim_addr", "same_language",
]


def build_training_labels(feature_df: pd.DataFrame, ground_truth: dict) -> pd.DataFrame:
    """Labels every (source1_entity_id, entity_id) candidate pair as
    1 (true match) or 0 (non-match), using the ground truth. Only
    candidate pairs that survived blocking get a row here -- a true
    match that blocking missed simply cannot be learned from (this is
    exactly why blocking recall matters more than anything in Step 2).
    """
    labeled = feature_df.copy()
    labeled["label"] = labeled.apply(
        lambda r: int(r["entity_id"] in ground_truth.get(r["source1_entity_id"], set())),
        axis=1,
    )
    return labeled


def train_matcher(labeled_df: pd.DataFrame, feature_columns=FEATURE_COLUMNS,
                   params: dict = None) -> lgb.Booster:
    """Trains a binary LightGBM classifier. Uses class-weighting
    (via scale_pos_weight-equivalent 'is_unbalance') since true
    matches are a small minority of candidate pairs after blocking --
    without this the model trivially predicts "no match" everywhere."""
    X = labeled_df[feature_columns]
    y = labeled_df["label"]

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "is_unbalance": True,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(default_params, train_data, num_boost_round=300)
    return model


def predict_probabilities(model: lgb.Booster, feature_df: pd.DataFrame,
                           feature_columns=FEATURE_COLUMNS) -> pd.DataFrame:
    out = feature_df.copy()
    out["match_probability"] = model.predict(out[feature_columns])
    return out


def save_model(model: lgb.Booster, path: str):
    model.save_model(path)


def load_model(path: str) -> lgb.Booster:
    return lgb.Booster(model_file=path)
