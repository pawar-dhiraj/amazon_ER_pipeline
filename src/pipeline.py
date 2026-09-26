"""
End-to-end orchestrator.

Usage (from the project root, with dataset/ laid out as the README describes):

    python -m src.pipeline --data-dir dataset --output-dir output --model-path model.txt

Runs, in order: Step 0 (split) -> Step 1 (preprocess) -> Step 2 (block)
-> Step 3 (features) -> Step 4 (train + tune threshold on held-out
validation) -> Step 5 (post-process) -> Step 6 (SHAP + surrogate-tree
sanity checks) -> writes matching_results.tsv + candidate_pairs.tsv to
--output-dir.

Pass --use-graph-boost to enable the mutual-reinforcement post-process
pass -- it is compared against the plain threshold on your OWN
validation split first (both scores are printed) so you can see its
real effect before it's applied to the actual test predictions.
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from . import blocking, evaluate, features, postprocess, preprocessing, train as train_mod


def load_source_files(data_dir: str, split: str):
    s1 = pd.read_csv(f"{data_dir}/{split}/{split}_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{data_dir}/{split}/{split}_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{data_dir}/{split}/{split}_source3.tsv", sep="\t", dtype=str)
    return s1, s2, s3


def run(data_dir: str, output_dir: str, model_path: str, top_k: int = 15,
        val_fraction: float = 0.2, random_state: int = 42, use_graph_boost: bool = False,
        use_embedding_blocking: bool = True, n_jobs: int = -1):
    # ---- Step 0: load + validation split (stratified by country) ----
    train_s1, train_s2, train_s3 = load_source_files(data_dir, "train")
    ground_truth = evaluate.parse_ground_truth_tsv(f"{data_dir}/train/train_ground_truth.tsv")

    fit_s1, val_s1 = train_test_split(
        train_s1, test_size=val_fraction, random_state=random_state,
        stratify=train_s1["country"] if train_s1["country"].nunique() > 1 else None,
    )

    # ---- Step 1: preprocessing (suffix dict mined from the FIT split only) ----
    suffix_map = preprocessing.build_suffix_dictionary(fit_s1["business_name"])
    fit_s1 = preprocessing.preprocess_dataframe(fit_s1, suffix_map)
    val_s1 = preprocessing.preprocess_dataframe(val_s1, suffix_map)
    train_s2 = preprocessing.preprocess_dataframe(train_s2, suffix_map)
    train_s3 = preprocessing.preprocess_dataframe(train_s3, suffix_map)
    others = pd.concat([train_s2, train_s3], ignore_index=True)

    # ---- Step 2: blocking (fit split) ----
    fit_candidates = blocking.generate_candidates(
        fit_s1, train_s2, train_s3, top_k=top_k,
        use_embedding_blocking=use_embedding_blocking, n_jobs=n_jobs,
    )
    val_candidates = blocking.generate_candidates(
        val_s1, train_s2, train_s3, top_k=top_k,
        use_embedding_blocking=use_embedding_blocking, n_jobs=n_jobs,
    )

    # ---- Step 3: features ----
    fit_features = features.build_feature_matrix(fit_candidates, fit_s1, others)
    val_features = features.build_feature_matrix(val_candidates, val_s1, others)

    # ---- Step 4: train + tune threshold on validation ----
    labeled = train_mod.build_training_labels(fit_features, ground_truth)
    model = train_mod.train_matcher(labeled)
    val_scored = train_mod.predict_probabilities(model, val_features)
    best_threshold, best_score = evaluate.tune_threshold(val_scored, ground_truth)
    print(f"[validation] best_threshold={best_threshold:.2f}  macro_F0.5={best_score:.4f}")

    # ---- Graph-consistency boost: measure its REAL effect before trusting it ----
    val_gt_ids = {row["entity_id"] for _, row in val_s1.iterrows()}
    val_accepted_plain = val_scored["match_probability"] >= best_threshold
    val_accepted_boosted = postprocess.graph_consistency_boost(
        val_scored, val_accepted_plain, best_threshold
    )

    def _to_pred_dict(scored_df, accepted_mask):
        preds = {}
        for s1_id in val_gt_ids:
            preds[s1_id] = set(
                scored_df.loc[accepted_mask & (scored_df["source1_entity_id"] == s1_id), "entity_id"]
            )
        return preds

    score_plain = evaluate.macro_f0_5(ground_truth, _to_pred_dict(val_scored, val_accepted_plain))
    score_boosted = evaluate.macro_f0_5(ground_truth, _to_pred_dict(val_scored, val_accepted_boosted))
    print(f"[validation] macro_F0.5 without graph-boost: {score_plain:.4f}")
    print(f"[validation] macro_F0.5 with graph-boost:    {score_boosted:.4f}")
    if use_graph_boost and score_boosted < score_plain:
        print("WARNING: graph-boost lowered validation F0.5 -- consider running with "
              "use_graph_boost=False instead.")

    train_mod.save_model(model, model_path)

    # ---- Step 6 (sanity check): SHAP feature importance ----
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(labeled[train_mod.FEATURE_COLUMNS])
        importance = pd.Series(
            abs(shap_values).mean(axis=0), index=train_mod.FEATURE_COLUMNS
        ).sort_values(ascending=False)
        print("\n[SHAP] mean |impact| per feature:")
        print(importance.to_string())
    except ImportError:
        print("shap not installed -- skipping interpretability check")

    # ---- Step 6b (sanity check): surrogate decision-tree rule extraction ----
    # A shallow tree trained to mimic the GBDT's own decisions, so you have
    # a human-readable rule set to drop straight into Documentation_template.md
    # alongside the SHAP plot -- not used for actual predictions.
    from sklearn.tree import DecisionTreeClassifier, export_text
    fit_predictions = model.predict(labeled[train_mod.FEATURE_COLUMNS]) >= best_threshold
    surrogate = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    surrogate.fit(labeled[train_mod.FEATURE_COLUMNS], fit_predictions)
    print("\n[Surrogate tree] human-readable approximation of the matcher's rules:")
    print(export_text(surrogate, feature_names=train_mod.FEATURE_COLUMNS))

    # ---- Now run the same steps on the actual TEST set ----
    test_s1, test_s2, test_s3 = load_source_files(data_dir, "test")
    test_s1 = preprocessing.preprocess_dataframe(test_s1, suffix_map)
    test_s2 = preprocessing.preprocess_dataframe(test_s2, suffix_map)
    test_s3 = preprocessing.preprocess_dataframe(test_s3, suffix_map)
    test_others = pd.concat([test_s2, test_s3], ignore_index=True)

    test_candidates = blocking.generate_candidates(
        test_s1, test_s2, test_s3, top_k=top_k,
        use_embedding_blocking=use_embedding_blocking, n_jobs=n_jobs,
    )
    test_features = features.build_feature_matrix(test_candidates, test_s1, test_others)
    test_scored = train_mod.predict_probabilities(model, test_features)

    # ---- Step 5: post-process + write submission files ----
    matching_df, candidate_df = postprocess.build_output_frames(
        test_scored, test_s1["entity_id"].tolist(), best_threshold,
        use_franchise_guard=True, use_graph_boost=use_graph_boost,
    )
    postprocess.write_outputs(matching_df, candidate_df, output_dir)
    print(f"\nWrote {output_dir}/matching_results.tsv and {output_dir}/candidate_pairs.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-path", default="model.txt")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--use-graph-boost", action="store_true",
                         help="Enable graph-consistency boosting on the final test "
                              "predictions. Check the printed validation comparison "
                              "first -- only pass this if boosted score >= plain score.")
    parser.add_argument("--no-embedding-blocking", action="store_true",
                         help="Skip the FAISS embedding blocker (fast, hashing-only "
                              "candidates) -- useful for a quick end-to-end debug pass "
                              "on 10M+ row datasets before committing to the slower "
                              "full run with embeddings.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="CPU cores to use for blocking (-1 = all cores minus one, "
                              "the default). Set to 1 to run single-process, which is "
                              "slower but gives cleaner tracebacks when debugging.")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.model_path, top_k=args.top_k,
        use_graph_boost=args.use_graph_boost,
        use_embedding_blocking=not args.no_embedding_blocking, n_jobs=args.n_jobs)
