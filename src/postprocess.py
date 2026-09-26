"""
Step 5 -- Decision & Post-processing.

- Applies the tuned global threshold from evaluate.tune_threshold().
- Franchise-guard: tightens the effective threshold specifically for
  pairs with high name similarity but low address similarity -- this
  is the precision-killing failure mode (two branches of the same
  chain, different addresses) that a single global threshold misses,
  and it is exactly the case F0.5's 2x precision weighting punishes
  hardest.
- Graph consistency: a light-touch pass, not a full clustering
  algorithm -- if S1-A links to a candidate that ALSO links strongly
  to another candidate of the same S1 anchor's neighborhood, that
  mutual reinforcement nudges borderline pairs, without overriding
  confident model decisions.
"""
from collections import defaultdict

import pandas as pd

FRANCHISE_NAME_SIM_HIGH = 0.90
FRANCHISE_ADDR_SIM_LOW = 0.35
FRANCHISE_EXTRA_MARGIN = 0.08  # require this much MORE probability to accept


def apply_franchise_guard(scored_df: pd.DataFrame, base_threshold: float) -> pd.Series:
    """Returns a boolean Series: True = accept as a match."""
    is_franchise_risk = (
        (scored_df["name_jaro_winkler"] >= FRANCHISE_NAME_SIM_HIGH)
        & (scored_df["addr_jaro_winkler"] <= FRANCHISE_ADDR_SIM_LOW)
        & (scored_df["both_have_address"] == 1.0)
    )
    effective_threshold = base_threshold + is_franchise_risk * FRANCHISE_EXTRA_MARGIN
    return scored_df["match_probability"] >= effective_threshold


def graph_consistency_boost(scored_df: pd.DataFrame, accepted: pd.Series,
                             base_threshold: float, boost_margin: float = 0.03) -> pd.Series:
    """Very light mutual-reinforcement pass: if an S1 entity already
    has >=1 accepted match, borderline additional candidates for that
    SAME S1 entity get accepted at a slightly lower effective threshold
    (captures "this S1 clearly has real matches in this source region,
    a close second candidate is plausible too").

    NOW IMPLEMENTED (previously a no-op stub) -- but still opt-in: call
    this only after comparing `evaluate.macro_f0_5` WITH vs WITHOUT it
    on your own validation split (see pipeline.py's --use-graph-boost
    flag). An overly aggressive version of this directly hurts
    precision under F0.5, so treat the comparison as a checkpoint you
    must clear before trusting it on the actual test predictions."""
    result = accepted.copy()
    accepted_counts = scored_df.loc[accepted, "source1_entity_id"].value_counts()

    borderline = (~accepted) & (
        scored_df["match_probability"] >= (base_threshold - boost_margin)
    )
    for idx in scored_df[borderline].index:
        s1_id = scored_df.loc[idx, "source1_entity_id"]
        if accepted_counts.get(s1_id, 0) >= 1:
            result.loc[idx] = True
    return result


def build_output_frames(scored_df: pd.DataFrame, all_s1_ids, base_threshold: float,
                         use_franchise_guard: bool = True,
                         use_graph_boost: bool = False) -> tuple:
    """Returns (matching_results_df, candidate_pairs_df) in the exact
    schema the challenge requires. use_graph_boost defaults to False --
    validate it against your own held-out macro F0.5 before enabling it
    on real test predictions (see pipeline.py)."""
    if use_franchise_guard:
        accepted = apply_franchise_guard(scored_df, base_threshold)
    else:
        accepted = scored_df["match_probability"] >= base_threshold

    if use_graph_boost:
        accepted = graph_consistency_boost(scored_df, accepted, base_threshold)

    matches = defaultdict(list)
    candidates = defaultdict(list)
    for idx, row in scored_df.iterrows():
        candidates[row["source1_entity_id"]].append(row["entity_id"])
        if accepted.loc[idx]:
            matches[row["source1_entity_id"]].append(row["entity_id"])

    def _to_df(mapping):
        rows = []
        for s1_id in all_s1_ids:
            ids = mapping.get(s1_id, [])
            # de-duplicate while preserving order
            seen, deduped = set(), []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    deduped.append(i)
            rows.append({"source1_entity_id": s1_id, "ids": ",".join(deduped)})
        return pd.DataFrame(rows)

    matching_df = _to_df(matches).rename(columns={"ids": "matched_entity_ids"})
    candidate_df = _to_df(candidates).rename(columns={"ids": "candidate_entity_ids"})
    return matching_df, candidate_df


def write_outputs(matching_df: pd.DataFrame, candidate_df: pd.DataFrame, output_dir: str):
    import os
    os.makedirs(output_dir, exist_ok=True)
    matching_df.to_csv(f"{output_dir}/matching_results.tsv", sep="\t", index=False)
    candidate_df.to_csv(f"{output_dir}/candidate_pairs.tsv", sep="\t", index=False)
