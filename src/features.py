"""
Step 3 -- Feature Engineering.

A compact, non-redundant feature set per (S1, candidate) pair. Kept
deliberately small (~15 features) rather than throwing in every
string-distance metric that exists -- redundant, highly-correlated
features mostly add overfitting risk on a modest-sized labeled set,
not signal (this is the PCA/decorrelation instinct applied at
design-time instead of after the fact).
"""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from rapidfuzz.fuzz import token_sort_ratio

try:
    import fasttext
    _LID_AVAILABLE = True
except ImportError:
    _LID_AVAILABLE = False

_LID_MODEL = None


def _char_ngrams(text, n=3):
    text = text or ""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _numeric_overlap(tokens_a: list, tokens_b: list) -> float:
    set_a, set_b = set(tokens_a or []), set(tokens_b or [])
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))


def _lang_id(text: str) -> str:
    """Cheap language-ID feature: a distribution-shift proxy that
    generalizes to France without ever having seen labeled French
    examples, since it only relies on the text itself. Returns
    'unknown' when fasttext's lid model is not installed/loaded."""
    global _LID_MODEL
    if not _LID_AVAILABLE or not text.strip():
        return "unknown"
    if _LID_MODEL is None:
        try:
            # Requires lid.176.bin downloaded separately (see README).
            _LID_MODEL = fasttext.load_model("lid.176.bin")
        except Exception:
            return "unknown"
    try:
        pred = _LID_MODEL.predict(text.replace("\n", " "))
        return pred[0][0].replace("__label__", "")
    except Exception:
        return "unknown"


def compute_pair_features(row_s1: pd.Series, row_cand: pd.Series,
                           emb_sim_name: float = None, emb_sim_addr: float = None) -> dict:
    name1, name2 = row_s1["name_norm"], row_cand["name_norm"]
    addr1, addr2 = row_s1["address_norm"], row_cand["address_norm"]

    ngrams1, ngrams2 = _char_ngrams(name1), _char_ngrams(name2)

    feats = {
        "name_jaro_winkler": JaroWinkler.normalized_similarity(name1, name2),
        "name_levenshtein": Levenshtein.normalized_similarity(name1, name2),
        "name_token_sort": token_sort_ratio(name1, name2) / 100.0,
        "name_char_ngram_jaccard": _jaccard(ngrams1, ngrams2),
        "addr_jaro_winkler": JaroWinkler.normalized_similarity(addr1, addr2),
        "addr_levenshtein": Levenshtein.normalized_similarity(addr1, addr2),
        "addr_char_ngram_jaccard": _jaccard(_char_ngrams(addr1), _char_ngrams(addr2)),
        "numeric_token_overlap": _numeric_overlap(
            row_s1["name_numeric_tokens"] + row_s1["address_numeric_tokens"],
            row_cand["name_numeric_tokens"] + row_cand["address_numeric_tokens"],
        ),
        "s1_has_address": float(row_s1["has_address"]),
        "cand_has_address": float(row_cand["has_address"]),
        "both_have_address": float(row_s1["has_address"] and row_cand["has_address"]),
        "same_country": float(row_s1.get("country") == row_cand.get("country")),
        "name_len_ratio": min(len(name1), len(name2)) / max(len(name1), len(name2), 1),
    }

    # Language-ID feature: a distribution-shift proxy that generalizes to
    # France without ever having seen a labeled French example, since it
    # only relies on the text itself. Wired in here (previously defined
    # but never called -- fixed). Degrades to "unknown" (never matches)
    # when lid.176.bin isn't installed, so the feature is always safe to
    # include even without the optional dependency.
    lang1 = _lang_id(f"{name1} {addr1}")
    lang2 = _lang_id(f"{name2} {addr2}")
    feats["same_language"] = float(lang1 == lang2 and lang1 != "unknown")

    # Embedding similarities, if precomputed upstream (see pipeline.py) --
    # kept optional so the feature builder still works with TF-IDF-only blocking.
    feats["emb_sim_name"] = emb_sim_name if emb_sim_name is not None else np.nan
    feats["emb_sim_addr"] = emb_sim_addr if emb_sim_addr is not None else np.nan

    return feats


def build_feature_matrix(candidate_pairs: dict, df_s1: pd.DataFrame,
                          df_others: pd.DataFrame,
                          emb_sim_lookup: dict = None) -> pd.DataFrame:
    """candidate_pairs: {s1_id: set(candidate_ids)}
    df_others: concatenated, preprocessed S2 + S3 dataframe (indexed by entity_id column).
    emb_sim_lookup: optional {(s1_id, cand_id): (sim_name, sim_addr)}.
    Returns a flat DataFrame, one row per (s1_id, cand_id) pair, with
    an `entity_id` (=cand_id) and `source1_entity_id` column plus features.
    """
    s1_lookup = df_s1.set_index("entity_id")
    other_lookup = df_others.set_index("entity_id")

    rows = []
    for s1_id, cand_ids in candidate_pairs.items():
        if s1_id not in s1_lookup.index:
            continue
        row_s1 = s1_lookup.loc[s1_id]
        for cand_id in cand_ids:
            if cand_id not in other_lookup.index:
                continue
            row_cand = other_lookup.loc[cand_id]
            emb_name, emb_addr = (None, None)
            if emb_sim_lookup is not None:
                emb_name, emb_addr = emb_sim_lookup.get((s1_id, cand_id), (None, None))
            feats = compute_pair_features(row_s1, row_cand, emb_name, emb_addr)
            feats["source1_entity_id"] = s1_id
            feats["entity_id"] = cand_id
            rows.append(feats)

    return pd.DataFrame(rows)
