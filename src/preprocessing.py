"""
Step 1 -- Preprocessing / Normalization.

- Unicode NFKC normalization (critical for French accented characters).
- Numeric tokens (postal codes, street numbers) pulled out BEFORE any
  fuzzy text matching touches them -- they are high-precision signals
  and must not be destroyed by lowercasing/punctuation-stripping.
- A legal-suffix / abbreviation dictionary that is MINED FROM THE
  TRAINING DATA ITSELF (frequency-based), not hand-coded per country.
  This is the deliberate departure from the problem statement's own
  hint to build country-specific rules: a data-driven dictionary is
  the only version of this idea that has a chance of saying anything
  useful about French legal suffixes (SARL, SAS, SA...) it never saw
  labeled examples for.
"""
import re
import unicodedata
from collections import Counter

import pandas as pd

NUMERIC_TOKEN_RE = re.compile(r"\d+")
NON_ALNUM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

# A small seed list of well-known legal-form abbreviations across the
# training countries (US, India) plus common global forms. This is a
# STARTING point only -- build_suffix_dictionary() below extends it
# using frequency mining so it is not purely hand-coded.
SEED_SUFFIX_MAP = {
    "corp": "corporation",
    "corp.": "corporation",
    "inc": "incorporated",
    "inc.": "incorporated",
    "llc": "limited liability company",
    "ltd": "limited",
    "ltd.": "limited",
    "pvt": "private",
    "pvt.": "private",
    "co": "company",
    "co.": "company",
    "&": "and",
}


def normalize_unicode(text: str) -> str:
    """NFKC normalization -- must run before any tokenization."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def extract_numeric_tokens(text: str) -> list:
    """Pull out digit sequences (postal codes, street numbers, unit
    numbers) so they survive as their own feature instead of being
    silently merged/lost during text normalization."""
    text = normalize_unicode(text)
    return NUMERIC_TOKEN_RE.findall(text)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Numeric
    tokens are intentionally KEPT in this string too (in addition to
    being extracted separately) so downstream string-similarity
    features still see them."""
    text = normalize_unicode(text)
    text = text.lower()
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_suffix_dictionary(business_names: pd.Series, min_count: int = 20) -> dict:
    """Frequency-mine trailing tokens across ALL training business
    names (not filtered by country) to extend the seed suffix map.
    A trailing token that appears often enough across many distinct
    business names is very likely a legal-form suffix rather than
    part of the brand name itself.
    """
    suffix_map = dict(SEED_SUFFIX_MAP)
    trailing_counter = Counter()
    for name in business_names.dropna():
        cleaned = clean_text(name)
        tokens = cleaned.split()
        if len(tokens) >= 2:
            trailing_counter[tokens[-1]] += 1

    for token, count in trailing_counter.items():
        if count >= min_count and token not in suffix_map and len(token) <= 6:
            # Candidate legal-suffix token: frequent, short, appears
            # at the END of many different business names.
            suffix_map[token] = token  # canonicalize to itself (dedup signal)
    return suffix_map


def normalize_business_name(name: str, suffix_map: dict) -> str:
    cleaned = clean_text(name)
    tokens = cleaned.split()
    tokens = [suffix_map.get(t, t) for t in tokens]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, suffix_map: dict) -> pd.DataFrame:
    """Adds normalized columns + extracted numeric-token columns.
    Does not mutate original business_name / business_address."""
    out = df.copy()
    out["name_norm"] = out["business_name"].apply(
        lambda x: normalize_business_name(x, suffix_map)
    )
    out["address_norm"] = out["business_address"].apply(clean_text)
    out["name_numeric_tokens"] = out["business_name"].apply(extract_numeric_tokens)
    out["address_numeric_tokens"] = out["business_address"].apply(extract_numeric_tokens)
    out["has_address"] = out["business_address"].notna() & (
        out["business_address"].astype(str).str.strip() != ""
    )
    return out
