# Pipeline How-To -- Implementation & Verification Guide

This is a companion to the challenge's own `README.md` -- it explains
how to bring up `src/` incrementally, what to check after each file,
and what "looks right" means before you move to the next step. Do NOT
run the full `pipeline.py` end-to-end on your first attempt -- work
through the checks below on a small sample first, or a silent bug in
an early stage (most commonly blocking) will waste your whole budget
of leaderboard submissions finding out later.

---

## 0. Setup & requirements

```bash
cd "student_resource"
python3 -m venv venv
source venv/bin/activate        # Windows (PowerShell): venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Core packages** (required, always installed): `pandas`, `numpy`,
`scikit-learn`, `scipy`, `lightgbm`, `rapidfuzz`, `shap`.

**Optional packages** (the pipeline degrades gracefully without them --
see the per-file notes below for what you lose):
- `sentence-transformers` + `faiss-cpu` -- needed for the multilingual
  embedding blocking/features. First run downloads
  `intfloat/multilingual-e5-large` (~1.1 GB, cached under
  `~/.cache/huggingface` after that). If you're on a slow connection
  or a low-RAM machine, edit `EMBEDDING_MODEL_NAME` in
  `src/blocking.py` to `"intfloat/multilingual-e5-small"` (~450 MB,
  faster, slightly weaker semantic quality).
- `fasttext-wheel` -- needed for the `same_language` feature. Also
  requires downloading the language-ID model file separately:
  ```bash
  curl -L -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
  ```
  Place `lid.176.bin` in the directory you run Python from (project
  root). Without it, `same_language` still computes -- it just always
  evaluates to `0` rather than raising an error.

**Python version**: 3.10+ recommended (LightGBM and rapidfuzz wheels are
most reliably prebuilt for 3.10-3.12; if you're on something older and
hit build errors, that's the first thing to check).

---

## 1. File-by-file: what to run, and what to check before moving on

Work through these in order. For each one, open a Python interactive
window in VS Code (or a scratch `.py` file you delete afterward) and
run the snippet on a SMALL slice of `train_source1.tsv` (e.g. `.head(100)`)
before ever touching the full dataset.

### `src/preprocessing.py`

```python
import pandas as pd
from src import preprocessing

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str).head(100)
suffix_map = preprocessing.build_suffix_dictionary(s1["business_name"])
s1_clean = preprocessing.preprocess_dataframe(s1, suffix_map)
print(s1_clean[["business_name", "name_norm", "business_address", "address_norm"]].head(10))
print("suffix_map size:", len(suffix_map))
```

**Check:**
- `name_norm` / `address_norm` are lowercase, punctuation-stripped, but
  still human-recognizable versions of the originals -- if a name
  becomes an empty string or gibberish, something is wrong with the
  cleaning regex for that row's characters (check for unusual Unicode).
- `suffix_map` size is somewhere in the dozens, not single digits (too
  few = frequency threshold too strict, few suffixes will normalize)
  and not in the thousands (too permissive = you're capturing real
  brand-name words as if they were legal suffixes -- lower `min_count`
  or raise it accordingly in `build_suffix_dictionary`).
- Spot-check a few known legal-suffix cases by eye ("XYZ Pvt Ltd" vs
  "XYZ Private Limited" should normalize toward the same tokens).

### `src/blocking.py`

```python
from src import blocking
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str).head(200)
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str).head(200)
s2_clean = preprocessing.preprocess_dataframe(s2, suffix_map)
s3_clean = preprocessing.preprocess_dataframe(s3, suffix_map)

candidates = blocking.generate_candidates(s1_clean, s2_clean, s3_clean, top_k=15)
sizes = [len(v) for v in candidates.values()]
print("avg candidates per S1 entity:", sum(sizes) / len(sizes))
print("entities with zero candidates:", sum(1 for s in sizes if s == 0))
```

**Check -- this is the single most important check in the whole
pipeline:**
- Average candidates per entity should be a small double-digit number
  (roughly `2 * top_k` if both blockers are firing, less if the
  embedding dependency isn't installed).
- **Recall check against ground truth** (do this on the full training
  set once the small-sample version looks sane):
  ```python
  from src import evaluate
  gt = evaluate.parse_ground_truth_tsv("dataset/train/train_ground_truth.tsv")
  hits, total = 0, 0
  for s1_id, true_ids in gt.items():
      if not true_ids:
          continue
      total += len(true_ids)
      hits += len(true_ids & candidates.get(s1_id, set()))
  print(f"blocking recall: {hits}/{total} = {hits/total:.3f}")
  ```
  If this recall is below ~0.9, no amount of matcher tuning downstream
  can fix it -- raise `top_k`, or check whether the embedding blocker
  is actually running (it silently falls back to TF-IDF-only if
  `sentence-transformers` isn't installed -- check `blocking._EMBEDDINGS_AVAILABLE`).
- If "entities with zero candidates" is high on rows you know have a
  true match in the ground truth, blocking is the bug, not the matcher.

### `src/features.py`

```python
from src import features
fm = features.build_feature_matrix(candidates, s1_clean, pd.concat([s2_clean, s3_clean]))
print(fm.describe())
```

**Check:**
- Every similarity column (`name_jaro_winkler`, `addr_jaro_winkler`,
  etc.) should range roughly `[0, 1]` -- values consistently at 0 or 1
  across the board suggest a normalization bug upstream (e.g. all
  strings ending up empty).
- `emb_sim_name` / `emb_sim_addr` are `NaN` unless you've wired
  precomputed embedding similarities into `emb_sim_lookup` -- this is
  expected with the current code (see the note at the bottom of this
  file) and LightGBM handles `NaN` natively, so it's not a bug to fix
  immediately.
- `same_language` should not be 100% zero once `lid.176.bin` is in
  place -- if it is, check the file path.

### `src/train.py`

```python
from src import train as train_mod
labeled = train_mod.build_training_labels(fm, gt)
print("positive rate:", labeled["label"].mean())
model = train_mod.train_matcher(labeled)
```

**Check:**
- Positive rate should be low (often under 10%) -- that's expected and
  is exactly why `is_unbalance=True` is set. **Zero positives** is a
  red flag: it means either blocking missed every true match in your
  sample, or an ID-format mismatch between `entity_id` columns and the
  ground truth file (check for accidental whitespace or dtype issues --
  everything should load as `dtype=str`).
- Training should complete in seconds on a sample this size without
  warnings about "no positive class" — if LightGBM complains, stop
  here and fix the label construction before going further.

### `src/evaluate.py`

Sanity-test against the challenge's own worked example before trusting
it on real data:

```python
from src import evaluate
score = evaluate.score_entity({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
print(score)  # should print 0.714... matching the problem statement's own example
```

**Check:** if this doesn't print ~0.714, do not proceed -- the metric
implementation itself is wrong and every downstream threshold-tuning
decision will be miscalibrated.

### `src/postprocess.py`

```python
from src import postprocess
scored = train_mod.predict_probabilities(model, fm)
matching_df, candidate_df = postprocess.build_output_frames(
    scored, s1_clean["entity_id"].tolist(), base_threshold=0.5
)
print(matching_df.head())
print("rows:", len(matching_df), "expected:", len(s1_clean))
```

**Check:**
- Row count of `matching_df` must exactly equal the number of S1
  entities you passed in -- this is one of the challenge's hard
  rejection rules.
- Spot-check a few `matched_entity_ids` cells for duplicate IDs (there
  shouldn't be any -- the de-dup logic in `_to_df` handles this, but
  verify on real data too).
- Every ID in `matching_df` should also appear in the corresponding row
  of `candidate_df` (matches are a subset of candidates) -- this is
  exactly what the challenge's own `utils/validate_submission.py`
  checks, so run that next.

---

## 2. Recommended order of implementation (with checkpoints)

1. **Preprocessing** on a 100-row sample -> eyeball `name_norm`/`address_norm`.
2. **Blocking** on a 100-200 row sample -> check recall against ground
   truth on that sample. Do not scale up until recall looks reasonable.
3. **Features** on the same sample -> check value ranges, no unexpected
   all-zero or all-NaN columns.
4. **Train** on the same sample -> check positive rate is non-zero,
   training completes cleanly.
5. **Evaluate** -> run the worked-example sanity check above
   independent of any of your own data.
6. **Postprocess** -> build a sample submission, run it through
   `utils/validate_submission.py` (the challenge's own validator) even
   on this tiny sample, to make sure the file format itself is right
   before you've invested time in the full run.
7. **Only now**, run the full pipeline end-to-end:
   ```bash
   python -m src.pipeline --data-dir dataset --output-dir output --model-path model.txt
   ```
8. Watch the printed console output in this order and confirm each
   number looks sane before trusting the next:
   `best_threshold` / `macro_F0.5` on validation -> the graph-boost
   comparison (with vs. without) -> SHAP feature importances (nothing
   should have near-zero importance across the board, which would
   suggest a broken feature) -> the surrogate-tree rules (should read
   as plausible business logic, e.g. splitting on `name_jaro_winkler`
   near the top).
9. Run `utils/validate_submission.py` against the real
   `output/matching_results.tsv` and `output/candidate_pairs.tsv`
   before uploading to the leaderboard.

---

## 3. One thing to know that isn't a bug

`emb_sim_name` / `emb_sim_addr` in `features.py` are always `NaN` in
the current code -- the embedding model runs inside `blocking.py` for
candidate retrieval, but its similarity scores aren't currently piped
back into the feature matrix (`emb_sim_lookup` is never populated by
`pipeline.py`). LightGBM handles `NaN` columns fine, so this doesn't
break anything, but it does mean you're not yet getting the full
semantic-similarity signal as an explicit feature -- only indirectly,
through which candidates got retrieved in the first place. Wiring this
up (computing and caching pairwise embedding similarity during
blocking, then passing it into `build_feature_matrix`) is a reasonable
next improvement if your validation F0.5 plateaus and SHAP shows the
existing features aren't enough.
