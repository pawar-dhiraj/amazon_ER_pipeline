"""
Step 2 -- Blocking / Candidate Generation (SCALED for 13M+ row corpora).

CHANGE LOG (perf pass): the previous version's `generate_candidates()`
did everything in one shot -- vectorize/encode the ENTIRE S2+S3
corpus, train + build a fresh FAISS index, query, then throw all of
it away -- every single time it was called. In pipeline.py that
function was called three times (fit split, val split, test split),
and fit_s1/val_s1 share the exact same train_s2/train_s3 corpus. That
meant the whole 10M+ row corpus was encoded and indexed from scratch
TWICE for identical data. That redundant encode+index-build is very
likely your actual time sink, not the querying itself.

FIX: every expensive, corpus-dependent step (HashingVectorizer
transform, sentence-embedding encode, FAISS IVFPQ train+add) is now
split into a "build once" half (`build_hashing_index`,
`build_embedding_index`, or the combined `build_candidate_indexes`)
and a cheap "query many times" half (`query_hashing_index`,
`query_embedding_index`, or the combined `generate_candidates_from_indexes`).
Build the bundle once per candidate corpus (train S2/S3, test S2/S3),
then query it once per S1 split.

The original single-call `generate_candidates()` / `faiss_embedding_block()`
/ `partitioned_hashing_block()` functions are kept as thin wrappers
(build+query in one call) purely for backward compatibility with the
existing test_*.py scripts -- prefer the split build/query functions
in new code (pipeline.py now uses them).

Other changes in this pass:
  - GPU support: encode functions accept a `device` argument and
    auto-detect CUDA if not given. GPU sizing (batch size) is picked
    automatically unless overridden.
  - Optional on-disk caching of the FAISS index (+ candidate ids) via
    `cache_dir`/`cache_key`, so a second run of the same corpus (e.g.
    after tweaking the matcher/threshold) doesn't re-encode anything.
  - `compute_pair_embedding_similarities()`: wires real emb_sim_name /
    emb_sim_addr values into features.py. It does NOT reuse the
    combined-text FAISS embeddings (those are one vector per entity,
    PQ-compressed, and not retrievable by id) -- instead it re-encodes
    name_norm/address_norm SEPARATELY, but only for the small set of
    entities that actually survived blocking (S1 rows + the union of
    their candidates), so it stays cheap regardless of corpus size.

Recall ceiling is still set HERE -- a true match that never becomes a
candidate can never be recovered downstream.
"""
from collections import defaultdict
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# MIT-licensed. -small by default; swap to -large only if you have a
# GPU or can spare the extra encode time (see EMBEDDING_MODEL_NAME
# override in build_candidate_indexes/build_embedding_index).
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

HASH_N_FEATURES = 2 ** 18  # fixed, constant memory regardless of corpus size
HASH_TRANSFORM_BATCH_SIZE = 200_000  # rows per HashingVectorizer.transform() call -- see _batched_hashing_transform
PREFIX_KEY_LEN = 2  # chars of name_norm used for partitioning
ADDRESS_KEY_LEN = 3  # chars of address_norm ALSO used -- see _partition_key for why
MAX_PARTITION_SIZE = 5000  # hard cap -- see _cap_oversized_partitions

# GPU vs CPU default encode batch sizes -- GPU VRAM can usually absorb
# a much bigger batch than a CPU core count would suggest. Override
# via encode_batch_size= on any of the functions below if needed.
GPU_ENCODE_BATCH_SIZE = 1024
CPU_ENCODE_BATCH_SIZE = 512


# --------------------------------------------------------------------------
# Device / encode helpers
# --------------------------------------------------------------------------

def _resolve_device(device=None):
    """Auto-detects CUDA if device isn't given explicitly. Falls back
    to CPU with no error if torch isn't installed or no GPU is found."""
    if device:
        return device
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _default_batch_size(device):
    return GPU_ENCODE_BATCH_SIZE if device == "cuda" else CPU_ENCODE_BATCH_SIZE


def _encode(model, texts, encode_batch_size, n_jobs, device="cpu"):
    """Encodes a list of texts. On GPU, just runs a normal (fast)
    single-process encode with a big batch size -- multiprocessing
    pools are a CPU-only trick and would only add overhead on GPU. On
    CPU, keeps the multi-process pool for n_jobs > 1 as before."""
    if device == "cuda":
        return model.encode(texts, batch_size=encode_batch_size,
                             normalize_embeddings=True, show_progress_bar=False,
                             device=device)
    if n_jobs and n_jobs > 1:
        pool = model.start_multi_process_pool()
        try:
            return model.encode_multi_process(
                texts, pool, batch_size=encode_batch_size, normalize_embeddings=True
            )
        finally:
            model.stop_multi_process_pool(pool)
    return model.encode(texts, batch_size=encode_batch_size,
                         normalize_embeddings=True, show_progress_bar=False, device=device)


def _combined_text(df):
    return (df["name_norm"].fillna("") + " " + df["address_norm"].fillna("")).values


def _batched_hashing_transform(vectorizer, texts, batch_size=HASH_TRANSFORM_BATCH_SIZE):
    """HashingVectorizer.transform() estimates a single nnz buffer for the
    ENTIRE input it's given in one call, and on a multi-million-row corpus
    with char_wb n-grams that estimate can blow past available RAM (seen:
    an 8GB single-allocation failure transforming an unbatched S2/S3
    corpus). Chunking keeps each call's internal buffer bounded regardless
    of total corpus size -- the result is identical to one big transform,
    just built incrementally."""
    texts = np.asarray(texts)
    if len(texts) <= batch_size:
        return vectorizer.transform(texts)
    chunks = [
        vectorizer.transform(texts[start:start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]
    return vstack(chunks, format="csr")


def _partition_key(df):
    """Blocking key: country + first-N chars of normalized name + a
    short slice of the ADDRESS too. Name-prefix alone skews badly --
    common English business words ("National", "General", "United",
    "American"...) cluster huge numbers of unrelated businesses under
    the same 2-3 character prefix, no matter how you tune the length.
    Adding an address-derived component gives a second, largely
    INDEPENDENT axis of separation: two "National ..." businesses
    almost always have different addresses, so combining the two
    fields spreads the skew far better than lengthening the name
    prefix alone would."""
    name_prefix = df["name_norm"].fillna("").str.slice(0, PREFIX_KEY_LEN)
    addr_prefix = df["address_norm"].fillna("").str.slice(0, ADDRESS_KEY_LEN)
    return (df["country"].fillna("").astype(str) + "|" + name_prefix + "|" + addr_prefix)


def _cap_oversized_partitions(groups_positions: dict, max_size: int, rng_seed: int = 0) -> dict:
    """Safety net on top of the better key above: even a good key can
    leave a few outlier partitions too large for brute-force NN to
    finish in reasonable time. Any partition still over max_size gets
    DETERMINISTICALLY down-sampled to max_size candidates."""
    rng = np.random.RandomState(rng_seed)
    capped = {}
    n_capped = 0
    for pkey, positions in groups_positions.items():
        if len(positions) > max_size:
            positions = rng.choice(positions, size=max_size, replace=False)
            n_capped += 1
        capped[pkey] = positions
    if n_capped:
        print(f"[blocking] capped {n_capped} oversized partition(s) down to {max_size} rows each")
    return capped


def _search_one_partition(pkey, s1_positions, cand_groups_positions, vec_s1_all, vec_cand_all,
                           cand_ids_all, s1_ids_all, top_k):
    """Lightweight per-partition step: slice the ALREADY-VECTORIZED
    sparse matrices down to this partition's rows, then NearestNeighbors
    on that slice. A closure (not a module-level function) is fine
    here because we use the `threading` backend -- threads share
    memory, so nothing needs to be pickled to reach a worker."""
    cand_positions = cand_groups_positions.get(pkey)
    if cand_positions is None or len(cand_positions) == 0:
        return {}

    vec_s1_part = vec_s1_all[s1_positions]
    vec_cand_part = vec_cand_all[cand_positions]

    k = min(top_k, len(cand_positions))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(vec_cand_part)
    _, indices = nn.kneighbors(vec_s1_part)

    cand_ids_part = cand_ids_all[cand_positions]
    s1_ids_part = s1_ids_all[s1_positions]
    local_result = defaultdict(set)
    for i, s1_id in enumerate(s1_ids_part):
        for idx in indices[i]:
            local_result[s1_id].add(cand_ids_part[idx])
    return local_result


# --------------------------------------------------------------------------
# Hashing blocker: build once, query many times
# --------------------------------------------------------------------------

def build_hashing_index(df_candidates, n_features=HASH_N_FEATURES,
                         transform_batch_size=HASH_TRANSFORM_BATCH_SIZE) -> dict:
    """Vectorizes df_candidates ONCE (stateless HashingVectorizer, no
    vocabulary dict) and partitions it by the blocking key ONCE.
    Returns a bundle to pass into query_hashing_index() as many times
    as you like (e.g. fit split, then val split) without repeating
    this work. Transforms in bounded batches (see
    _batched_hashing_transform) so this is safe to call directly on a
    full multi-million-row S2/S3 corpus."""
    if len(df_candidates) == 0:
        return None

    df_candidates = df_candidates.assign(_pkey=_partition_key(df_candidates)).reset_index(drop=True)
    vectorizer = HashingVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        n_features=n_features, alternate_sign=False, norm="l2",
    )
    vec_cand_all = _batched_hashing_transform(
        vectorizer, _combined_text(df_candidates), batch_size=transform_batch_size
    )
    cand_ids_all = df_candidates["entity_id"].values
    cand_groups_positions = df_candidates.groupby("_pkey").indices
    cand_groups_positions = _cap_oversized_partitions(cand_groups_positions, MAX_PARTITION_SIZE)

    return {
        "vectorizer": vectorizer,
        "vec_cand_all": vec_cand_all,
        "cand_ids_all": cand_ids_all,
        "cand_groups_positions": cand_groups_positions,
    }


def query_hashing_index(bundle, df_s1, top_k=15, n_jobs=-1, verbose_partition_stats=True,
                         transform_batch_size=HASH_TRANSFORM_BATCH_SIZE) -> dict:
    """Cheap half: vectorize just df_s1 (small), then run the
    lightweight slice+NN step per partition in parallel via
    threading. Safe to call repeatedly against the same bundle."""
    if bundle is None or len(df_s1) == 0:
        return defaultdict(set)

    df_s1 = df_s1.assign(_pkey=_partition_key(df_s1)).reset_index(drop=True)
    vec_s1_all = _batched_hashing_transform(
        bundle["vectorizer"], _combined_text(df_s1), batch_size=transform_batch_size
    )
    s1_ids_all = df_s1["entity_id"].values
    s1_groups_positions = df_s1.groupby("_pkey").indices

    if verbose_partition_stats:
        sizes = sorted((len(v) for v in bundle["cand_groups_positions"].values()), reverse=True)
        print(f"[blocking] {len(sizes)} candidate partitions -- "
              f"largest: {sizes[0] if sizes else 0}, median: {sizes[len(sizes)//2] if sizes else 0}, "
              f"top-5 sizes: {sizes[:5]}")

    partial_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_search_one_partition)(
            pkey, s1_positions, bundle["cand_groups_positions"],
            vec_s1_all, bundle["vec_cand_all"], bundle["cand_ids_all"], s1_ids_all, top_k,
        )
        for pkey, s1_positions in s1_groups_positions.items()
    )

    result = defaultdict(set)
    for partial in partial_results:
        for s1_id, cand_ids in partial.items():
            result[s1_id] |= cand_ids
    return result


def partitioned_hashing_block(df_s1, df_candidates, top_k=15, n_features=HASH_N_FEATURES,
                               n_jobs=-1, verbose_partition_stats=True) -> dict:
    """Backward-compatible single-call wrapper (build+query in one
    shot). Prefer build_hashing_index()/query_hashing_index() directly
    when the SAME candidate corpus will be queried more than once."""
    bundle = build_hashing_index(df_candidates, n_features=n_features)
    return query_hashing_index(bundle, df_s1, top_k=top_k, n_jobs=n_jobs,
                                verbose_partition_stats=verbose_partition_stats)


# --------------------------------------------------------------------------
# FAISS embedding blocker: build once, query many times
# --------------------------------------------------------------------------

def _dynamic_nlist(n_vectors: int) -> int:
    """IVF cell count: roughly 4*sqrt(N), clamped to a sane range so
    tiny validation-sample runs and the full 10M+ run both work."""
    return int(np.clip(4 * np.sqrt(max(n_vectors, 1)), 64, 8192))


def build_embedding_index(df_candidates, model=None, model_name=EMBEDDING_MODEL_NAME,
                           device=None, encode_batch_size=None, index_add_batch_size=8192,
                           train_sample_size=200_000, n_jobs=-1,
                           cache_dir=None, cache_key=None) -> dict:
    """Builds (or loads from disk cache) a FAISS IVFPQ index over
    df_candidates ONCE. Returns a bundle to pass into
    query_embedding_index() as many times as needed -- e.g. once for
    the fit split and once for the val split -- WITHOUT re-encoding or
    re-building the index for what is otherwise the identical
    train_s2/train_s3 corpus. This is the main fix for the redundant
    full-corpus re-embedding that was happening on every pipeline.py
    call.

    device: "cuda" / "cpu" / None (auto-detect). cache_dir + cache_key:
    if both given and a cached index exists on disk, loads it instead
    of re-encoding (useful across separate script runs, e.g. after
    only changing the matcher/threshold). Caller is responsible for
    clearing the cache dir if the underlying preprocessed data or
    model changes.
    """
    if not _EMBEDDINGS_AVAILABLE or not _FAISS_AVAILABLE or len(df_candidates) == 0:
        return None

    device = _resolve_device(device)
    resolved_n_jobs = (os.cpu_count() or 2) - 1 if n_jobs == -1 else max(1, n_jobs)
    if encode_batch_size is None:
        encode_batch_size = _default_batch_size(device)
    if _FAISS_AVAILABLE:
        faiss.omp_set_num_threads(resolved_n_jobs)

    if cache_dir and cache_key:
        idx_path = os.path.join(cache_dir, f"{cache_key}.faiss")
        ids_path = os.path.join(cache_dir, f"{cache_key}_ids.npy")
        if os.path.exists(idx_path) and os.path.exists(ids_path):
            print(f"[blocking] loading cached FAISS index '{cache_key}' from {cache_dir}")
            index = faiss.read_index(idx_path)
            ids = np.load(ids_path, allow_pickle=True)
            model = model or SentenceTransformer(model_name, device=device)
            return {"index": index, "ids": ids, "model": model, "model_name": model_name,
                    "device": device, "dim": getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()}

    print(f"[blocking] building FAISS index over {len(df_candidates)} rows "
          f"(device={device}, batch_size={encode_batch_size})")
    model = model or SentenceTransformer(model_name, device=device)
    texts_cand = list(_combined_text(df_candidates))
    cand_ids = df_candidates["entity_id"].values
    dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()

    nlist = _dynamic_nlist(len(texts_cand))
    m_pq = 32 if dim % 32 == 0 else 16  # sub-quantizer count must divide dim evenly
    nbits = 8

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits, faiss.METRIC_INNER_PRODUCT)

    # --- Train on a bounded sample, never the full corpus ---
    # FAISS recommends ~40 training points per IVF cell -- bump the
    # sample size up to cover the nlist we actually chose (avoids the
    # "please provide at least N training points" warning and the
    # weaker index quality that comes with under-training).
    min_train_points = 40 * nlist
    effective_train_size = min(len(texts_cand), max(train_sample_size, min_train_points))
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(texts_cand), size=effective_train_size, replace=False)
    train_texts = [texts_cand[i] for i in sample_idx]
    train_emb = _encode(model, train_texts, encode_batch_size, resolved_n_jobs, device)
    index.train(np.asarray(train_emb, dtype="float32"))

    # --- Add candidate embeddings in batches -- never hold all 10M+ in memory at once ---
    for start in range(0, len(texts_cand), index_add_batch_size):
        batch = texts_cand[start:start + index_add_batch_size]
        emb = _encode(model, batch, encode_batch_size, resolved_n_jobs, device)
        index.add(np.asarray(emb, dtype="float32"))

    if cache_dir and cache_key:
        os.makedirs(cache_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(cache_dir, f"{cache_key}.faiss"))
        np.save(os.path.join(cache_dir, f"{cache_key}_ids.npy"), cand_ids)
        print(f"[blocking] cached FAISS index '{cache_key}' to {cache_dir}")

    return {"index": index, "ids": cand_ids, "model": model, "model_name": model_name,
            "device": device, "dim": dim}


def query_embedding_index(bundle, df_s1, top_k=15, encode_batch_size=None,
                           index_add_batch_size=8192, n_jobs=-1, nprobe=16) -> dict:
    """Cheap half: only encodes df_s1 (small) and searches the
    already-built index. Safe to call repeatedly against the same
    bundle for different S1 splits."""
    if bundle is None or len(df_s1) == 0:
        return defaultdict(set)

    device = bundle["device"]
    model = bundle["model"]
    index = bundle["index"]
    cand_ids = bundle["ids"]
    resolved_n_jobs = (os.cpu_count() or 2) - 1 if n_jobs == -1 else max(1, n_jobs)
    if encode_batch_size is None:
        encode_batch_size = _default_batch_size(device)

    index.nprobe = nprobe  # higher = better recall, slower search; tune against validation recall
    texts_s1 = list(_combined_text(df_s1))
    s1_ids = df_s1["entity_id"].values
    k = min(top_k, len(cand_ids))

    result = defaultdict(set)
    for start in range(0, len(texts_s1), index_add_batch_size):
        batch_texts = texts_s1[start:start + index_add_batch_size]
        batch_ids = s1_ids[start:start + index_add_batch_size]
        emb = _encode(model, batch_texts, encode_batch_size, resolved_n_jobs, device)
        _, indices = index.search(np.asarray(emb, dtype="float32"), k)
        for i, s1_id in enumerate(batch_ids):
            for idx in indices[i]:
                if idx != -1:  # FAISS returns -1 for unfilled slots
                    result[s1_id].add(cand_ids[idx])
    return result


def faiss_embedding_block(df_s1, df_candidates, top_k=15, model_name=EMBEDDING_MODEL_NAME,
                           encode_batch_size=None, index_add_batch_size=8192,
                           train_sample_size=200_000, nprobe=16, n_jobs=-1, device=None) -> dict:
    """Backward-compatible single-call wrapper (build+query in one
    shot). Prefer build_embedding_index()/query_embedding_index()
    directly when the SAME candidate corpus will be queried more than
    once (e.g. fit + val splits sharing train_s2/train_s3)."""
    bundle = build_embedding_index(
        df_candidates, model_name=model_name, device=device,
        encode_batch_size=encode_batch_size, index_add_batch_size=index_add_batch_size,
        train_sample_size=train_sample_size, n_jobs=n_jobs,
    )
    return query_embedding_index(
        bundle, df_s1, top_k=top_k, encode_batch_size=encode_batch_size,
        index_add_batch_size=index_add_batch_size, n_jobs=n_jobs, nprobe=nprobe,
    )


# --------------------------------------------------------------------------
# Combined build-once / query-many orchestration
# --------------------------------------------------------------------------

def build_candidate_indexes(df_source2, df_source3, top_k=15, n_features=HASH_N_FEATURES,
                             use_embedding_blocking=True, model_name=EMBEDDING_MODEL_NAME,
                             device=None, encode_batch_size=None, train_sample_size=200_000,
                             n_jobs=-1, cache_dir=None, cache_prefix=None) -> dict:
    """Builds every reusable index (hashing + FAISS) ONCE per
    candidate source (S2, S3). Pass the returned bundle into
    generate_candidates_from_indexes() as many times as needed (fit
    split, val split, ...) to avoid re-vectorizing / re-encoding /
    re-building anything for what is otherwise the same corpus.
    cache_prefix (e.g. "train" or "test") namespaces the on-disk FAISS
    cache so train and test indexes never collide.
    """
    bundles = {}
    for label, df_cand in (("s2", df_source2), ("s3", df_source3)):
        hashing_bundle = build_hashing_index(df_cand, n_features=n_features)
        emb_bundle = None
        if use_embedding_blocking and len(df_cand) > 0:
            cache_key = f"{cache_prefix}_{label}" if (cache_dir and cache_prefix) else None
            emb_bundle = build_embedding_index(
                df_cand, model_name=model_name, device=device,
                encode_batch_size=encode_batch_size, train_sample_size=train_sample_size,
                n_jobs=n_jobs, cache_dir=cache_dir, cache_key=cache_key,
            )
        bundles[label] = {"hashing": hashing_bundle, "embedding": emb_bundle}
    return bundles


def generate_candidates_from_indexes(df_s1, bundles: dict, top_k=15, n_jobs=-1) -> dict:
    """Cheap half of candidate generation: queries the already-built
    hashing + FAISS bundles for this df_s1 split. Call once per split
    against bundles built once via build_candidate_indexes()."""
    candidates = defaultdict(set)

    for label in ("s2", "s3"):
        b = bundles.get(label)
        if b is None:
            continue
        hashing_result = query_hashing_index(b["hashing"], df_s1, top_k=top_k, n_jobs=n_jobs)
        for s1_id in df_s1["entity_id"]:
            candidates[s1_id] |= hashing_result.get(s1_id, set())

        if b["embedding"] is not None:
            emb_result = query_embedding_index(b["embedding"], df_s1, top_k=top_k, n_jobs=n_jobs)
            for s1_id in df_s1["entity_id"]:
                candidates[s1_id] |= emb_result.get(s1_id, set())

    for s1_id in df_s1["entity_id"]:
        candidates.setdefault(s1_id, set())

    return candidates


def generate_candidates(df_s1, df_source2, df_source3, top_k=15,
                         use_embedding_blocking=True, n_jobs=-1, device=None) -> dict:
    """Backward-compatible single-call wrapper (build+query in one
    shot) -- this is what the existing test_blocking.py,
    test_blocking_recall.py and test_utils.py call, so their behavior
    is unchanged. In pipeline.py, prefer build_candidate_indexes() +
    generate_candidates_from_indexes() when the SAME S2/S3 corpus will
    be queried more than once."""
    bundles = build_candidate_indexes(
        df_source2, df_source3, top_k=top_k,
        use_embedding_blocking=use_embedding_blocking, n_jobs=n_jobs, device=device,
    )
    return generate_candidates_from_indexes(df_s1, bundles, top_k=top_k, n_jobs=n_jobs)


# --------------------------------------------------------------------------
# Embedding-similarity features (wires emb_sim_name / emb_sim_addr into
# features.py -- previously always NaN, see PIPELINE_HOWTO.md's note)
# --------------------------------------------------------------------------

def compute_pair_embedding_similarities(candidate_pairs: dict, df_s1: pd.DataFrame,
                                         df_others: pd.DataFrame, model_name=EMBEDDING_MODEL_NAME,
                                         device=None, encode_batch_size=None, n_jobs=-1) -> dict:
    """Populates real emb_sim_name / emb_sim_addr values for
    features.build_feature_matrix()'s emb_sim_lookup. Deliberately
    does NOT reuse the combined-text FAISS embeddings from
    build_embedding_index() above -- those are one PQ-compressed
    vector per entity (name+address concatenated) built for the FULL
    candidate corpus and aren't retrievable by id after compression.
    Instead this re-encodes name_norm and address_norm SEPARATELY, but
    only for the entities that actually survived blocking (df_s1's
    rows + the small union of ids in candidate_pairs) -- typically a
    few thousand texts even against a 10M+ row corpus, so this stays
    cheap regardless of dataset size.

    Returns {(s1_id, cand_id): (sim_name, sim_addr)}.
    """
    if not _EMBEDDINGS_AVAILABLE or len(df_s1) == 0:
        return {}

    needed_cand_ids = set()
    for cand_ids in candidate_pairs.values():
        needed_cand_ids |= cand_ids
    if not needed_cand_ids:
        return {}

    device = _resolve_device(device)
    if encode_batch_size is None:
        encode_batch_size = _default_batch_size(device)
    resolved_n_jobs = (os.cpu_count() or 2) - 1 if n_jobs == -1 else max(1, n_jobs)

    model = SentenceTransformer(model_name, device=device)

    df_cand_subset = df_others[df_others["entity_id"].isin(needed_cand_ids)]
    if len(df_cand_subset) == 0:
        return {}

    def _embed_field(df, col):
        texts = df[col].fillna("").tolist()
        emb = _encode(model, texts, encode_batch_size, resolved_n_jobs, device)
        return dict(zip(df["entity_id"].values, np.asarray(emb, dtype="float32")))

    print(f"[blocking] computing emb_sim features for {len(df_s1)} S1 rows x "
          f"{len(df_cand_subset)} surviving candidates")
    s1_name_emb = _embed_field(df_s1, "name_norm")
    s1_addr_emb = _embed_field(df_s1, "address_norm")
    cand_name_emb = _embed_field(df_cand_subset, "name_norm")
    cand_addr_emb = _embed_field(df_cand_subset, "address_norm")

    result = {}
    for s1_id, cand_ids in candidate_pairs.items():
        if s1_id not in s1_name_emb:
            continue
        v_name1, v_addr1 = s1_name_emb[s1_id], s1_addr_emb[s1_id]
        for cand_id in cand_ids:
            if cand_id not in cand_name_emb:
                continue
            # Embeddings are L2-normalized (normalize_embeddings=True in
            # _encode), so a plain dot product IS cosine similarity.
            sim_name = float(np.dot(v_name1, cand_name_emb[cand_id]))
            sim_addr = float(np.dot(v_addr1, cand_addr_emb[cand_id]))
            result[(s1_id, cand_id)] = (sim_name, sim_addr)
    return result