"""
Step 2 -- Blocking / Candidate Generation (SCALED for 13M+ row corpora).

The original TfidfVectorizer + brute-force NearestNeighbors approach
does not scale to this dataset size, for two SEPARATE reasons, both
fixed below:

  1. TfidfVectorizer.fit() builds and holds an explicit
     {ngram_string: column_index} vocabulary dict in memory. Across
     ~12.5M business name + address strings, char n-grams (2-4) can
     produce tens of millions of distinct entries -- this dict alone
     is what raised the MemoryError, before the actual sparse matrix
     was even built.
     FIX: HashingVectorizer. It is STATELESS -- no fit(), no
     vocabulary dict, just a fixed-size hash of each n-gram into one
     of `n_features` buckets. Memory use is now CONSTANT regardless of
     corpus size (a few hundred MB for the hash space, not a dict that
     grows with the data).

  2. A brute-force NearestNeighbors search comparing every S1 row
     against every candidate row is an O(n_s1 x n_candidates)
     operation. At ~2.2M x ~10M this is computationally infeasible
     regardless of how much memory you have.
     FIX (two complementary techniques, unioned):
       a) Partition both frames by a cheap blocking key (country +
          first-2-chars of normalized name) BEFORE running any
          vectorizer/NN step, so each NN call only ever compares
          within a partition of a few thousand rows, not millions.
       b) FAISS IndexIVFPQ for the embedding signal: a
          product-quantized approximate index that compresses each
          embedding vector to a few dozen bytes, so even 10M+ vectors
          fit comfortably in a few hundred MB, with sub-linear search
          time. This is a GLOBAL (unpartitioned) recall net that
          catches true matches a prefix-based partition would miss
          (e.g. a typo in the first two characters of the name).

Recall ceiling is still set HERE -- a true match that never becomes a
candidate can never be recovered downstream.
"""
from collections import defaultdict
import os

import numpy as np
from joblib import Parallel, delayed
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

# MIT-licensed. NOTE: reverted to -small for this dataset size.
# -large (~560M params) was the right call for a hackathon-scale
# corpus; at 10M+ candidate rows, encoding time (not memory) becomes
# the bottleneck, and -small is roughly 3-4x faster to encode with
# only a modest quality drop. Swap back to -large only if you have
# a GPU or can spare the extra encode time.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

HASH_N_FEATURES = 2 ** 18  # fixed, constant memory regardless of corpus size
PREFIX_KEY_LEN = 2  # chars of name_norm used for partitioning


def _combined_text(df):
    return (df["name_norm"].fillna("") + " " + df["address_norm"].fillna("")).values


def _partition_key(df):
    """Cheap blocking key: country + first-N chars of normalized name.
    Deliberately short (2 chars) to keep recall loss from partitioning
    itself low -- the FAISS embedding pass below is the safety net for
    whatever this key structure misses (e.g. a typo in the first
    couple of characters, or a completely different word order)."""
    prefix = df["name_norm"].fillna("").str.slice(0, PREFIX_KEY_LEN)
    return df["country"].fillna("").astype(str) + "|" + prefix


def _process_partition(s1_group, cand_group, top_k, n_features):
    """Module-level (picklable) worker -- runs the hashing+kNN search
    for ONE partition. Kept as a plain function, not a method or
    closure, because joblib's process-based backend needs to pickle
    it to ship to worker processes; a nested/local function can't be
    pickled and would silently fall back to (slower) threads or
    error out."""
    if cand_group is None or len(cand_group) == 0:
        return {}

    vectorizer = HashingVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        n_features=n_features, alternate_sign=False, norm="l2",
    )
    vec_s1 = vectorizer.transform(_combined_text(s1_group))
    vec_cand = vectorizer.transform(_combined_text(cand_group))

    k = min(top_k, len(cand_group))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(vec_cand)
    _, indices = nn.kneighbors(vec_s1)

    cand_ids = cand_group["entity_id"].values
    s1_ids = s1_group["entity_id"].values
    local_result = defaultdict(set)
    for i, s1_id in enumerate(s1_ids):
        for idx in indices[i]:
            local_result[s1_id].add(cand_ids[idx])
    return local_result


def partitioned_hashing_block(df_s1, df_candidates, top_k=15,
                               n_features=HASH_N_FEATURES, n_jobs=-1) -> dict:
    """Memory-safe blocking for very large corpora. Partitions both
    frames by _partition_key and runs HashingVectorizer + kNN WITHIN
    each partition -- so no single NN call ever sees more than a
    partition's worth of rows, and no vocabulary is ever built.

    n_jobs=-1 (default) parallelizes across ALL CPU cores using
    joblib's process-based backend: with a corpus this size you'll
    have many thousands of independent partitions, and each one's
    fit+kneighbors call is CPU-bound and embarrassingly parallel --
    this is the single biggest speed lever available here, since
    partitions don't share any state. Set n_jobs=1 to disable (useful
    for debugging -- easier to read a traceback from a single process).
    """
    if len(df_candidates) == 0:
        return defaultdict(set)

    df_s1 = df_s1.assign(_pkey=_partition_key(df_s1))
    df_candidates = df_candidates.assign(_pkey=_partition_key(df_candidates))

    cand_groups = dict(tuple(df_candidates.groupby("_pkey")))
    s1_groups = list(df_s1.groupby("_pkey"))

    partial_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_partition)(s1_group, cand_groups.get(pkey), top_k, n_features)
        for pkey, s1_group in s1_groups
    )

    result = defaultdict(set)
    for partial in partial_results:
        for s1_id, cand_ids in partial.items():
            result[s1_id] |= cand_ids

    return result


def _dynamic_nlist(n_vectors: int) -> int:
    """IVF cell count: roughly 4*sqrt(N), clamped to a sane range so
    tiny validation-sample runs and the full 10M+ run both work."""
    return int(np.clip(4 * np.sqrt(max(n_vectors, 1)), 64, 8192))


def _encode(model, texts, encode_batch_size, n_jobs):
    """Encodes a list of texts, using sentence-transformers' official
    multi-process pool when n_jobs > 1 (spreads encoding across CPU
    cores via multiple worker processes, each running its own copy of
    the model) -- this is the single biggest lever for the 10M+ text
    encode time, since it's pure CPU-bound work with no shared state
    between texts. Falls back to plain single-process encode()
    otherwise (n_jobs == 1, or a target_devices list isn't sensible on
    this machine)."""
    if n_jobs and n_jobs > 1:
        pool = model.start_multi_process_pool()
        try:
            return model.encode_multi_process(
                texts, pool, batch_size=encode_batch_size, normalize_embeddings=True
            )
        finally:
            model.stop_multi_process_pool(pool)
    return model.encode(texts, batch_size=encode_batch_size,
                         normalize_embeddings=True, show_progress_bar=False)


def faiss_embedding_block(df_s1, df_candidates, top_k=15,
                           model_name=EMBEDDING_MODEL_NAME,
                           encode_batch_size=512, index_add_batch_size=8192,
                           train_sample_size=200_000, nprobe=16, n_jobs=-1) -> dict:
    """Global (unpartitioned) approximate-nearest-neighbor blocking
    using a product-quantized FAISS index -- this is what makes 10M+
    embeddings fit in memory at all (compressed to a few dozen bytes
    per vector instead of ~1.5-6 KB for a raw float32 vector),
    catching true matches the partitioned hashing blocker above would
    miss due to its prefix key. Returns empty result (caller just
    skips the union) if sentence-transformers or faiss aren't
    installed -- see requirements.txt; at this corpus size, installing
    both is strongly recommended rather than optional.

    n_jobs controls CPU parallelism for the ENCODING step (the actual
    bottleneck at this scale) via sentence-transformers' multi-process
    pool, and also sets FAISS's own OpenMP thread count for the
    index build/search step. n_jobs=-1 (default) uses all cores minus
    one; pass n_jobs=1 to disable and debug single-process."""
    if not _EMBEDDINGS_AVAILABLE or not _FAISS_AVAILABLE or len(df_candidates) == 0:
        return defaultdict(set)

    resolved_n_jobs = (os.cpu_count() or 2) - 1 if n_jobs == -1 else max(1, n_jobs)
    faiss.omp_set_num_threads(resolved_n_jobs)  # FAISS's own thread count for add()/search()

    model = SentenceTransformer(model_name)
    texts_cand = list(_combined_text(df_candidates))
    cand_ids = df_candidates["entity_id"].values
    dim = model.get_sentence_embedding_dimension()

    nlist = _dynamic_nlist(len(texts_cand))
    m_pq = 32 if dim % 32 == 0 else 16  # sub-quantizer count must divide dim evenly
    nbits = 8

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits, faiss.METRIC_INNER_PRODUCT)

    # --- Train on a bounded sample, never the full corpus ---
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(texts_cand), size=min(train_sample_size, len(texts_cand)),
                             replace=False)
    train_texts = [texts_cand[i] for i in sample_idx]
    train_emb = _encode(model, train_texts, encode_batch_size, resolved_n_jobs)
    index.train(np.asarray(train_emb, dtype="float32"))

    # --- Add candidate embeddings in batches -- never hold all 10M+ in memory at once ---
    for start in range(0, len(texts_cand), index_add_batch_size):
        batch = texts_cand[start:start + index_add_batch_size]
        emb = _encode(model, batch, encode_batch_size, resolved_n_jobs)
        index.add(np.asarray(emb, dtype="float32"))

    index.nprobe = nprobe  # higher = better recall, slower search; tune against your validation recall

    # --- Query S1 in batches too ---
    texts_s1 = list(_combined_text(df_s1))
    s1_ids = df_s1["entity_id"].values
    k = min(top_k, len(texts_cand))

    result = defaultdict(set)
    for start in range(0, len(texts_s1), index_add_batch_size):
        batch_texts = texts_s1[start:start + index_add_batch_size]
        batch_ids = s1_ids[start:start + index_add_batch_size]
        emb = _encode(model, batch_texts, encode_batch_size, resolved_n_jobs)
        _, indices = index.search(np.asarray(emb, dtype="float32"), k)
        for i, s1_id in enumerate(batch_ids):
            for idx in indices[i]:
                if idx != -1:  # FAISS returns -1 for unfilled slots
                    result[s1_id].add(cand_ids[idx])
    return result


def generate_candidates(df_s1, df_source2, df_source3, top_k=15,
                         use_embedding_blocking=True, n_jobs=-1) -> dict:
    """Runs the partitioned hashing blocker (always) and the FAISS
    embedding blocker (if use_embedding_blocking and deps available)
    against S2 and S3 separately, unions all candidate sets per S1
    entity. Returns dict: s1_entity_id -> set(candidate_ids).

    use_embedding_blocking defaults to True but is exposed as a flag:
    at 10M+ rows, encoding every candidate is the slow part (compute
    time, not memory, once you're on HashingVectorizer + FAISS-PQ) --
    set it False for a fast first pass while you're still debugging
    the rest of the pipeline, then turn it back on for your real run.

    n_jobs=-1 (default) parallelizes both blockers across CPU cores
    (joblib processes for hashing partitions, sentence-transformers'
    multi-process pool + FAISS OpenMP threads for embeddings). Set
    n_jobs=1 to run single-process, which is slower but much easier to
    debug if something goes wrong (clean, single tracebacks instead of
    ones from worker processes).
    """
    candidates = defaultdict(set)

    for df_cand in (df_source2, df_source3):
        if len(df_cand) == 0:
            continue
        hashing_result = partitioned_hashing_block(df_s1, df_cand, top_k=top_k, n_jobs=n_jobs)
        for s1_id in df_s1["entity_id"]:
            candidates[s1_id] |= hashing_result.get(s1_id, set())

        if use_embedding_blocking:
            emb_result = faiss_embedding_block(df_s1, df_cand, top_k=top_k, n_jobs=n_jobs)
            for s1_id in df_s1["entity_id"]:
                candidates[s1_id] |= emb_result.get(s1_id, set())

    for s1_id in df_s1["entity_id"]:
        candidates.setdefault(s1_id, set())

    return candidates
