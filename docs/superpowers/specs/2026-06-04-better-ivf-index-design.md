# Better IVF Index — Design Spec

**Date:** 2026-06-04  
**Goal:** Replace MiniBatchKMeans with standard KMeans + more clusters + more n_init to produce a higher-quality index, enabling lower nprobe with same or better recall. Net result: lower p99 + fewer FP/FN.

---

## Baseline (current)

| Parameter | Value |
|-----------|-------|
| Algorithm | MiniBatchKMeans |
| N_CLUSTERS | 4000 |
| n_init | 3 |
| batch_size | 100_000 |
| nprobe (Go) | 20 |
| Avg vecs/probe | ~750 |
| Vecs scanned/query | ~15,000 |
| Score (nprobe=20) | 5581 (FP=18, FN=2) |
| Score (nprobe=24) | 5603 (FP=14, FN=2) |

---

## Why MiniBatchKMeans is limiting

MiniBatchKMeans processes random subsets (batch_size=100k) per iteration. It converges fast but centroids are statistically biased — each centroid only sees a fraction of its true cluster at each step. The resulting cluster boundaries are "good enough" but not globally optimal.

Standard KMeans runs Lloyd's algorithm: every iteration computes exact assignments and exact centroid means over all N vectors. With 16 CPU cores and 19GB RAM, 3M×16 floats (~190MB) fits entirely in memory. The cost is time — acceptable for an offline build step.

---

## Changes

### 1. `ml/build_index.py` — algorithm

Replace `MiniBatchKMeans` with `KMeans`:

```python
# Before
from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3, batch_size=100_000)

# After
from sklearn.cluster import KMeans
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=N_INIT, max_iter=300, verbose=1)
```

- `n_jobs=-1` is implicit (KMeans uses all BLAS threads by default)
- `init='k-means++'` is the default — keep it (better than random init)
- `max_iter=300` — default, sufficient for convergence at this scale
- `verbose=1` — show iteration progress (useful for long runs)

### 2. `ml/build_index.py` — constants

```python
N_CLUSTERS = 8000   # was 4000
N_INIT     = 10     # was 3 (MiniBatchKMeans n_init)
```

**Why 8000?**  
8000 clusters → avg 375 vecs/cluster.  
At nprobe=15: scans 15 × 375 = **5,625 vecs/query** (vs 15,000 currently — 2.7× fewer).  
For nprobe to achieve equivalent recall, the relationship is roughly: `nprobe ∝ C × target_fraction_scanned`. Better clustering (KMeans) compresses true neighbors into fewer, tighter clusters, so the required nprobe scales sublinearly with C.

**Why n_init=10?**  
Each init is an independent KMeans run from a fresh k-means++ seed; the best inertia wins. Going from 3→10 dramatically reduces the probability of landing in a local optimum. n_init=20 is theoretically better but triples time — 10 is the standard production value for datasets of this size.

**Estimated build time (16 cores, 19GB RAM):**  
~20-40 min per KMeans run → n_init=10 → **3-7 hours total**.  
Optional fast path: cuML GPU KMeans on RTX 3060 Ti reduces this to ~15-30 min total.

### 3. `internal/search/knn.go` — nprobe

After the new index is built and benchmarked, update:

```go
// knn.go
const nprobe = 15  // target; adjust after benchmark (range: 12-20)
```

Start at nprobe=15 for the first benchmark run. The goal is to find the lowest nprobe where FP+FN stabilizes (diminishing returns on detection gain vs latency cost).

---

## Optional: GPU fast path (cuML)

If the build time is prohibitive, install cuML:

```bash
# Requires CUDA 12 + mamba
mamba install -c rapidsai -c conda-forge cuml

# Then in build_index.py:
from cuml.cluster import KMeans
# Same API as sklearn — no other changes needed
```

RTX 3060 Ti (8GB VRAM) can hold 3M×16 float32 = ~190MB easily. Expected time: ~2-5 min per run → n_init=10 → **20-50 min total**.

This is optional — sklearn on 16 cores is sufficient. cuML just reduces iteration time.

---

## Benchmark methodology

After building the new index, run local bench for nprobe ∈ {10, 12, 15, 20} and record each result in `references/bench/result.csv`:

```
nprobe | FP | FN | weighted_E | p99 (ms) | final_score
```

Acceptance criteria for the new index:
- At nprobe=15: final_score ≥ 5550 (≥ current nprobe=20 score of 5581 is ideal)
- p99 at nprobe=15 < p99 at nprobe=20 (latency win)
- If nprobe=12 achieves score ≥ 5550: use nprobe=12

If the new index at nprobe=20 does NOT beat baseline score=5581, the clustering quality improvement was insufficient — retry with n_init=20 or investigate cuML.

---

## What does NOT change

- Index binary format: `IVF2` magic, same layout — only centroid count changes (8000 instead of 4000)
- `index.go` LoadIVFIndex: no changes needed (reads C dynamically from file header)
- `knn.go` KNN logic: no changes, only `nprobe` constant updated
- Vector dimensions: still padded to 16
- int16 quantization: still ×10000 scale
- k=5, threshold=0.6: fixed by spec, untouched

---

## Files changed

| File | Change |
|------|--------|
| `ml/build_index.py` | KMeans, N_CLUSTERS=8000, N_INIT=10 |
| `internal/search/knn.go` | nprobe constant (after bench) |

---

## Risks

**Risk 1: KMeans diverges or is slow on 3M vectors**  
Mitigation: `verbose=1` shows per-iteration inertia. If inertia plateaus before max_iter, it converged. If time is unacceptable, fall back to n_init=5 or use cuML.

**Risk 2: More clusters = lower recall at nprobe=15 (worse than baseline)**  
Mitigation: benchmark nprobe=20 first on the new index. If worse than baseline at same nprobe, clustering quality didn't offset the cluster boundary split problem. In that case, keep N_CLUSTERS=6000 (compromise).

**Risk 3: Index size**  
Centroids: 8000 × 16 × 4 = 512KB (negligible). Vectors: unchanged (N × 16 × 2). No memory budget impact.
