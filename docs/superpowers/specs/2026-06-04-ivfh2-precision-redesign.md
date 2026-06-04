# IVF_H2 Precision Redesign — Design Spec

**Date:** 2026-06-04  
**Status:** Approved  
**Goal:** Maximize detection score by improving IVF index precision and reducing nprobe scan cost.

---

## Problem

Current single-level IVF (C=8000, nprobe=25) has:
- Phase 1: scans all 8000 centroids every query — 128K float ops
- Persistent FP=10, FN=1 at nprobe=25 (score 5656)
- Remote p99=30ms due to CPU throttling under concurrent load
- No routing by transaction type — null-tx and active-tx vectors pollute each other's clusters

---

## Solution Overview

Four independent improvements combined:

1. **Dual index split** — hard-route by `last_transaction == nil`
2. **2-level hierarchical IVF** — reduce Phase 1 from 8000 to ~300 centroid comparisons
3. **Boundary oversampling + balanced k-means** — eliminate structural FP/FN
4. **Adaptive nprobe + centroid pruning** — fast exit for clear cases, exact pruning for uncertain

---

## Architecture

```
Request → service routing
  req.LastTransaction == nil → firstTxIdx.KNN(query, k)
  otherwise                  → subsequentTxIdx.KNN(query, k)

Files:
  index/first_tx.ivfh       K1=64,  K2=32 → 2048 leaf clusters
  index/subsequent_tx.ivfh  K1=128, K2=32 → 4096 leaf clusters

Per-query ops (subsequent_tx):
  Phase 1: 128 macro-centroid scans   (vs 8000 today — 62× less)
  Phase 2: 4×32 = 128 micro-centroid scans
  Phase 3 fast path: 8 clusters × ~488 vecs = ~3904 vecs  (vs 9375 today — 2.4× less)
  Phase 3 clear case: early exit after 1-3 clusters = ~200-600 vecs scanned
```

---

## Binary Format: IVF_H2

Filename extension: `.ivfh`. Little-endian, mmap-safe.

```
Offset  Size          Field
──────────────────────────────────────────────────────
0       4B            magic "IVFH"
4       4B float32    D_safe  (confidence radius for fraudCount==0 fast exit)
8       4B uint32     K1      (number of macro clusters)
12      4B uint32     K2      (micro clusters per macro cluster)
16      4B uint32     N       (total vectors)
20      K1×16×4B      macro_centroids   (float32, row-major)
+       K1×K2×16×4B   micro_centroids   (float32, indexed by macro_id*K2 + micro_id)
+       K1×K2×4B      cluster_starts    (uint32, offset into flat_vectors)
+       K1×K2×4B      cluster_sizes     (uint32)
+       K1×K2×4B      cluster_radius    (float32, max L2 dist from centroid to any vector in cluster)
+       N×16×2B       flat_vectors      (int16 ×10000, sorted by cluster assignment)
+       N×1B          flat_labels       (uint8: 0=legit, 1=fraud)
```

**D_safe computation:** 99th percentile of `maxDist` (distance to 5th nearest neighbor) across
a 10k sample of vectors whose true brute-force k=5 gives fraudCount==0.

**cluster_radius computation:** `max(dist(v, centroid))` for all vectors v in that cluster.
Enables exact centroid pruning in Go without per-vector overhead.

---

## build_index.py Redesign

### Step 1: Data Split

```python
null_mask = (vectors[:, 5] == -1.0) & (vectors[:, 6] == -1.0)
vectors_first, labels_first     = vectors[null_mask],  labels[null_mask]
vectors_subseq, labels_subseq   = vectors[~null_mask], labels[~null_mask]
```

Validates that null-tx / active-tx populations are mutually exclusive (assert no overlap).
Prints split sizes and fraction.

### Step 2: Boundary Oversampling (per split)

```python
# GPU brute-force KNN on a 50k sample
nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
nbrs.fit(vectors_gpu)
_, indices = nbrs.kneighbors(sample_gpu)
fraud_counts = labels[indices].sum(axis=1)

boundary_mask = (fraud_counts == 2) | (fraud_counts == 3)
boundary_vecs = sample[boundary_mask]
boundary_labs = sample_labels[boundary_mask]

# Duplicate boundary vectors 3× before clustering
vectors_aug = np.concatenate([vectors, boundary_vecs, boundary_vecs, boundary_vecs])
labels_aug  = np.concatenate([labels,  boundary_labs, boundary_labs, boundary_labs])
```

Forces KMeans to allocate denser clusters near the decision boundary (fraudScore=0.6 threshold),
reducing structural FP/FN from imprecise centroid boundaries.

**Important:** augmented vectors (`vectors_aug`) are used ONLY for KMeans fitting. The final index
stores only the original N vectors. Boundary duplicates are discarded after clustering — they exist
solely to bias centroid placement.

### Step 3: 2-Level KMeans

```python
# Level 1: macro clusters (fit on augmented vectors for better centroid placement)
km1 = KMeans(n_clusters=K1, init='scalable-k-means++', n_init=10, max_iter=300)
km1.fit(vectors_gpu_aug)
macro_centroids = km1.cluster_centers_  # (K1, 16)

# Assign original (non-augmented) vectors to macro clusters
macro_assignments_orig = km1.predict(vectors_gpu)  # (N,) — original vectors only

# Level 2: micro clusters within each macro (also fit on augmented, assign original)
micro_centroids   = np.zeros((K1, K2, 16), dtype=np.float32)
micro_assignments = np.zeros(N, dtype=np.int32)  # N = original vector count

for i in range(K1):
    # Use augmented subset for fitting
    mask_aug  = km1.predict(vectors_gpu_aug) == i
    vecs_aug_i = vectors_gpu_aug[mask_aug]
    km2 = KMeans(n_clusters=K2, init='scalable-k-means++', n_init=5, max_iter=200)
    km2.fit(vecs_aug_i)
    micro_centroids[i] = km2.cluster_centers_
    # Assign original vectors in this macro cluster
    mask_orig = macro_assignments_orig == i
    micro_assignments[mask_orig] = i * K2 + km2.predict(vectors_gpu[mask_orig])
```

### Step 4: Balanced K-Means Post-Processing

```python
N_leaves = K1 * K2
cluster_sizes = np.bincount(micro_assignments, minlength=N_leaves).astype(np.int32)
max_size = int(N / N_leaves * 1.5)  # 1.5× average leaf size

for macro_id in range(K1):
    for micro_local in range(K2):
        c = macro_id * K2 + micro_local
        if cluster_sizes[c] <= max_size:
            continue
        # Vectors in overflow cluster, sorted by dist to their centroid (descending)
        idxs = np.where(micro_assignments == c)[0]
        cent = micro_centroids[macro_id, micro_local]
        dists = np.linalg.norm(vectors[idxs] - cent, axis=1)
        overflow_idxs = idxs[np.argsort(-dists)[:cluster_sizes[c] - max_size]]
        # Reassign overflow vectors to nearest micro sibling with capacity
        for v in overflow_idxs:
            siblings = [macro_id * K2 + j for j in range(K2) if j != micro_local
                        and cluster_sizes[macro_id * K2 + j] < max_size]
            if not siblings:
                continue  # all siblings full — leave as is
            sib_cents = micro_centroids[macro_id][
                [s % K2 for s in siblings]]
            nearest_sib = siblings[np.argmin(
                np.linalg.norm(sib_cents - vectors[v], axis=1))]
            micro_assignments[v] = nearest_sib
            cluster_sizes[c] -= 1
            cluster_sizes[nearest_sib] += 1
```

Ensures no single cluster scan dominates Phase 3 latency.

### Step 5: D_safe Computation

D_safe is the 99th percentile of the distance to the 5th nearest neighbor across a sample of
vectors whose true brute-force k=5 gives fraudCount==0. Stored as L2 distance (not squared).
In knn.go, D_safe is pre-squared at load time (`DSafeSq = DSafe * DSafe`) for cheap comparison.

```python
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

# Sample up to 10k legit vectors (label==0)
legit_idxs = np.where(labels == 0)[0]
sample_idx = legit_idxs[:min(10_000, len(legit_idxs))]
sample_vecs = cp.asarray(vectors[sample_idx], dtype=cp.float32)

# Brute-force k=5 on FULL dataset
nbrs = cuNearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean',
                          output_type='numpy')
nbrs.fit(cp.asarray(vectors, dtype=cp.float32))
dists_sq, neighbor_idx = nbrs.kneighbors(sample_vecs)
# dists_sq: (sample_size, 5) — squared euclidean distances

# Keep only samples where true brute-force gives fraudCount==0
neighbor_labels = labels[neighbor_idx]           # (sample_size, 5)
fraud_counts    = neighbor_labels.sum(axis=1)    # (sample_size,)
truly_legit     = fraud_counts == 0

# D_safe = 99th percentile of distance-to-5th-neighbor for truly legit samples
# Note: cuML kneighbors returns squared distances; take sqrt
max_dists = np.sqrt(dists_sq[truly_legit, 4])   # L2 distance to 5th neighbor
D_safe = float(np.percentile(max_dists, 99))
print(f"D_safe = {D_safe:.6f} (99th pct of max neighbor dist for fraudCount==0)")
```

### Step 6: Cluster Radius Computation

```python
for c in range(K1 * K2):
    vecs_in_c = flat_vectors[cluster_starts[c]:cluster_starts[c]+cluster_sizes[c]]
    centroid_c = micro_centroids_flat[c]
    dists_to_centroid = np.linalg.norm(vecs_in_c - centroid_c, axis=1)
    cluster_radius[c] = dists_to_centroid.max()
```

### Step 7: Write IVF_H2

Write all fields in the exact order specified in the binary format above.

---

## index.go: IVFHIndex

New struct (replaces `IVFIndex` for the hierarchical format):

```go
type IVFHIndex struct {
    K1, K2, N      int
    DSafe           float32  // L2 distance threshold (stored as dist, not dist²)
    DSafeSq         float32  // DSafe*DSafe — pre-computed at load time for cheap comparison
    NCoarseProbe    int      // top macro clusters to probe (4 for subseq, 3 for first)
    MacroCentroids  []float32  // K1×16 float32
    MicroCentroids  []float32  // K1×K2×16 float32, indexed by [macro*K2+micro]*16
    Starts          []uint32   // K1×K2
    Sizes           []uint32   // K1×K2
    Radii           []float32  // K1×K2 — max L2 dist from centroid to any vector in cluster
    Vectors         []int16    // N×16 (mmap zero-copy)
    Labels          []uint8    // N    (mmap zero-copy)
    mmap            []byte
}
```

`LoadIVFHIndex(path string, nCoarseProbe int) (*IVFHIndex, error)` — mmap file, parse header,
populate all slices. `Vectors` and `Labels` are zero-copy views into the mmap region.
`DSafeSq` is computed as `DSafe * DSafe` at load time.

Old `IVFIndex` and `IVF2` format remain unchanged. Both `IVFIndex` and `IVFHIndex` implement
`search.Index` interface (`KNN(query [16]float32, k int) int`).

---

## knn.go: Search Engine

Constants:
```go
const (
    nCoarseProbeSubseq = 4   // macro clusters to probe (subsequent_tx)
    nCoarseProbeFirst  = 3   // macro clusters to probe (first_tx)
    nprobeInit         = 8   // micro clusters for fast path
    nprobeMax          = 20  // micro clusters for repair path
    invScale           = float32(1.0 / 10000.0)
)
```

### Phase 1: Macro Centroid Scan

```go
// Scan K1 macro-centroids, find top N_coarse
var topMacro [nCoarseProbeSubseq]centEntry
// Unrolled 16-dim loop (same pattern as current centroid scan)
// Uses incremental base pointer to avoid multiply per centroid
```

### Phase 2: Micro Centroid Scan

```go
// For each macro in topMacro, scan its K2 micro-centroids
// Accumulate into topMicro [nprobeMax]centEntry
// Sort topMicro ascending by dist (needed for centroid pruning)
```

Sorting nprobeMax=20 entries is O(20 log 20) ≈ O(90) — negligible.

### Phase 3: Vector Scan + Adaptive Nprobe

```go
// Scan topMicro[:nprobeInit] (fast path)
// Build top-5 heap (knnEntry{dist, label})
// Apply existing optimizations: bounds hint, dim-0 early exit, distL2i16_16

// Fast exit check after nprobeInit:
fraudCount := countFraud(top)
if len(top) == k {
    if fraudCount == 5 {
        return 5  // all fraud — safe, needs 3+ wrong to become FP
    }
    if fraudCount == 0 && maxDist < idx.DSafeSq {
        return 0  // clearly legit — maxDist < DSafe², within confidence radius
    }
}

// Repair path: scan remaining topMicro[nprobeInit:nprobeMax]
for _, ce := range topMicro[nprobeInit:] {
    // Centroid pruning: lower bound on distance to any vector in this cluster
    dCentroid := sqrt32(ce.dist)
    radius := idx.Radii[ce.id]
    lowerBound := dCentroid - radius
    if lowerBound > 0 && lowerBound*lowerBound > maxDist {
        break  // remaining clusters are even farther — stop
    }
    // ... scan vectors in cluster
}

return countFraud(top)
```

`sqrt32` is `math.Sqrt` cast to float32 — called at most nprobeMax-nprobeInit=12 times per
uncertain query. Negligible cost vs vector scan.

---

## Routing: service/fraud_detection.go

```go
type Service struct {
    firstTxIdx    search.Index
    subsequentIdx search.Index
    norm          *vectorizer.Norm
    mcc           vectorizer.MCCMap
}

func (s *Service) Detect(req dto.FraudRequest) dto.FraudResponse {
    query := vectorizer.Vectorize(req, s.norm, s.mcc)
    var fraudCount int
    if req.LastTransaction == nil {
        fraudCount = s.firstTxIdx.KNN(query, 5)
    } else {
        fraudCount = s.subsequentIdx.KNN(query, 5)
    }
    score := float64(fraudCount) / 5.0
    return dto.FraudResponse{
        Approved:   score < 0.6,
        FraudScore: score,
    }
}
```

Both indexes loaded at startup via `cmd/api/main.go` from `index/first_tx.ivfh` and
`index/subsequent_tx.ivfh`.

---

## Implementation Phases

| Phase | Files Changed | Prerequisite |
|-------|---------------|--------------|
| 1 | `ml/build_index.py` | GPU access |
| 2 | `internal/search/index.go` (add IVFHIndex) | Phase 1 output |
| 3 | `internal/search/knn.go` (new KNN on IVFHIndex) | Phase 2 |
| 4 | `internal/service/fraud_detection.go`, `cmd/api/main.go` | Phase 3 |
| 5 | Tests, bench, PROGRESS.md update | Phase 4 |

---

## Expected Impact

| Metric | Current | Expected |
|--------|---------|----------|
| Phase 1 centroid ops | 128K | ~4K (32×) |
| Fast path vecs/query | 9375 | ~3904 (2.4×) |
| Clear case vecs/query | 9375 | ~200-600 (15-45×) |
| FP | 10 | ≤5 (boundary oversampling) |
| FN | 1 | 0 (repair path) |
| nprobe effective | 25 | 8-20 adaptive |
| Remote p99 | ~30ms | estimated ~10-15ms |

---

## Constraints Unchanged

- `k=5` fixed by spec
- `threshold=0.6` fixed by spec
- `fraud_score = fraudCount / 5.0` fixed by spec
- RAM budget: 168MB per instance
- Both indexes fit within budget (split reduces per-index size vs unified 95MB)
