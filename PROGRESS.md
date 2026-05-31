# PROGRESS

**Last updated:** 2026-05-31

## Status

Local score **+5536 / +6000**. Remote submission **+4051 / +6000**.
Remote gap is purely latency: p99=30ms remote vs 0.96ms local.
Detection identical both environments: FP:19, FN:5.

Next path: **VP-tree** — exact search, O(log N) queries, FP:0 FN:0 target, eliminates remote latency gap.

---

## Architecture

```
Client → HAProxy :9999 (TCP round-robin)
           ├── api1  unix:/run/sock/api1.sock
           └── api2  unix:/run/sock/api2.sock
```

Each instance: 168 MB RAM, 0.45 CPU. HAProxy: 14 MB, 0.10 CPU. Total: 1 CPU, 350 MB.

---

## What is implemented

- `cmd/api/main.go` — loads IVF index + vectorizer, serves unix socket, `os.Chmod(sock, 0666)` for HAProxy (UID 99).
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim feature vector
- `internal/search/index.go` — `IVFIndex` + `LoadIVFIndex` (IVF1 binary format)
- `internal/search/knn.go` — IVF KNN (nprobe=15, C=4000). Optimized: unrolled 14-dim loops, `*invScale`, incremental base pointer, bounds-check hints, partial distance early exit at dim 0 and dim 7, stack-allocated working arrays.
- `internal/service/fraud_detection.go` — returns fraudCount int. k=5, threshold=0.6 (FIXED BY SPEC).
- `internal/handler/fraud_score.go` — pre-computed 6-entry response array, zero JSON encoding per request.
- `ml/build_index.py` — MiniBatchKMeans(C=4000, n_init=3) on 3M vectors → IVF binary
- `Makefile` — `index`, `bench`, `bench-fast`, `submission` targets
- Tests — 12 unit tests

---

## Fixed constraints (spec — never change)

- `k=5`: test labels generated with exact brute-force k=5. Changing k diverges from ground truth → more FP/FN.
- `threshold=0.6`: explicitly fixed in `DETECTION_RULES.md`.
- `fraud_score = fraudCount / 5.0`: always divides by 5.

---

## Score history

| Date | Score | p99 | p99_score | FP | FN | Config |
|---|---|---|---|---|---|---|
| 2026-05-30 | -6000 | 2002ms | -3000 | 0 | 0 | brute-force KNN |
| 2026-05-30 | +5322 | 1.64ms | 2786 | 19 | 5 | IVF C=4000 nprobe=15 |
| 2026-05-30 | +5070 | 3.13ms | 2504 | 15 | 4 | nprobe=20 (CPU saturated) |
| 2026-05-30 | +5292 | 1.76ms | 2755 | 19 | 5 | nprobe=15 + pre-computed response |
| **2026-05-30** | **+5536** | **0.65ms** | **3000 (MAX)** | 19 | 5 | **nprobe=15 + KNN optimized** |
| 2026-05-31 | +4051 | 30.57ms | 1514 | 19 | 5 | **remote submission** (same code) |

---

## Remote vs local gap analysis

Remote p99=30ms vs local p99=0.96ms (31× slower). Detection identical → problem is pure compute speed.

Root cause: remote CPU is weaker. IVF Phase 1 scans 4000 centroids (224KB), Phase 2 scans ~11,250 vectors per query. Under 0.45 CPU with concurrent load, CFS bandwidth throttling causes p99 spikes.

Attempted fixes:
- `GOMAXPROCS(1)`: hurt local (p99 1.37ms, score 5400). Reverted — runtime auto-detects on remote.
- nprobe tuning: nprobe=20 → more errors or worse p99. nprobe=16 → crossed 1ms ceiling. Not viable.
- n_init=10 index rebuild: FP/FN slightly worse (different local minimum). Reverted to n_init=3.

Scoring formula (from EVALUATION.md):
```
score_p99 = 1000 × log₁₀(1000 / max(p99_ms, 1))
```
Every 10× faster = +1000 pts.

| remote p99 target | p99_score | total score |
|---|---|---|
| 30ms (current) | 1514 | 4051 |
| 10ms | 2000 | 4537 |
| 3ms | 2523 | 5060 |
| 1ms | 3000 | 5537 |

---

## Next: VP-tree (exact search)

### Why VP-tree beats IVF

IVF is approximate: KMeans clusters are spatial approximations. Queries that fall between cluster boundaries miss true top-5 neighbors → FP/FN.

VP-tree is **exact binary partitioning by actual distance** — the "sorted array enables binary search" principle extended to N dimensions:

- Binary search: split [0..100] at midpoint, eliminate half the search space
- VP-tree: pick pivot, compute median distance tau, partition into inner ball (dist ≤ tau) and outer ball (dist > tau), recurse with pruning

Because partitioning is by real distance (not approximate centroid), VP-tree finds the exact same top-5 as brute force → same labels → **FP:0, FN:0 theoretical**.

| | IVF (current) | VP-tree (target) |
|---|---|---|
| partition quality | KMeans approximation | exact distance threshold |
| comparisons/query | ~15,250 (Phase 1+2) | ~50-200 (with pruning) |
| FP/FN | 19/5 | 0/0 (exact) |
| det_score | 2536 | 3000 |
| p99 on remote (est.) | 30ms | <1ms |

Projected remote score with VP-tree: **~5700-6000** (vs current 4051).

### Memory budget

- Vectors (int16, 3M × 14 × 2B): 84MB
- VP-tree nodes (16B × 3M): 48MB
- Labels (uint8, 3M): 3MB
- Total: ~135MB < 168MB ✓

### Implementation plan

**Python build (`ml/build_index.py`):**
1. Recursive tree construction: pick random pivot, compute distances to all points in partition, split at median into inner/outer
2. Store nodes as flat array (pivot index, tau, left child, right child) + reordered vectors + labels
3. New binary format: `[uint32 N][uint32 nodeCount][nodes...][vectors...][labels...]`

**Go query (`internal/search/`):**
1. `LoadVPIndex(path)` → `VPIndex` struct
2. `VPIndex.KNN(query, k)` → branch-and-bound with max-heap pruning
3. Prune subtree when `|dist - tau| > maxDist` (inner) or `dist - tau > maxDist` (outer)

**Tests:**
- Small synthetic tree, verify exact k=5 results match brute force

### Build time estimate

O(N log N) distance computations. With numpy vectorized distances on 3M vectors × 22 levels: ~10-20 min build. Acceptable.

---

## KNN performance history

| Config | μs/op | Notes |
|---|---|---|
| C=2000, nprobe=20, brute inner | 124μs | original |
| C=4000, nprobe=15, unoptimized | 79μs | IVF baseline |
| C=4000, nprobe=20, pre-opt | 88μs | too slow under load |
| **C=4000, nprobe=15, optimized** | **~40μs est.** | current best |
| VP-tree (target) | **~1-5μs est.** | 50-200 comparisons |

---

## KNN optimizations applied (knn.go, current IVF)

1. `/ 10000.0` → `* invScale` — multiply ~4× faster than divide
2. Unrolled 14-dim loop — eliminates loop overhead
3. Incremental `base += dims` — replaces `i * dims` multiply per vector
4. Bounds-check hint `_ = slice[base+13]` — elides 13 checks per vector
5. Partial distance early exit (dim 0 + dim 7) — skip clearly-distant vectors
6. Query locals `q0..q13` — extract to stack before loops
7. Stack-allocated working arrays `[nprobe]centEntry`, `[5]knnEntry` — eliminates heap allocs per request
