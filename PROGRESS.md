# PROGRESS

**Last updated:** 2026-05-31

## Status

VP-tree implementation complete. All 21 tests pass.
Awaiting index rebuild (`uv run ml/build_index.py`) and remote submission.

Previous remote: **+4051 / +6000** (p99=30ms, FP:19, FN:5)
Target with VP-tree: **~5700-6000** (p99<1ms, FP:0, FN:0 exact)

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

- `cmd/api/main.go` — magic-byte auto-detect (IVF1/VPT1), loads correct index, serves unix socket
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim feature vector
- `internal/search/index.go` — `IVFIndex` + `LoadIVFIndex` (IVF1 format) + `Index` interface
- `internal/search/knn.go` — IVF KNN (nprobe=15, C=4000), fully optimized
- `internal/search/vp_index.go` — `VPNode`, `VPIndex`, `LoadVPIndex` (VPT1 binary format)
- `internal/search/vp_knn.go` — `VPIndex.KNN` iterative DFS, branch-and-bound, zero heap allocs
- `internal/service/fraud_detection.go` — uses `search.Index` interface, k=5, threshold=0.6 (FIXED BY SPEC)
- `internal/handler/fraud_score.go` — pre-computed 6-entry response array, zero JSON encoding
- `ml/build_index.py` — `--algo {vptree,ivf}` flag (vptree default); IVF: MiniBatchKMeans(C=4000, n_init=3); VP: recursive DFS tree → VPT1 binary
- `Makefile` — `index`, `bench`, `bench-fast`, `submission` targets
- Tests — 21 unit tests

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

VP-tree eliminates this: ~50-200 vector comparisons per query vs ~15,250 for IVF. Branch-and-bound pruning via triangle inequality. Exact results → FP:0, FN:0.

Scoring formula (from EVALUATION.md):
```
score_p99 = 1000 × log₁₀(1000 / max(p99_ms, 1))
```

| remote p99 target | p99_score | total score |
|---|---|---|
| 30ms (current IVF) | 1514 | 4051 |
| 3ms | 2523 | 5523 (det_score=3000) |
| 1ms | 3000 | 6000 (MAX) |

---

## VP-tree binary format (VPT1)

```
[4B]   "VPT1" magic
[4B]   uint32 N          — total vectors
[4B]   uint32 nodeCount
[4B]   uint32 leafSize   — stored, not used at query time

[nodeCount × 40B] node array:
  [4B]  float32 tau       — split radius; 0 for leaves
  [4B]  uint32  childOff  — right child node index (internal) | vec array start (leaf)
  [2B]  uint16  count     — 0 = internal; >0 = leaf (# vecs ≤ 16)
  [2B]  pad
  [28B] int16[14] vec     — pivot ×10000; zeroed for leaves

[N × 28B]  int16[14] vectors — DFS-reordered, ×10000
[N × 1B]   uint8     labels  — DFS-reordered
```

Memory: ~135 MB per instance (84 MB vectors + 48 MB nodes + 3 MB labels) < 168 MB budget.

---

## Next steps

1. **Rebuild index:** `uv run ml/build_index.py` — produces VPT1 format (~10-20 min on 3M vectors)
2. **Bench locally:** `make bench`
3. **Submit:** `make submission`

---

## KNN performance history

| Config | μs/op | Notes |
|---|---|---|
| C=2000, nprobe=20, brute inner | 124μs | original |
| C=4000, nprobe=15, unoptimized | 79μs | IVF baseline |
| C=4000, nprobe=20, pre-opt | 88μs | too slow under load |
| **C=4000, nprobe=15, optimized** | **~40μs** | IVF best |
| **VP-tree (target)** | **~1-5μs** | exact, 50-200 comparisons |
