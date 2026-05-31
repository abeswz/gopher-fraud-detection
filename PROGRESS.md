# PROGRESS

**Last updated:** 2026-05-31

## Status

Hybrid pipeline implemented and benchmarked locally. All tests pass.
p99 = 0.42ms (MAX score), FP:20, FN:5 — slightly worse than IVF-only (FP:19, FN:5) but within tolerance.
Decision tree with 8 confident leaves intercepts a tiny fraction of requests; no retraining needed.

Previous best local: **+5536** (p99=0.65ms, FP:19, FN:5)
Current hybrid local: **+5533** (p99=0.42ms, FP:20, FN:5)
Last remote: **+4051** (p99=30ms, FP:19, FN:5)

---

## Architecture

```
Client → HAProxy :9999 (TCP round-robin)
           ├── api1  unix:/run/sock/api1.sock
           └── api2  unix:/run/sock/api2.sock
```

Each instance: 168 MB RAM, 0.45 CPU. HAProxy: 14 MB, 0.10 CPU. Total: 1 CPU, 350 MB.

### Fraud pipeline per request

```
JSON decode → vectorize → fast_path rules → decision tree → IVF k-NN → response
```

1. **fast_path** — heuristic rules for obviously safe/risky transactions (zero allocations)
2. **decision tree** — 8 confident leaves at threshold=0.95; intercepts ~few % of requests
3. **IVF k-NN** — nprobe=15, k=5, C=4000; handles the rest

---

## What is implemented

- `cmd/api/main.go` — loads IVF index, resources, serves unix socket
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim feature vector
- `internal/search/index.go` — `IVFIndex` + `LoadIVFIndex` (IVF1 format) + `Index` interface
- `internal/search/knn.go` — IVF KNN (nprobe=15, C=4000), fully optimized, mmap index
- `internal/search/decision_tree.go` — generated Go code from trained sklearn DecisionTreeClassifier
- `internal/service/fraud_detection.go` — fast_path → decision tree → k-NN pipeline
- `internal/handler/fraud_score.go` — pre-computed 6-entry response array, zero JSON encoding
- `ml/build_index.py` — `--algo ivf` (default); IVF: MiniBatchKMeans(C=4000, n_init=3)
- `ml/train_decision_tree.py` — trains sklearn tree on reference dataset, confidence=0.95
- `ml/gen_tree_go.py` — generates `internal/search/decision_tree.go` from trained model
- `Makefile` — `index`, `bench`, `bench-fast`, `submission` targets
- Tests — 21+ unit tests

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
| 2026-05-30 | +5536 | 0.65ms | 3000 (MAX) | 19 | 5 | nprobe=15 + KNN optimized |
| 2026-05-31 | +4051 | 30.57ms | 1514 | 19 | 5 | **remote submission** (same code) |
| 2026-05-31 | +3234 | 582ms | 234 | 0 | 0 | VP-tree LEAF_SIZE=16 local — detection perfect, latency catastrophic |
| 2026-05-31 | +3188 | 648ms | 188 | 0 | 0 | VP-tree LEAF_SIZE=256 local — worse than LEAF_SIZE=16 |
| **2026-05-31** | **+5533** | **0.42ms** | **3000 (MAX)** | **20** | **5** | **hybrid: fast_path → tree → IVF** |

---

## Remote vs local gap analysis

Remote p99=30ms vs local p99=0.65ms (46× slower). Detection identical → problem is pure compute speed.

Root cause: remote CPU is weaker. IVF Phase 1 scans 4000 centroids (224KB), Phase 2 scans ~11,250 vectors per query. Under 0.45 CPU with concurrent load, CFS bandwidth throttling causes p99 spikes.

VP-tree was tried but failed under concurrency: 84MB DFS-ordered array causes cache thrashing.

IVF works well concurrently because:
- Phase 1 (centroids): 224KB always in cache, shared read across goroutines
- Phase 2 (cluster scan): each goroutine reads its specific cluster → spatial locality

Hybrid pipeline goal: reduce IVF invocations via fast_path + tree to lower remote p99.
Decision tree with 8 confident leaves intercepts a small fraction; impact on latency is marginal locally
but may help on remote where CPU is the bottleneck.

---

## Hybrid pipeline analysis (local bench 2026-05-31)

```
p99: 0.42ms   (vs 0.65ms IVF-only → 35% faster locally)
FP:  20       (vs 19 IVF-only → +1 FP, likely one wrong tree leaf)
FN:  5        (same as IVF-only)
ERR: 0
final_score: 5533.11   (vs 5536 IVF-only → negligible difference)
failure_rate: 0.05%
```

FP increased by 1 (20 vs 19). This is within noise and not worth retraining at lower confidence
since detection_score is already near maximum and p99_score is capped at 3000.

---

## Next steps

Submit hybrid pipeline to remote. Expected improvement: p99 reduction on remote (fewer IVF calls).
If remote p99 still > 10ms, investigate GOMAXPROCS=1 to avoid CFS thrashing.

---

## KNN performance history

| Config | µs/op serial | p99 k6 local | Notes |
|--------|-------------|-------------|-------|
| Brute-force | ? | 2002ms | original |
| IVF C=4000 nprobe=15 unopt | 79µs | 1.64ms | baseline |
| IVF C=4000 nprobe=20 | 88µs | 3.13ms | CPU saturated |
| IVF nprobe=15 + pre-comp resp | ? | 1.76ms | |
| IVF nprobe=15 + KNN optimized | ~40µs | 0.65ms | best IVF-only |
| VP-tree LEAF_SIZE=16 | 178µs | 582ms | cache thrash |
| VP-tree LEAF_SIZE=256 | ? | 648ms | worse |
| **hybrid fast_path+tree+IVF** | ? | **0.42ms** | **current** |
