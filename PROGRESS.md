# PROGRESS

**Last updated:** 2026-06-04

## Status

IVF_H2 dual-split hierarchical index implemented. First result: score 5526, but accuracy regressed vs flat IVF nprobe=40. Latency improved significantly (0.385ms vs 0.679ms).

**Current IVF_H2 local: +5526** (p99=0.385ms, FP=19, FN=6, NCoarseProbe=3/4)
**Previous best local: +5687** (p99=0.679ms, FP=10, FN=0, flat IVF nprobe=40, C=8000)
Last remote: +4051 (p99=30ms, FP=19, FN=5)

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
JSON decode → vectorize → [fastPath] → [RawTree] → IVF_H2 k-NN → response
```

Routes by `LastTx == nil`:
- nil → `first_tx.ivfh` (NCoarseProbeFirst=3)
- non-nil → `subsequent_tx.ivfh` (NCoarseProbeSubseq=4)

---

## Index

- **Algorithm:** IVF_H2 — 2-level hierarchical KMeans (cuML `scalable-k-means++`)
- **Format:** IVFH binary (magic "IVFH")
- **first_tx.ivfh** — 19 MB, K1=64, K2=32 (2048 leaf clusters), NCoarseProbe=3
- **subsequent_tx.ivfh** — 75.9 MB, K1=128, K2=32 (4096 leaf clusters), NCoarseProbe=4
- **nprobeInit=8** micro clusters (fast path), **nprobeMax=20** (repair path)

### Split rationale

`first_tx` = ~20% of data (null `last_transaction` sentinel at dims 5+6 = -1.0).
`subsequent_tx` = ~80% of data.
Separate indexes avoid centroid pollution between the two distributions.

### KNN phases (per query)

1. **Phase 1** — scan K1 macro centroids, keep top NCoarseProbe
2. **Phase 2** — scan K2 micro centroids per selected macro, keep top nprobeMax (sorted ascending)
3. **Phase 3** — scan vectors in top-8 micro clusters (fast path); fast exits at fraudCount==5 or (fraudCount==0 && dist < DSafeSq); then repair path with centroid pruning (triangle inequality)

---

## Accuracy regression analysis

| Config | Score | p99 | FP | FN |
|--------|-------|-----|----|----|
| flat IVF C=8000 nprobe=40 | 5687 | 0.679ms | 10 | 0 |
| **IVF_H2 NCoarseProbe=3/4** | **5526** | **0.385ms** | **19** | **6** |

IVF_H2 is ~43% faster in p99 but misses more neighbors. Root cause: NCoarseProbe=3/4 + nprobeInit=8 = 24-32 effective micro clusters probed vs 40 in flat IVF. Need to tune upward.

---

## What is implemented

- `cmd/api/main.go` — loads two IVFH indexes, serves unix socket
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim → 16-dim padded feature vector
- `internal/search/index.go` — `IVFIndex` (IVF2) + `IVFHIndex` (IVFH) + `Index` interface
- `internal/search/knn.go` — flat IVF KNN + IVFHIndex.KNN (3-phase hierarchical)
- `internal/service/fraud_detection.go` — dual-index routing by LastTx
- `internal/handler/fraud_score.go` — HTTP handler
- `ml/build_index.py` — IVF_H2 builder: split, boundary oversample, 2-level KMeans, D_safe, IVFH binary
- Tests — 48 passing

---

## Fixed constraints (spec — never change)

- `k=5`: test labels generated with exact brute-force k=5. Changing k diverges from ground truth → more FP/FN.
- `threshold=0.6`: explicitly fixed in `DETECTION_RULES.md`.
- `fraud_score = fraudCount / 5.0`: always divides by 5.

---

## Score history

| Date | Score | p99 | p99_score | FP | FN | Config |
|------|-------|-----|-----------|----|----|--------|
| 2026-05-30 | -6000 | 2002ms | -3000 | 0 | 0 | brute-force KNN |
| 2026-05-30 | +5322 | 1.64ms | 2786 | 19 | 5 | IVF C=4000 nprobe=15 |
| 2026-05-30 | +5536 | 0.65ms | 3000 (MAX) | 19 | 5 | nprobe=15 + KNN optimized |
| 2026-05-31 | +4051 | 30.57ms | 1514 | 19 | 5 | **remote submission** |
| 2026-05-31 | +5533 | 0.42ms | 3000 (MAX) | 20 | 5 | hybrid: fast_path → tree → IVF |
| 2026-06-04 | +5687 | 0.679ms | 3000 (MAX) | 10 | 0 | cuML 8000c flat IVF nprobe=40 |
| **2026-06-04** | **+5526** | **0.385ms** | **3000 (MAX)** | **19** | **6** | **IVF_H2 dual-split NCoarseProbe=3/4** |

---

## Next steps

Tune IVF_H2 to recover detection quality:
1. Increase `NCoarseProbeSubseq` from 4 → 6-8, `NCoarseProbeFirst` from 3 → 4-5
2. Increase `nprobeMax` from 20 → 30-40 (repair path covers more clusters)
3. Run local sweep to find NCoarseProbe that recovers FN=0-1
4. Target: match 5687 with lower p99 (IVF_H2 base latency advantage is ~43%)
