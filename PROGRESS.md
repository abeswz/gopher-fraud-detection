# PROGRESS

**Last updated:** 2026-06-04

## Status

Dynamic IVF Repair implemented (nprobeRepair=64, 3-phase scan). NCoarseProbe raised to 6/8. Both changes added p99 overhead with negligible accuracy gain: FN 6→5, FP unchanged at 19. Root cause of remaining errors is the NCoarseProbe ceiling — true neighbors live in macro clusters we never visit, no amount of micro-probe expansion helps.

**Current local: +5729** (p99=0.542ms, FP=7, FN=0, NCoarseProbe=12/16, nprobeRepair=128, K1=256/K2=16)
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
- nil → `first_tx.ivfh` (NCoarseProbeFirst=12)
- non-nil → `subsequent_tx.ivfh` (NCoarseProbeSubseq=16)

---

## Index

- **Algorithm:** IVF_H2 — 2-level hierarchical KMeans (cuML `scalable-k-means++`)
- **Format:** IVFH binary (magic "IVFH")
- **first_tx.ivfh** — K1=256, K2=16 (4096 leaf clusters), NCoarseProbe=12
- **subsequent_tx.ivfh** — K1=256, K2=16 (4096 leaf clusters), NCoarseProbe=16
- **nprobeInit=8** (phase 1 fast), **nprobeMax=20** (phase 2 standard repair), **nprobeRepair=128** (phase 3 deep repair)

### Split rationale

`first_tx` = ~20% of data (null `last_transaction` sentinel at dims 5+6 = -1.0).
`subsequent_tx` = ~80% of data.
Separate indexes avoid centroid pollution between the two distributions.

### KNN phases (per query)

1. **Macro scan** — compare all K1 centroids, keep top NCoarseProbe
2. **Micro candidate selection** — scan K2 micro centroids per selected macro, keep top nprobeRepair=64 (sorted ascending by distance)
3. **Phase 1 fast scan** — scan first 8 micro clusters; exit if fraudCount==5 or (fraudCount==0 && maxDist < DSafeSq)
4. **Phase 2 standard repair** — scan clusters 8–19 with centroid-distance pruning; exit if fraudCount==0 or fraudCount==5
5. **Phase 3 deep repair** — only when fraudCount is 1–4 (ambiguous): scan clusters 20–63 with centroid-distance pruning

---

## Accuracy progression

| Config | Score | p99 | FP | FN |
|--------|-------|-----|----|----|
| flat IVF C=8000 nprobe=40 | 5687 | 0.679ms | 10 | 0 |
| IVF_H2 K1=128/K2=32 NCoarseProbe=3/4 | 5526 | 0.385ms | 19 | 6 |
| IVF_H2 + dynamic repair NCoarseProbe=6/8 | 5537 | ~0.39ms | 19 | 5 |
| IVF_H2 K1=256/K2=16 NCoarseProbe=12/16 nprobeRepair=128 | **5729** | **0.542ms** | **7** | **0** |

Rebuilding with K1=256/K2=16 (finer macro clusters, smaller leaf buckets) plus aggressive probe settings was the key gain. FN=0 achieved. FP=7 remains — these are queries where IVF finds 3+ fraud neighbors that are NOT the true top-5. True legit neighbors live in unprobed macro clusters.

**Remaining bottleneck: FP=7.** Each FP costs 1× weighted error (FN costs 3×). E=7 → 300×log10(8)=271 detection penalty. Eliminating all FP = +271 points → theoretical max ~+6000.

---

## What is implemented

- `cmd/api/main.go` — loads two IVFH indexes, serves unix socket
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim → 16-dim padded feature vector
- `internal/search/index.go` — `IVFIndex` (IVF2) + `IVFHIndex` (IVFH) + `Index` interface
- `internal/search/knn.go` — IVFHIndex.KNN (3-phase: fast → standard repair → deep repair)
- `internal/service/fraud_detection.go` — dual-index routing by LastTx
- `internal/handler/fraud_score.go` — HTTP handler
- `ml/build_index.py` — IVF_H2 builder: split, boundary oversample, 2-level KMeans, D_safe, IVFH binary
- Tests — 35 passing

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
| 2026-06-04 | +5526 | 0.385ms | 3000 (MAX) | 19 | 6 | IVF_H2 dual-split NCoarseProbe=3/4 |
| 2026-06-04 | +5537 | ~0.39ms | 3000 (MAX) | 19 | 5 | IVF_H2 + 3-phase repair NCoarseProbe=6/8 |
| 2026-06-04 | +5676 | 0.502ms | 3000 (MAX) | 8 | 1 | K1=256/K2=16 NCoarseProbe=8/12 nprobeRepair=96 |
| **2026-06-04** | **+5729** | **0.542ms** | **3000 (MAX)** | **7** | **0** | **K1=256/K2=16 NCoarseProbe=12/16 nprobeRepair=128** |
