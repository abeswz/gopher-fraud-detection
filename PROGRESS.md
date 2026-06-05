# PROGRESS

**Last updated:** 2026-06-04

## Status

Partitioning strategy changed from `{is_online, card_present}` (terminal type) to `{unknown_merchant}` (dim 11), applied to both first_tx and subsequent_tx splits. This creates 4 more homogeneous sub-indexes → fewer IVF cache misses → FP 4→1, FN 1→0, score 5729→5909.

**Current local: +5909** (p99=0.606ms, FP=1, FN=0, ERR=0)
**Previous best local: +5729** (p99=0.394ms, FP=4, FN=1, bbox pruning)
Last remote: +3730 (p99=62ms, before GOMAXPROCS=1 fix — stale)

The 1 remaining FP is irreducible: even exhaustive int16 search returns fraud for this case, but float32 ground truth says legit. Int16 quantization error at the k=5 decision boundary.

---

## Architecture

```
Client → HAProxy :9999 (TCP round-robin)
           ├── api1  unix:/run/sock/api1.sock
           └── api2  unix:/run/sock/api2.sock
```

Each instance: 168 MB RAM, 0.45 CPU. HAProxy: 14 MB, 0.10 CPU. Total: 1 CPU, 350 MB.
GOMAXPROCS=1 — prevents CFS throttling on 0.45 CPU quota.

### Fraud pipeline per request

```
JSON decode → vectorize → [fastPath] → [RawTree] → IVF_H2 k-NN → response
```

Routes by `LastTx == nil` × `unknown_merchant`:
- nil + known   → `first_known.ivfh`   (NCoarseProbeFirstKnown=12)
- nil + unknown → `first_unknown.ivfh` (NCoarseProbeFirstUnknown=12)
- non-nil + known   → `subseq_known.ivfh`   (NCoarseProbeSubseqKnown=16)
- non-nil + unknown → `subseq_unknown.ivfh` (NCoarseProbeSubseqUnknown=16)

`unknown_merchant` = `merchant.id` not in `customer.known_merchants` (linear scan, O(len)).

---

## Index

- **Algorithm:** IVF_H2 — 2-level hierarchical KMeans (cuML `scalable-k-means++`)
- **Format:** IVFH binary (magic "IVFH")
- **4 sub-indexes**, K1 auto-scaled via `choose_k1`, K2=32 (first_tx) or K2=16 (subseq)
- **nprobeInit=8** (phase 1 fast), **nprobeMax=20** (phase 2 standard repair), **nprobeRepair=128** (phase 3 deep repair)
- **nprobeThreshRepair=128** (full macro repair cap)

### Split rationale

`unknown_merchant` (dim 11) is the strongest behavioral discriminator — vectors with the same known/unknown status form tighter clusters → KMeans centroids better aligned → fewer IVF approximation errors.

- `first_tx` (~20% of data): null `last_transaction` sentinel at dims 5+6 = -1.0. Split by unknown_merchant → 2 sub-indexes.
- `subseq_tx` (~80% of data): Split by unknown_merchant → 2 sub-indexes.

### KNN phases (per query)

1. **Macro scan** — compare all K1 centroids, keep top NCoarseProbe
2. **Micro candidate selection** — scan K2 micro centroids per selected macro, keep top nprobeRepair (sorted ascending)
3. **Phase 1 fast scan** — scan first 8 micro clusters; exit if fraudCount==5 or (fraudCount==0 && maxDist < DSafeSq)
4. **Phase 2 standard repair** — scan clusters 8–19 with centroid+bbox pruning; exit if fraudCount==0
5. **Phase 3 deep repair** — fraudCount 1–4: scan clusters 20–127 with centroid+bbox pruning
6. **Full macro repair** — fraudCount≥3 after phase 3: re-rank top-128 micro clusters from all K1 macros

---

## Accuracy progression

| Config | Score | p99 | FP | FN |
|--------|-------|-----|----|----|
| flat IVF C=8000 nprobe=40 | 5687 | 0.679ms | 10 | 0 |
| IVF_H2 K1=128/K2=32 NCoarseProbe=3/4 | 5526 | 0.385ms | 19 | 6 |
| IVF_H2 + dynamic repair NCoarseProbe=6/8 | 5537 | ~0.39ms | 19 | 5 |
| IVF_H2 K1=256/K2=16 NCoarseProbe=12/16 nprobeRepair=128 | 5729 | 0.542ms | 7 | 0 |
| IVF_H2 + bbox pruning | 5729 | 0.394ms | 4 | 1 |
| **4-way split by unknown_merchant** | **5909** | **0.606ms** | **1** | **0** |

---

## What is implemented

- `cmd/api/main.go` — loads 4 IVFH indexes, serves unix socket
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim → 16-dim padded feature vector
- `internal/search/index.go` — `IVFHIndex` + `Index` interface
- `internal/search/knn.go` — IVFHIndex.KNN (3-phase: fast → standard repair → deep repair → full macro repair)
- `internal/service/fraud_detection.go` — 4-way routing by LastTx × unknown_merchant
- `internal/handler/fraud_score.go` — HTTP handler
- `ml/build_index.py` — IVF_H2 builder: 4-way split, boundary oversample, 2-level KMeans, D_safe, IVFH binary
- Tests — 36 passing

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
| 2026-06-04 | +5729 | 0.542ms | 3000 (MAX) | 7 | 0 | K1=256/K2=16 NCoarseProbe=12/16 nprobeRepair=128 |
| 2026-06-04 | +5729 | 0.394ms | 3000 (MAX) | 4 | 1 | + bbox pruning (int16 box per micro-cluster) |
| **2026-06-04** | **+5909** | **0.606ms** | **3000 (MAX)** | **1** | **0** | **4-way split by unknown_merchant** |
