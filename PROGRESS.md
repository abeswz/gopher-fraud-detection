# PROGRESS

**Last updated:** 2026-06-04

## Status

IVF index rebuilt with cuML `scalable-k-means++` (8000 clusters, n_init=10). nprobe tuned to 40 to compensate for halved avg cluster size vs previous 4000-cluster index.

**Current best local: +5687** (p99=0.679ms, FP=10, FN=0, nprobe=40, C=8000)
Previous best local: +5536 (p99=0.42ms, hybrid pipeline, C=4000 nprobe=15)
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
JSON decode → vectorize → IVF k-NN → response
```

1. **IVF k-NN** — nprobe=40, k=5, C=8000 cuML index

---

## Index

- **Algorithm:** cuML `KMeans(init='scalable-k-means++', n_clusters=8000, n_init=10, max_iter=300)`
- **Format:** IVF2 binary, 95 MB
- **Avg cluster size:** 375 vectors (8000 clusters × 3M vecs)
- **nprobe=40** searches 40×375 = 15000 vecs/query — same coverage as old C=4000 nprobe=20

### Why nprobe=40 for C=8000

Old index (C=4000): avg_size=750, nprobe=20 → 15k vecs/query
New index (C=8000): avg_size=375, nprobe=40 → 15k vecs/query

Doubling clusters halves avg cluster size. Same nprobe value = half the coverage = worse recall.
nprobe must scale proportionally. At nprobe=40, FN=0 (perfect recall on test set).

---

## What is implemented

- `cmd/api/main.go` — loads IVF index, resources, serves unix socket
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim feature vector
- `internal/search/index.go` — `IVFIndex` + `LoadIVFIndex` (IVF2 format, dynamic C from header)
- `internal/search/knn.go` — IVF KNN (nprobe=40, C=8000), fully optimized
- `internal/service/fraud_detection.go` — vectorize → k-NN pipeline
- `internal/handler/fraud_score.go` — HTTP handler
- `ml/build_index.py` — cuML KMeans scalable-k-means++, C=8000
- `ml/pyproject.toml` — cuml-cu12, cupy-cuda12x, python=3.10
- `Makefile` — `index`, `bench`, `bench-fast`, `submission` targets
- Tests — 37 passing

---

## Fixed constraints (spec — never change)

- `k=5`: test labels generated with exact brute-force k=5. Changing k diverges from ground truth → more FP/FN.
- `threshold=0.6`: explicitly fixed in `DETECTION_RULES.md`.
- `fraud_score = fraudCount / 5.0`: always divides by 5.

---

## nprobe sweep (C=8000 cuML index, 2026-06-04)

| nprobe | score | p99 | FP | FN | vecs/query |
|--------|-------|-----|----|----|------------|
| 20 | 5516 | 0.486ms | 28 | 4 | 7500 |
| 25 | 5575 | 0.538ms | 19 | 2 | 9375 |
| 30 | 5597 | 0.583ms | 15 | 2 | 11250 |
| **40** | **5687** | **0.679ms** | **10** | **0** | **15000** |

All p99 ≤ 1ms → p99_score = 3000 (max). Chose nprobe=40 for best detection_score.

---

## Score history

| Date | Score | p99 | p99_score | FP | FN | Config |
|------|-------|-----|-----------|----|----|--------|
| 2026-05-30 | -6000 | 2002ms | -3000 | 0 | 0 | brute-force KNN |
| 2026-05-30 | +5322 | 1.64ms | 2786 | 19 | 5 | IVF C=4000 nprobe=15 |
| 2026-05-30 | +5536 | 0.65ms | 3000 (MAX) | 19 | 5 | nprobe=15 + KNN optimized |
| 2026-05-31 | +4051 | 30.57ms | 1514 | 19 | 5 | **remote submission** |
| 2026-05-31 | +5533 | 0.42ms | 3000 (MAX) | 20 | 5 | hybrid: fast_path → tree → IVF |
| **2026-06-04** | **+5687** | **0.679ms** | **3000 (MAX)** | **10** | **0** | **cuML 8000c nprobe=40** |

---

## Remote vs local gap analysis

Remote p99=30ms vs local p99=0.65ms (46× slower). Root cause: remote CPU is weaker + CFS bandwidth throttling under concurrent load.

IVF at nprobe=40 scans same 15k vecs/query as old nprobe=20 on C=4000. Remote latency impact should be similar (both scan ~15k vectors). Better detection (FN=0) should raise remote score regardless.

---

## Next steps

Submit to remote. Expected: remote score improvement driven by FN reduction (FN was 5, now 0 locally).
p99 may stay similar to last remote submission (~30ms) since vecs/query count unchanged.
