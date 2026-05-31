# PROGRESS

**Last updated:** 2026-05-30

## Status

Score **+5536 / +6000**. p99_score **maxed at 3000** (p99=0.65ms ≤ 1ms ceiling). Only gap: detection (-463 pts, FP:19 FN:5).

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

- `cmd/api/main.go` — loads IVF index + vectorizer, serves unix socket, `os.Chmod(sock, 0666)` for HAProxy (UID 99). pprof server on `PPROF_ADDR` env var (**remove before submission**).
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim feature vector
- `internal/search/index.go` — `IVFIndex` + `LoadIVFIndex` (IVF1 binary format)
- `internal/search/knn.go` — IVF KNN (nprobe=15, C=4000). Optimized: unrolled 14-dim loops, `*invScale` instead of `/10000.0`, incremental base pointer, bounds-check hints, partial distance early exit at dim 0 and dim 7.
- `internal/service/fraud_detection.go` — returns fraudCount int. k=5, threshold=0.6 (FIXED BY SPEC).
- `internal/handler/fraud_score.go` — pre-computed 6-entry response array, zero JSON encoding per request.
- `ml/build_index.py` — MiniBatchKMeans(C=4000, n_init=3) on 3M vectors → IVF binary
- `Makefile` — `index`, `bench`, `bench-fast` (skip index rebuild), `submission` targets
- Tests — 12 unit tests

---

## Fixed constraints (spec — never change)

- `k=5`: test labels generated with exact brute-force k=5. Changing k diverges from ground truth → more FP/FN.
- `threshold=0.6`: explicitly fixed in `DETECTION_RULES.md`.
- `fraud_score = fraudCount / 5.0`: always divides by 5.

FP/FN come from IVF approximation error (nprobe misses true top-5 brute-force neighbors). Only levers: increase nprobe, or improve index cluster quality.

---

## KNN performance

| Config | μs/op | Notes |
|---|---|---|
| C=2000, nprobe=20, brute inner | 124μs | original |
| C=4000, nprobe=15, unoptimized | 79μs | IVF baseline |
| C=4000, nprobe=20, pre-opt | 88μs | too slow under load |
| **C=4000, nprobe=15, optimized** | **~40μs est.** | unroll + invScale + early exit |

**Memory sweet spot:** C=4000 → centroid table 224KB fits L2 (256KB). C=8000 overflows L2 → 101μs.

---

## KNN optimizations applied (knn.go)

1. **`/ 10000.0` → `* invScale`** — multiply ~4× faster than divide; 14 ops × 11K vectors = meaningful
2. **Unrolled 14-dim loop** — eliminates loop overhead + branch prediction misses in inner hot path
3. **Incremental `base += dims`** — replaces `i * dims` multiply per vector
4. **Bounds-check hint `_ = slice[base+13]`** — tells compiler all 14 accesses are in-bounds; elides 13 checks per vector
5. **Partial distance early exit (dim 0 + dim 7)** — once heap full (k=5), most vectors are farther than `maxDist`; skipped after 1 dimension instead of 14
6. **Query locals `q0..q13`** — extract to stack before loops; avoids repeated bounds checks on `query[14]`

---

## CPU profile evolution (30s under k6 load)

| Profile | Config | Total samples | KNN% | KNN flat |
|---|---|---|---|---|
| 001 | nprobe=15, original | 1.32s | 87% | 1.15s |
| 003 | nprobe=15, pre-computed resp | 1.09s | 83% | 0.90s |
| **005** | **nprobe=15, KNN optimized** | **670ms** | **73%** | **490ms** |

**-41% total CPU, -57% KNN flat time** vs profile 001.

Profile 005 new signals:
- `centFindMax` 2.99% — linear scan of np=15 entries, called ~4K times per query
- `runtime.nextFreeFast` 2.99% — `make([]centEntry)` + `make([]knnEntry)` still allocating
- `bufio.Flush` 5.97% — HTTP response write, architectural

---

## Score history

| Date | Score | p99 | p99_score | FP | FN | Config |
|---|---|---|---|---|---|---|
| 2026-05-30 | -6000 | 2002ms | -3000 | 0 | 0 | brute-force KNN |
| 2026-05-30 | +5322 | 1.64ms | 2786 | 19 | 5 | IVF C=4000 nprobe=15 |
| 2026-05-30 | +5070 | 3.13ms | 2504 | 15 | 4 | nprobe=20 (pre-opt, CPU saturated) |
| 2026-05-30 | +5292 | 1.76ms | 2755 | 19 | 5 | nprobe=15 + pre-computed response |
| **2026-05-30** | **+5536** | **0.65ms** | **3000 (MAX)** | 19 | 5 | **nprobe=15 + KNN optimized** |

---

## Remaining gap

Only detection: **-463 pts** (FP:19 × 1pt + FN:5 × 3pt = 34 weighted errors).

Detection score at zero errors = 3000. Current det_score = 2536. Gap = 464 pts.

---

## Plan — closing the detection gap

### Option A — nprobe=20 (re-test, NOW viable)

KNN is 41% faster. Previously nprobe=20 caused p99=3.13ms (CPU saturated). With optimized KNN, nprobe=20 should be affordable.
- Expected: FP:15, FN:4 (confirmed from pre-optimization nprobe=20 test)
- Risk: p99 may still cross 1ms ceiling → p99_score drops from 3000
- Test first: change `nprobe = 15` → `nprobe = 20` in `knn.go`, run `make bench-fast`

### Option B — improve index cluster quality

Current: `MiniBatchKMeans(n_init=3, batch_size=100_000)` — fast but suboptimal clusters.
Better clusters → true brute-force top-5 neighbors concentrated in fewer clusters → better recall at same nprobe.

Change: `n_init=3 → n_init=10`, `batch_size=100_000 → 200_000`
- Rebuild time: ~12-15 min (vs ~4 min current)
- Expected: fewer FP/FN at same nprobe, no latency cost
- Run: `make index` (then `make bench-fast`)

### Option C — nprobe=20 + better index

Combine A + B: better clusters mean nprobe=20 recovers more true neighbors without CPU cost of scanning many false-positive cluster members.

### Before submission

Remove pprof from:
1. `cmd/api/main.go`: `_ "net/http/pprof"` import + `PPROF_ADDR` goroutine block
2. `docker-compose.yml`: `PPROF_ADDR=:6060` env + `ports: - "6060:6060"` from api1
