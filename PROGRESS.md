# PROGRESS

**Last updated:** 2026-05-30

## Status

Fully implemented. IVF search optimized (C=4000, nprobe=15). Pending: `make bench` to validate end-to-end score.

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

- `cmd/api/main.go` — loads IVF index + vectorizer, serves unix socket, `os.Chmod(sock, 0666)` for HAProxy (UID 99)
- `internal/dto/fraud.go` — request/response types
- `internal/vectorizer/vectorizer.go` — 14-dim feature vector (normalization, MCC risk, weekday `(int(wd)+6)%7`, sentinel -1.0 for null last_transaction)
- `internal/search/index.go` — `IVFIndex` + `LoadIVFIndex` (IVF1 binary format)
- `internal/search/knn.go` — IVF KNN (nprobe=15, C=4000 clusters)
- `internal/service/fraud_detection.go` — wires vectorizer + IVF KNN
- `internal/handler/` + `internal/router/` — thin HTTP handlers
- `ml/build_index.py` — MiniBatchKMeans(C=4000) on 3M vectors → IVF binary
- `ml/pyproject.toml` — numpy + scikit-learn
- `Dockerfile` — distroless/static-debian12, copies index/ + resources/
- `docker-compose.yml` — two API instances + HAProxy
- `haproxy.cfg` — TCP round-robin over unix sockets
- `Makefile` — `index`, `bench`, `submission` targets
- Tests — 12 unit tests covering vectorizer, search, service

---

## KNN performance (Go microbenchmark, real 3M index)

| Config | μs/op | Vectors/query | Change |
|---|---|---|---|
| C=2000, nprobe=20 (baseline) | 124μs | 30,000 | — |
| C=4000, nprobe=20 | 88μs | 15,000 | -29% |
| C=4000, nprobe=15 | **79μs** | 11,250 | -36% total |

**Bottleneck confirmed:** memory-bound, not compute-bound. pprof shows 2.84s stalled on `idx.Vectors` loads. Integer arithmetic path tested (eliminates float conversion) → neutral result, reverted.

**Why C=4000 is the sweet spot:** centroid table at C=4000 = 224KB → fits in L2 cache (256KB). At C=8000 = 448KB → overflows L2, phase-1 centroid scan regressed to 101μs.

---

## History

| Date | Event |
|------|-------|
| 2026-05-30 | Full implementation via subagent-driven development |
| 2026-05-30 | Bug: HAProxy (UID 99) couldn't connect to unix socket (perms 0755). Fix: `os.Chmod(sock, 0666)` in main.go |
| 2026-05-30 | `make bench` showed 99% HTTP errors, p99=2002ms, score=-6000 |
| 2026-05-30 | Root cause: brute-force KNN over 3M vectors = ~40ms/req at 0.45 CPU → max ~11 req/s |
| 2026-05-30 | Fix: replaced brute-force with IVF (C=2000, nprobe=20) → ~150x speedup |
| 2026-05-30 | Perf investigation: profiled with pprof, confirmed memory-bound |
| 2026-05-30 | Tuned C=4000 (L2 cache sweet spot) + nprobe=15 → 79μs/op (-36% vs original IVF) |

---

## Last bench result

```
p99:2002ms  score:-6000  FP:0  FN:0  ERR:13712
```

**Before IVF fix** (brute-force). Running `make bench` now with optimized IVF.

---

## Next step

```bash
make bench   # rebuilds index (K-means ~4min), then Docker + k6
```

Expected: p99 < 10ms, ERR ≈ 0, score > +4000.
