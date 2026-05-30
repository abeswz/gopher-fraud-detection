# PROGRESS

**Last updated:** 2026-05-30

## Status

Fully implemented. IVF search deployed. Pending: `make bench` with new index to validate throughput fix.

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
- `internal/search/knn.go` — IVF KNN (nprobe=20, C=2000 clusters)
- `internal/service/fraud_detection.go` — wires vectorizer + IVF KNN
- `internal/handler/` + `internal/router/` — thin HTTP handlers
- `ml/build_index.py` — MiniBatchKMeans(C=2000) on 3M vectors → IVF binary
- `ml/pyproject.toml` — numpy + scikit-learn
- `Dockerfile` — distroless/static-debian12, copies index/ + resources/
- `docker-compose.yml` — two API instances + HAProxy
- `haproxy.cfg` — TCP round-robin over unix sockets
- `Makefile` — `index`, `bench`, `submission` targets
- Tests — 11 unit tests covering vectorizer, search, service

---

## History

| Date | Event |
|------|-------|
| 2026-05-30 | Full implementation via subagent-driven development |
| 2026-05-30 | Bug: HAProxy (UID 99) couldn't connect to unix socket (perms 0755). Fix: `os.Chmod(sock, 0666)` in main.go |
| 2026-05-30 | `make bench` showed 99% HTTP errors, p99=2002ms, score=-6000 |
| 2026-05-30 | Root cause: brute-force KNN over 3M vectors = ~40ms/req at 0.45 CPU → max ~11 req/s. k6 ramps to 900 req/s |
| 2026-05-30 | Fix: replaced brute-force with IVF (C=2000, nprobe=20). Searches ~30K vecs vs 3M → ~150x speedup |

---

## Last bench result

```
p99:2002ms  score:-6000  FP:0  FN:0  ERR:13712
```

**Before IVF fix** (brute-force, 3M vecs). New index not yet built.

---

## Next step

```bash
make bench   # rebuilds index (K-means ~5min), then Docker + k6
```

Expected after IVF: p99 < 10ms, ERR ≈ 0, score > +4000.
