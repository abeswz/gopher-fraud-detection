# PROGRESS

**Last updated:** 2026-05-30

## Status

Score **+5322 / +6000**. IVF optimized (C=4000, nprobe=15). Two gaps remain: detection (-463 pts) and latency (-214 pts).

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
p99:1.64ms  score:+5322  FP:19  FN:5  ERR:0
```

C=4000, nprobe=15. KNN microbenchmark: 79μs/op. Full request p99: 1.64ms → 1.56ms spent outside KNN (JSON, vectorizer, socket, HAProxy).

---

## Plan — reaching +6000

### Gap breakdown

| Gap | Points lost | Max recoverable |
|---|---|---|
| Detection errors (19 FP + 5 FN) | -463 | +463 |
| Latency p99=1.64ms vs ≤1ms | -214 | +214 |

### Step 1 — tune threshold and k (no index rebuild, low risk)

**Why:** 19 FP weight 1 each, 5 FN weight 3 each = 34 weighted errors. FN costs 3× more, so reducing FN is priority. Different k or threshold can shift this balance.

**What to test:**
- `k=7` (currently 5) — more voters → more robust majority, likely fewer FN
- `threshold=0.5` (currently 0.6) — harder to approve → fewer FN, likely more FP
- `threshold=0.7` — easier to approve → fewer FP, likely more FN

**How:** change constants in `service/fraud_detection.go`, run `make bench` (no index rebuild needed — skip `make index` by running docker steps directly), compare FP/FN/score.

**Constraint:** `make bench` always runs `index` (4 min rebuild). Workaround: run docker+k6 steps manually to iterate faster.

### Step 2 — profile full HTTP request path

**Why:** KNN=79μs but p99=1.64ms → 1.56ms unaccounted. Closing the latency gap to <1ms requires knowing where that time goes.

**What to do:** add `net/http/pprof` endpoint to the API (behind a build tag or env var), run `make bench`, capture CPU profile under real load with `go tool pprof http://localhost:PORT/debug/pprof/profile?seconds=30`.

**Candidates for hidden latency:**
- `encoding/json` Unmarshal/Marshal (reflection-based, slow)
- `time.Parse` inside vectorizer for `requested_at`
- GC pressure from per-request allocations (slice growth in KNN, JSON decode)
- HAProxy → unix socket round-trip overhead

### Step 3 — reduce allocations in hot path (after profiling confirms)

**Likely fixes based on common Go patterns:**
- Pre-allocate `topC` and `top` slices outside KNN, pass as args (avoids `make` per request)
- Replace `encoding/json` with `github.com/bytedance/sonic` or hand-rolled decoder for the known-schema request
- Reuse `[14]float32` query vector (already on stack — fine)

### Step 4 — nprobe tuning (after detection is solved)

nprobe=15 may miss some real neighbors in clusters 16-20 → contributes to FP/FN. Test nprobe=18 after detection is improved to see if recall gains outweigh latency cost (~15μs more).

---

## Score history

| Date | Score | p99 | FP | FN | ERR | Config |
|---|---|---|---|---|---|---|
| 2026-05-30 | -6000 | 2002ms | 0 | 0 | 13712 | brute-force KNN |
| 2026-05-30 | **+5322** | **1.64ms** | 19 | 5 | 0 | IVF C=4000 nprobe=15 |
