# CLAUDE.md — gopher-fraud-detection

## Objective

High-performance fraud detection API in Go. Rinha de Backend 2026.
Vector search over 3M labeled transactions. Score maximized by minimizing p99 and detection errors.

> **Required maintenance:** After each architectural change or remote test result, rewrite `PROGRESS.md` with the latest state. One file, always current — not a changelog, not appended.

**Deadline:** 2026-06-05T23:59:59.999-03:00

---

## Stack

- **Go 1.26** — stdlib only, no frameworks
- **HAProxy** — TCP round-robin to two unix socket backends
- **uv + numpy** (offline tooling) — index builder only

No Gin, no Fiber, no GORM, no Zap. Use `net/http`, `encoding/json`, `log/slog`.

---

## Architecture

```
Client → HAProxy :9999 → api1 (unix:/run/sock/api1.sock)
                        → api2 (unix:/run/sock/api2.sock)
```

**Resource budget** (hard limits: 1 CPU, 350 MB total):
- Each API instance: 168 MB RAM, 0.45 CPU
- HAProxy: 14 MB RAM, 0.10 CPU

**Fraud pipeline per request:**
1. Decode JSON → `dto.FraudRequest`
2. `vectorizer.Vectorize(req, norm, mcc)` → `[14]float32`
3. `search.Index.KNN(query, k=5)` → fraud count
4. `fraud_score = fraudCount / 5.0`, `approved = fraud_score < 0.6`

**Index format** (`index/references.bin`) — int16 scaled ×10000 (87 MB per instance):
```
[4 bytes]  uint32 LE: N records
[N × 29 bytes]: 14 × int16 LE (vector) + 1 byte label (0=legit, 1=fraud)
```
Sentinel `-1.0` (null last_transaction) encodes as `-10000` int16.

---

## Package Structure

```
cmd/api/main.go              entry: load index + resources, serve unix socket
internal/
  dto/fraud.go               request/response types — do not add logic here
  vectorizer/vectorizer.go   Vectorize(req, norm, mcc) → [14]float32
  search/
    index.go                 LoadIndex(path) → Index
    knn.go                   Index.KNN(query [14]float32, k int) → int
  service/fraud_detection.go wire vectorizer + search
  handler/                   HTTP handlers (keep thin)
  router/router.go           route registration
index/                       gitignored — binary built by ml/build_index.py
resources/                   mcc_risk.json, normalization.json, example-payloads.json
```

---

## Detection Rules

Full spec in `references/rules/DETECTION_RULES.md`. Summary:

- 14 dimensions, `clamp(x) = max(0.0, min(1.0, x))`
- Normalization constants from `resources/normalization.json`
- MCC risk from `resources/mcc_risk.json` (default `0.5` for unknown MCC)
- Indices 5 and 6 use `-1.0` sentinel when `last_transaction` is `null`
- KNN with k=5, Euclidean distance, threshold 0.6

---

## Scoring

`final_score = score_p99 + score_det` (range: −6000 to +6000)

- **p99_score**: log₁₀ scale, ceiling +3000 at p99 ≤ 1ms, floor −3000 at p99 > 2000ms
- **detection_score**: weighted errors (FP=1, FN=3, Err=5); cutoff −3000 if failure_rate > 15%
- HTTP errors are the most expensive failure — return `approved:true, fraud_score:0.0` on internal error rather than 500

Full formula in `references/rules/EVALUATION.md`.

---

## Engineering Priorities

1. Correctness (detection rules must be exact)
2. Memory efficiency (must stay within per-instance budget)
3. Latency (p99 target: sub-10ms)
4. Simplicity

---

## Go Code Standards

Follow standard Go conventions (`gofmt`, `go vet`, `staticcheck`).

- Errors returned, never silenced with `_` except at shutdown
- No globals except index/resources loaded at startup via `sync.Once`
- Table-driven tests where inputs vary; no tests for pure data structs (`dto/`)
- No `interface{}` / `any` without justification
- `context.Context` as first arg on any function that may block or be cancelled
- Prefer value receivers for small structs; pointer receivers for structs holding large data
- `log/slog` for structured logging; `log.Fatal` only at startup

---

## Testing

Tests must cover behavior that can break production:
- `vectorizer.Vectorize`: use the exact examples from `DETECTION_RULES.md` (both legit and fraud cases)
- `search.KNN`: small synthetic index, verify top-5 selection and fraud count
- `service.CalculateFraudScore`: end-to-end with a small in-memory index

Do not write tests for:
- DTOs (no logic)
- Coverage padding
- Happy-path-only routing

---

## Commands

```bash
# Format and lint
go fmt ./...
go vet ./...

# Build index (requires uv + numpy + resources/references.json.gz)
uv run ml/build_index.py

# Test
go test ./...

# Build
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o fraud-api ./cmd/api

# Local bench
make bench

# Push submission branch + open test issue
make submission
```

---

## Commits

Atomic, English, conventional format. Each commit must build and pass tests.
No WIP commits. No "fix typo" chains — squash before pushing.

Examples:
```
feat(search): implement brute-force KNN over int16 index
fix(vectorizer): use UTC when parsing requested_at timestamp
perf(search): reduce allocations in inner distance loop
```

---

## Rules References

All challenge constraints live in `references/rules/`:
- `API.md` — request/response contract
- `ARCHITECTURE.md` — infra limits, docker-compose requirements
- `DETECTION_RULES.md` — vectorization spec (authoritative)
- `DATASET.md` — reference file formats
- `EVALUATION.md` — scoring formula
- `SUBMISSION.md` — how to trigger the official test
- `FAQ.md` — common pitfalls
