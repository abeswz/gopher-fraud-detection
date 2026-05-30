# Design: gopher-fraud-detection

**Date:** 2026-05-30  
**Repo:** https://github.com/abeswz/gopher-fraud-detection  
**Participant ID:** abeswz-gopher  
**Goal:** Minimal correct Go implementation of the Rinha de Backend 2026 fraud detection challenge. No performance optimization — correctness first.

---

## Architecture

Two API instances behind HAProxy (round-robin, TCP mode, unix sockets). Constraints: 1 CPU total, 350 MB RAM total across all services.

```
Client → HAProxy :9999 → api1 (unix:/run/sock/api1.sock)
                        → api2 (unix:/run/sock/api2.sock)
```

Each API instance: 168 MB RAM, 0.45 CPU.  
HAProxy: 14 MB RAM, 0.10 CPU.

---

## Memory Constraint & Binary Format

Brute-force KNN over 3M float32 vectors = 168 MB vectors + 3 MB labels + ~30 MB Go runtime = **~201 MB per instance**. Exceeds the 168 MB limit.

Solution: store vectors as **int16 scaled ×10000**. Range [-1.0, 1.0] maps to [-10000, 10000] (fits int16). The sentinel value -1 (for null `last_transaction`) maps to -10000.

Memory: 3,000,000 × (14 × 2 + 1) = **87 MB per instance**. ~80 MB headroom for Go runtime.

Conversion at query time:
```go
float32(v) / 10000.0
```

Zero external dependencies required.

### Binary file format (`index/references.bin`)

```
[4 bytes]  uint32, little-endian: N (number of vectors, = 3,000,000)
[N × 29 bytes]:
  [28 bytes] 14 × int16, little-endian (vector dimensions)
  [1 byte]   label: 0 = legit, 1 = fraud
```

Total size: ~87 MB.

---

## Package Structure

```
cmd/api/main.go              entry point — load index + resources, start HTTP server on unix socket
internal/
  dto/fraud.go               request/response types (exists, keep as-is)
  vectorizer/
    vectorizer.go            Vectorize(req, norm, mcc) → [14]float32
  search/
    index.go                 LoadIndex(path) → Index struct ([]int16 vectors, []uint8 labels)
    knn.go                   Index.KNN(query [14]float32, k int) → fraudCount int
  service/
    fraud_detection.go       wire vectorizer + search → FraudResponse
  handler/                   (exists, keep)
  router/                    (exists, keep)
ml/
  build_index.py             references.json.gz → index/references.bin
  pyproject.toml             uv deps: numpy only
index/                       gitignored — binary output lives here
  references.bin
resources/                   (exists) mcc_risk.json, normalization.json
```

---

## Vectorizer

Implements the 14-dim transformation from `DETECTION_RULES.md`.

Loaded at startup:
- `resources/normalization.json` → `Normalization` struct
- `resources/mcc_risk.json` → `map[string]float32`

`clamp(x)` = `max(0.0, min(1.0, x))`

| idx | field | formula |
|-----|-------|---------|
| 0 | amount | clamp(amount / max_amount) |
| 1 | installments | clamp(installments / max_installments) |
| 2 | amount_vs_avg | clamp((amount / avg_amount) / amount_vs_avg_ratio) |
| 3 | hour_of_day | hour(requested_at UTC) / 23.0 |
| 4 | day_of_week | weekday(requested_at UTC) / 6.0  (Mon=0, Sun=6) |
| 5 | minutes_since_last_tx | clamp(minutes / max_minutes) or -1 if last_tx null |
| 6 | km_from_last_tx | clamp(km_from_current / max_km) or -1 if last_tx null |
| 7 | km_from_home | clamp(km_from_home / max_km) |
| 8 | tx_count_24h | clamp(tx_count_24h / max_tx_count_24h) |
| 9 | is_online | 1.0 if true, else 0.0 |
| 10 | card_present | 1.0 if true, else 0.0 |
| 11 | unknown_merchant | 1.0 if merchant.id not in known_merchants, else 0.0 |
| 12 | mcc_risk | mcc_risk[merchant.mcc], default 0.5 |
| 13 | merchant_avg_amount | clamp(merchant.avg_amount / max_merchant_avg_amount) |

`minutes_since_last_tx`: diff in minutes between `transaction.requested_at` and `last_transaction.timestamp`.

---

## KNN Search

Algorithm: **brute-force L2**, no approximation.

```
for each reference vector r:
    dist = sum((query[i] - r[i])^2, i=0..13)
maintain min-heap of size k=5
return fraud count among top-5
```

`fraud_score = fraudCount / 5.0`  
`approved = fraud_score < 0.6`

No SIMD, no concurrency per request. Simple inner loop. Acceptable latency for "correctness first" goal — expected p99 in the hundreds of ms range.

---

## ML Tooling (`ml/`)

Not committed to git (`ml/` excluded from `.gitignore`... wait — `ml/` is local and the index binary is large; add `index/` to `.gitignore`).

`ml/build_index.py`:
1. Open `resources/references.json.gz`
2. Parse JSON array (streaming if possible)
3. For each record: scale float32 vector × 10000 → int16, encode label (fraud=1, legit=0)
4. Write binary to `index/references.bin`
5. Print: N vectors written, file size

`ml/pyproject.toml`: `uv` project, deps: `numpy`.

Run: `uv run ml/build_index.py`

---

## Dockerfile

Multi-stage build:

```dockerfile
FROM golang:1.26 AS builder
WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o fraud-api ./cmd/api

FROM gcr.io/distroless/static-debian12
COPY --from=builder /app/fraud-api /fraud-api
COPY index/ /app/index/
ENV INDEX_PATH=/app/index/references.bin
ENV GOMAXPROCS=2
ENTRYPOINT ["/fraud-api"]
```

Note: switching from `FROM scratch` to `distroless/static` to allow easier debugging if needed. Can revert to scratch.

The `index/` directory must be populated (via `make index`) before running `docker build`.

---

## Makefile

Targets: `index`, `bench`, `submission`.

```makefile
IMAGE_REPO     := ghcr.io/abeswz/gopher-fraud-detection
GIT_SHA        := $(shell git rev-parse --short HEAD)
IMAGE          := $(IMAGE_REPO):$(GIT_SHA)
PORT           := 9999
READY_TIMEOUT  := 300
PARTICIPANT    := abeswz-gopher
RINHA_REPO     := zanfranceschi/rinha-de-backend-2026

index:
    uv run ml/build_index.py

bench: index
    docker compose --compatibility down
    docker compose --compatibility up --build --force-recreate -d
    @i=0; until curl -sf http://localhost:$(PORT)/ready >/dev/null 2>&1; do \
        printf '.'; sleep 1; i=$$((i+1)); \
        [ $$i -ge $(READY_TIMEOUT) ] && echo " timeout" && exit 1; \
    done; echo " ready"
    k6 run test/test.js
    @jq -r '"p99:\(.p99) score:\(.scoring.final_score) FP:\(.scoring.breakdown.false_positive_detections) FN:\(.scoring.breakdown.false_negative_detections) ERR:\(.scoring.breakdown.http_errors)"' test/results.json

submission: index
    docker build --network=host -t $(IMAGE) .
    docker push $(IMAGE)
    @ORIG=$$(git rev-parse --abbrev-ref HEAD); \
    sed 's|build: \.|image: $(IMAGE)|' docker-compose.yml > /tmp/sub-compose.yml; \
    cp info.json /tmp/sub-info.json; \
    cp haproxy.cfg /tmp/sub-haproxy.cfg; \
    git checkout --orphan submission-tmp; \
    git rm -rf . >/dev/null 2>&1; \
    cp /tmp/sub-compose.yml docker-compose.yml; \
    cp /tmp/sub-info.json info.json; \
    cp /tmp/sub-haproxy.cfg haproxy.cfg; \
    git add docker-compose.yml info.json haproxy.cfg; \
    git commit -m "submission: $(GIT_SHA)"; \
    git branch -D submission 2>/dev/null || true; \
    git branch -m submission-tmp submission; \
    git push origin submission --force; \
    git checkout $$ORIG; \
    echo "pushed submission branch"
    gh issue create \
        --repo $(RINHA_REPO) \
        --title "rinha/test $(PARTICIPANT)" \
        --body "rinha/test $(PARTICIPANT)"
```

---

## info.json

```json
{
    "participants": ["abesnow"],
    "social": ["https://github.com/abeswz", "https://www.linkedin.com/in/anves"],
    "source-code-repo": "https://github.com/abeswz/gopher-fraud-detection",
    "stack": ["go", "haproxy"],
    "open_to_work": false
}
```

---

## .gitignore additions

```
index/
ml/
```

Both `index/` (large binary) and `ml/` (local tooling, not for GitHub) are excluded.

---

## API

`GET /ready` — returns 200 once index is loaded into memory.  
`POST /fraud-score` — vectorize → KNN → respond.

Use `sync.Once` or an `atomic.Bool` to signal readiness after index load completes. The `/ready` handler blocks (or returns 503) until `loaded = true`.

---

## Error handling

- Malformed JSON → 400
- Index load failure → fatal (no point serving without data)
- All other paths internal to the server → 500

---

## Testing

Unit tests for:
- `vectorizer.Vectorize` — known input → expected 14-dim vector (use examples from DETECTION_RULES.md)
- `search.KNN` — small synthetic dataset, verify top-5 correct
- `service.CalculateFraudScore` — end-to-end with mock index
