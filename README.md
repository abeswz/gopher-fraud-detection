# gopher-fraud-detection

Fraud detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

Two Go instances behind HAProxy. Each loads a 3M-vector IVF index at startup and scores transactions via approximate KNN (k=5, L2, IVF C=2000 nprobe=20) over a 14-dim feature vector.

**Limits:** 1 CPU, 350 MB RAM total.

---

## Prerequisites

- Go 1.26+
- Docker + Docker Compose
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [k6](https://k6.io/docs/get-started/installation/) (load testing)
- `jq`

Files not in the repo (obtain separately):
```
resources/references.json.gz   # 3M labeled transactions
resources/normalization.json    # normalization constants
resources/mcc_risk.json         # MCC risk scores
test/test.js                    # k6 test script
test/test-data.json             # k6 test dataset
```

---

## Local development

```bash
# Run unit tests
go test ./...

# Build binary
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o fraud-api ./cmd/api
```

---

## Local benchmark

Builds the IVF index (K-means, ~5 min first run), spins up the full stack, runs k6:

```bash
make bench
```

Output:
```
p99:4ms score:5800 FP:12 FN:3 ERR:0
```

---

## Architecture

```
Client → HAProxy :9999
           ├── api1  unix:/run/sock/api1.sock
           └── api2  unix:/run/sock/api2.sock
```

Each instance loads `index/references.bin` (IVF: 2000 clusters, int16 vectors) at startup. No shared state.

Pipeline per request:
1. Decode JSON → `FraudRequest`
2. Vectorize → `[14]float32` (normalization + MCC risk)
3. IVF KNN(k=5, nprobe=20): find 20 nearest clusters → search ~30K vectors → fraud count
4. `fraud_score = count / 5.0`, `approved = score < 0.6`
