# gopher-fraud-detection

Fraud detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

Two Go API instances behind HAProxy. Each loads a 3M-vector int16 index at startup and scores transactions via brute-force KNN (k=5, L2) over a 14-dim feature vector.

**Limits:** 1 CPU, 350 MB RAM total.

---

## Prerequisites

- Go 1.26+
- Docker + Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [k6](https://k6.io/docs/get-started/installation/) (load testing)
- `jq`, `gh` (GitHub CLI)

The following files are not in the repo and must be obtained separately:
```
resources/references.json.gz   # 3M labeled transactions (source data)
resources/normalization.json    # normalization constants
resources/mcc_risk.json         # MCC risk scores
test/test.js                    # k6 test script
```

---

## Local development

```bash
# Build the vector index from source data (~87 MB, takes ~10s)
make index

# Run all unit tests
go test ./...

# Build the binary
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o fraud-api ./cmd/api
```

---

## Local benchmark

Spins up the full stack (HAProxy + 2 API instances), waits for ready, runs k6, prints results:

```bash
make bench
```

Output:
```
p99:4ms score:5800 FP:12 FN:3 ERR:0
```

---

## Submission

Builds the Docker image, pushes to ghcr.io, creates the `submission` branch, and opens the test issue:

```bash
make submission
```

Requires `docker login ghcr.io` and `gh auth login` beforehand.

---

## Architecture

```
Client → HAProxy :9999
           ├── api1  unix:/run/sock/api1.sock
           └── api2  unix:/run/sock/api2.sock
```

Each instance loads `index/references.bin` (int16, N×14 dims + label) at startup. No shared state between instances.

Pipeline per request:
1. Decode JSON → `FraudRequest`
2. Vectorize to `[14]float32` (normalization + MCC risk)
3. KNN(k=5) over 3M vectors → fraud count
4. `fraud_score = count / 5.0`, `approved = score < 0.6`
