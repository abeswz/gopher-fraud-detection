# Design: AVX2 Distance + Raw Feature Tree

**Date:** 2026-05-31
**Status:** Approved
**Goal:** Remote p99 < 1ms, maintain or improve detection quality (FP/FN ≤ current baseline).

---

## Problem

Remote p99 = 30ms (local = 0.42ms, 71× gap). Root cause: weak remote CPU under CFS quota (0.45 CPU). Nearly all requests reach IVF k-NN, which is the expensive path (~40µs locally, ~2ms remotely under load). GOMAXPROCS=1 already set; no further free wins from scheduler tuning.

Two levers:
1. **Fewer IVF calls** — route most requests through a fast tree before vectorizing
2. **Faster IVF calls** — AVX2 SIMD on the distance computation for requests that do reach IVF

---

## Architecture

### Current pipeline

```
JSON → fast_path → vectorize → decision_tree(8 leaves) → IVF k-NN → response
```

### New pipeline

```
JSON → fast_path → raw_tree(21 features) → vectorize → IVF k-NN(AVX2) → response
```

Each stage returns `(fraudCount int, hit bool)`. `hit=true` short-circuits — later stages are skipped.

### Expected coverage per layer

| Layer | Coverage | Cost |
|---|---|---|
| fast_path | ~79% (52.7% legit + 26.4% risky) | ~0 µs |
| raw_tree | ~15–18% of total | ~1–5 µs |
| IVF k-NN | ~3–6% of total | ~15 µs (AVX2) |

With IVF handling only ~5% of requests, remote p99 is dominated by tree traversal, not IVF. Combined with AVX2 speeding up the remaining IVF calls, remote p99 < 1ms is achievable.

---

## Component 1: AVX2 k-NN Distance

### Index format change

Vectors padded from 14 to 16 int16 per entry. Dims 14 and 15 are always 0; they contribute nothing to distance.

```
Before: N × 14 × int16 = 84 MB
After:  N × 16 × int16 = 96 MB  (+12 MB per instance, fits in 168 MB budget)
```

Magic string changes `"IVF1"` → `"IVF2"` so `LoadIVFIndex` detects the format and reads 16-dim layout.

### Query type change

```go
// vectorizer returns [16]float32; positions 14 and 15 = 0.0
// Index interface: KNN(query [16]float32, k int) int
```

### Go assembly

Files:
- `internal/search/knn_amd64.go` — Go stubs with `//go:build amd64`
- `internal/search/knn_amd64.s` — plan9 AMD64 AVX2 implementation
- `internal/search/knn_generic.go` — `//go:build !amd64` scalar fallback (existing logic)

Core function: `distL2i16_16(vecs []int16, base int, q *[16]float32) float32`

Algorithm (two clean YMM passes):
```
Y6 = broadcast(1.0/10000.0)

; dims 0–7
VPMOVSXWD (SI), Y0      ; 8 × i16 → i32
VCVTDQ2PS Y0, Y0        ; → f32
VMULPS Y6, Y0, Y0       ; scale
VMOVUPS (DI), Y7        ; q[0..7]
VSUBPS Y7, Y0, Y0       ; diff
VMULPS Y0, Y0, Y0       ; d²  →  Y0

; dims 8–15 (same pattern)  →  Y1

VADDPS Y1, Y0, Y0       ; sum both YMM

; horizontal reduction Y0 → scalar
VEXTRACTF128 $1, Y0, X1
VADDPS X1, X0, X0       ; 4 floats
VHADDPS X0, X0, X0      ; 2 floats
VHADDPS X0, X0, X0      ; 1 float → return
VZEROALL
RET
```

Early-exit logic (check at dim 0, dim 7) stays in Go. After heap is full (k=5), caller compares partial accumulated distance against `maxDist` before calling `distL2i16_16`. If partial already exceeds `maxDist`, skip the full call.

### Expected speedup

| Config | µs/op serial | p99 local |
|---|---|---|
| Current scalar | ~40 µs | 0.42 ms |
| AVX2 16-dim | ~12–15 µs (est.) | ~0.15 ms (est.) |

---

## Component 2: Raw Feature Tree

### Features (21 total)

Extracted directly from `dto.FraudRequest` — no normalization required.

| # | Feature | Source |
|---|---|---|
| 0 | amount | transaction.amount |
| 1 | amount / customer.avg_amount | ratio |
| 2 | installments | transaction.installments |
| 3 | tx_count_24h | customer.tx_count_24h |
| 4 | km_from_home | terminal.km_from_home |
| 5 | is_known_merchant | 1 if merchant.id in customer.known_merchants |
| 6 | mcc_risk_score | from mcc_risk.json (default 0.5) |
| 7 | merchant.avg_amount | merchant.avg_amount |
| 8 | amount / merchant.avg_amount | ratio |
| 9 | hour_of_day | parsed from transaction.requested_at (UTC) |
| 10 | is_online | terminal.is_online |
| 11 | card_present | terminal.card_present |
| 12 | last_km_from_current | last_transaction.km_from_current (-1 if null) |
| 13 | last_time_delta_sec | seconds between last_transaction and requested_at (-1 if null) |
| 14 | amount_over_max | 1 if amount > 10000, else 0 |
| 15 | installments_normalized | installments / 12.0 |
| 16 | tx_velocity | tx_count_24h / 24.0 |
| 17 | is_safe_mcc | 1 if mcc in {5411, 5812, 5912, 5311} |
| 18 | is_risky_mcc | 1 if mcc in {7995, 7801, 7802} |
| 19 | amount_normalized | clamp(amount / 10000.0) |
| 20 | customer_avg_normalized | clamp(avg_amount / 5000.0) |

Features 12 and 13 use `-1` as sentinel for null `last_transaction`. Tree learns to handle this.

Division-by-zero: features 1 and 8 use ratios. If denominator is 0 (new customer with avg_amount=0, or merchant with avg_amount=0), use `0.0` for the ratio — not NaN. Tree splits on concrete floats; NaN would cause undefined traversal.

### Training

Script: `ml/train_raw_tree.py`

```
1. Load resources/references.json.gz (3M entries)
2. Extract 21 features per entry using same logic as ExtractRawFeatures
3. Labels: k-NN ground truth (already in dataset label field)
4. Train: DecisionTreeClassifier(max_depth=None, min_samples_leaf=50)
5. Prune: retain leaves where confidence >= 0.85
6. Save: ml/models/raw_tree.pkl
7. Report: coverage %, FP/FN on held-out validation split
```

Threshold 0.85 is the starting point. Adjust after measuring coverage vs FP/FN tradeoff on validation set.

### Code generation

Two files — boundary between generated and handwritten:

**`internal/service/raw_features.go`** — handwritten, stable:
```go
type RawFeatures struct { /* 21 float32 fields */ }

// ExtractRawFeatures — zero allocation, runs before vectorizer.
// Ratio features (1, 8) return 0.0 when denominator is 0.
func ExtractRawFeatures(req dto.FraudRequest, mccRisk map[string]float32) RawFeatures
```

**`internal/service/raw_tree.go`** — generated by `ml/gen_raw_tree_go.py`, same pattern as `decision_tree.go`:
```go
// RawTreePredict returns (fraudCount, hit).
// hit=true only when leaf confidence >= 0.85.
func RawTreePredict(f RawFeatures) (int, bool)
```

`gen_raw_tree_go.py` reads `ml/models/raw_tree.pkl` and emits the tree as a Go if/else chain. `raw_features.go` is never touched by the generator.

### Integration in fraud_detection.go

```go
func (s *Service) Score(req dto.FraudRequest) dto.FraudResponse {
    if count, ok := fastPath(req); ok {
        return respond(count)
    }
    if count, ok := RawTreePredict(ExtractRawFeatures(req, s.mccRisk)); ok {
        return respond(count)
    }
    vec := s.vectorizer.Vectorize(req)   // only ~5% of requests reach here
    count := s.index.KNN(vec, 5)        // AVX2
    return respond(count)
}
```

---

## Testing Strategy

### Unit tests

| File | What it tests |
|---|---|
| `internal/search/knn_amd64_test.go` | `distL2i16_16` vs scalar for 100+ random vectors; padding correctness; !amd64 fallback compiles |
| `internal/search/index_test.go` | `LoadIVFIndex` with IVF2 magic; 16-dim distances equal 14-dim distances |
| `internal/service/raw_tree_test.go` | `ExtractRawFeatures` values; `RawTreePredict` vs IVF ground truth on 1000-sample set; FP/FN delta |

### Benchmarks

```
BenchmarkKNNScalar    — existing baseline
BenchmarkKNNAVX2      — new implementation
BenchmarkFullPipeline — fast_path + raw_tree + IVF combined
```

Target: `BenchmarkKNNAVX2` < 15 µs/op.

### Acceptance criteria

| Metric | Minimum | Target |
|---|---|---|
| FP delta vs current | ≤ +2 | 0 |
| FN delta vs current | 0 | 0 |
| p99 local | ≤ 0.42 ms | ≤ 0.20 ms |
| `go test ./...` | all pass | all pass |
| AVX2 bench speedup | ≥ 2× | ≥ 3× |
| Remote p99 (after submission) | < 1.4 ms | < 1.0 ms |

---

## Implementation Order

Each step is a separate commit. Validates independently before the next step.

```
1. Pad index to 16 dims (build_index.py + LoadIVFIndex + [16]float32 type)
   → go test ./... passes, existing bench unchanged
2. AVX2 Go assembly (distL2i16_16 + knn_amd64.s + generic fallback)
   → BenchmarkKNNAVX2 ≥ 2× faster, all tests pass
3. Train raw_tree (ml/train_raw_tree.py + ml/gen_raw_tree_go.py)
   → validate coverage and FP/FN on training data
4. Wire raw_tree into pipeline (fraud_detection.go)
   → go test ./..., make bench — confirm p99 local improves
5. make submission → validate remote
```

---

## Constraints (unchanged)

- k=5 mandatory (spec)
- threshold=0.6 mandatory (spec)
- fraud_score = fraudCount / 5.0 (spec)
- CGO_ENABLED=0 (Dockerfile) — AVX2 via Go assembly only
- 168 MB RAM per instance — 16-dim index at 96 MB fits
- GOMAXPROCS=1 already set in Dockerfile
