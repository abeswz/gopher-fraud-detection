# gopher-fraud-detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Go fraud detection API using brute-force KNN over a 3M int16-encoded vector index loaded from a binary file at startup.

**Architecture:** Two API instances behind HAProxy share no state; each loads the full 87 MB binary index into memory on startup. Requests are vectorized to 14 float32 dims, then KNN (k=5, L2, brute-force) over the in-memory index determines the fraud score.

**Tech Stack:** Go 1.26 (stdlib only), Python 3 + uv (index builder only), HAProxy, distroless/static Docker image.

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `cmd/api/main.go` | Modify | Load index + resources, set service globals, then serve |
| `internal/dto/fraud.go` | Keep as-is | Request/response types |
| `internal/handler/ready.go` | Keep as-is | Always 200 once server starts (index loaded before Serve) |
| `internal/handler/fraud_score.go` | Keep as-is | Decode JSON → call service → encode response |
| `internal/router/router.go` | Keep as-is | Routes |
| `internal/service/fraud_detection.go` | Modify | Wire vectorizer + index; package-level globals |
| `internal/vectorizer/vectorizer.go` | Create | Load normalization+mcc_risk; Vectorize → [14]float32 |
| `internal/search/index.go` | Create | LoadIndex from binary file → Index struct |
| `internal/search/knn.go` | Create | Index.KNN(query, k) → fraudCount int |
| `internal/vectorizer/vectorizer_test.go` | Create | Unit tests using known examples from DETECTION_RULES.md |
| `internal/search/index_test.go` | Create | Unit tests with synthetic binary data |
| `internal/search/knn_test.go` | Create | Unit tests with synthetic Index |
| `ml/build_index.py` | Create | references.json.gz → index/references.bin |
| `ml/pyproject.toml` | Create | uv project with numpy dep |
| `Makefile` | Create | index, bench, submission targets |
| `Dockerfile` | Modify | Fix broken RUN, switch to distroless, copy resources+index, env vars |
| `.gitignore` | Modify | Add index/ and ml/ |

---

## Task 1: Fix Dockerfile

**Files:**
- Modify: `Dockerfile`

The current Dockerfile has a broken multi-line `RUN` (no backslashes — each line is a separate shell token, not a continued command), uses `FROM scratch` (no SSL certs), doesn't copy `resources/` or `index/`, and lacks env vars for resource paths.

- [ ] **Step 1: Overwrite Dockerfile**

```dockerfile
FROM golang:1.26 AS builder

WORKDIR /app

COPY go.mod ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -trimpath \
    -buildvcs=false \
    -ldflags="-s -w" \
    -o fraud-api \
    ./cmd/api

FROM gcr.io/distroless/static-debian12

COPY --from=builder /app/fraud-api /fraud-api
COPY index/ /app/index/
COPY resources/ /app/resources/

ENV INDEX_PATH=/app/index/references.bin
ENV NORM_PATH=/app/resources/normalization.json
ENV MCC_PATH=/app/resources/mcc_risk.json
ENV GOMAXPROCS=2

ENTRYPOINT ["/fraud-api"]
```

- [ ] **Step 2: Commit**

```bash
git add Dockerfile
git commit -m "fix: correct Dockerfile multi-line RUN, switch to distroless, add resource env vars"
```

---

## Task 2: ML Index Builder

**Files:**
- Create: `ml/pyproject.toml`
- Create: `ml/build_index.py`

Converts `resources/references.json.gz` (pre-vectorized float32 vectors) to `index/references.bin` (compact int16 binary).

Binary format:
- 4 bytes: uint32 little-endian — N (number of records)
- N × 29 bytes: 14 × int16 LE (vector × 10000, rounded) + 1 byte label (0=legit, 1=fraud)
- Sentinel -1.0 in float → -10000 in int16

- [ ] **Step 1: Create `ml/pyproject.toml`**

```toml
[project]
name = "build-index"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create `ml/build_index.py`**

```python
import gzip
import json
import struct
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    src = root / "resources" / "references.json.gz"
    dst = root / "index" / "references.bin"
    dst.parent.mkdir(exist_ok=True)

    with gzip.open(src) as f:
        records = json.load(f)

    n = len(records)

    with open(dst, "wb") as out:
        out.write(struct.pack("<I", n))
        for rec in records:
            vec = rec["vector"]
            label = 1 if rec["label"] == "fraud" else 0
            for v in vec:
                scaled = round(v * 10000)
                out.write(struct.pack("<h", scaled))
            out.write(struct.pack("B", label))

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"{n} vectors written, {size_mb:.1f} MB → {dst}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run and verify**

```bash
cd /path/to/repo
uv run ml/build_index.py
```

Expected output (approximately):
```
3000000 vectors written, 87.4 MB → /path/to/repo/index/references.bin
```

If `uv` is not installed: `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`

- [ ] **Step 4: Commit**

```bash
git add ml/pyproject.toml ml/build_index.py
git commit -m "feat: add ML index builder — references.json.gz to int16 binary"
```

---

## Task 3: Search Index Loader

**Files:**
- Create: `internal/search/index.go`
- Create: `internal/search/index_test.go`

The `Index` struct holds vectors as flat `[]int16` (N×14) and labels as `[]uint8`.  
`LoadIndex` reads the entire file into memory first (avoids 3M syscalls), then parses in-place.

- [ ] **Step 1: Write the failing test**

Create `internal/search/index_test.go`:

```go
package search

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

func TestLoadIndex_Basic(t *testing.T) {
	// Build a 2-vector binary: vector0 all-zeros legit, vector1 all-10000 fraud
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, uint32(2))
	for j := 0; j < 14; j++ {
		binary.Write(buf, binary.LittleEndian, int16(0))
	}
	buf.WriteByte(0) // legit
	for j := 0; j < 14; j++ {
		binary.Write(buf, binary.LittleEndian, int16(10000))
	}
	buf.WriteByte(1) // fraud

	tmp := t.TempDir() + "/test.bin"
	if err := os.WriteFile(tmp, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	idx, err := LoadIndex(tmp)
	if err != nil {
		t.Fatalf("LoadIndex error: %v", err)
	}
	if idx.N != 2 {
		t.Errorf("N: got %d, want 2", idx.N)
	}
	if idx.Labels[0] != 0 {
		t.Errorf("label[0]: got %d, want 0 (legit)", idx.Labels[0])
	}
	if idx.Labels[1] != 1 {
		t.Errorf("label[1]: got %d, want 1 (fraud)", idx.Labels[1])
	}
	// vector 1 (offset 14) dim 0 should be 10000
	if idx.Vectors[14] != 10000 {
		t.Errorf("Vectors[14]: got %d, want 10000", idx.Vectors[14])
	}
	// vector 0 dim 0 should be 0
	if idx.Vectors[0] != 0 {
		t.Errorf("Vectors[0]: got %d, want 0", idx.Vectors[0])
	}
}

func TestLoadIndex_SentinelMinus1(t *testing.T) {
	// vector with -1 sentinel (int16 = -10000)
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, uint32(1))
	for j := 0; j < 14; j++ {
		if j == 5 || j == 6 {
			binary.Write(buf, binary.LittleEndian, int16(-10000))
		} else {
			binary.Write(buf, binary.LittleEndian, int16(0))
		}
	}
	buf.WriteByte(0)

	tmp := t.TempDir() + "/test.bin"
	os.WriteFile(tmp, buf.Bytes(), 0644)

	idx, err := LoadIndex(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if idx.Vectors[5] != -10000 {
		t.Errorf("Vectors[5]: got %d, want -10000", idx.Vectors[5])
	}
	if idx.Vectors[6] != -10000 {
		t.Errorf("Vectors[6]: got %d, want -10000", idx.Vectors[6])
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /path/to/repo
go test ./internal/search/... -run TestLoadIndex -v
```

Expected: `FAIL` — `undefined: LoadIndex` (or similar compile error)

- [ ] **Step 3: Create `internal/search/index.go`**

```go
package search

import (
	"encoding/binary"
	"fmt"
	"os"
)

type Index struct {
	Vectors []int16
	Labels  []uint8
	N       int
}

func LoadIndex(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("index file too small: %d bytes", len(data))
	}

	n := int(binary.LittleEndian.Uint32(data[0:4]))

	const recordSize = 14*2 + 1
	expected := 4 + n*recordSize
	if len(data) != expected {
		return nil, fmt.Errorf("index size mismatch: got %d bytes, want %d", len(data), expected)
	}

	vectors := make([]int16, n*14)
	labels := make([]uint8, n)

	offset := 4
	for i := 0; i < n; i++ {
		for j := 0; j < 14; j++ {
			vectors[i*14+j] = int16(binary.LittleEndian.Uint16(data[offset : offset+2]))
			offset += 2
		}
		labels[i] = data[offset]
		offset++
	}

	return &Index{Vectors: vectors, Labels: labels, N: n}, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
go test ./internal/search/... -run TestLoadIndex -v
```

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add internal/search/index.go internal/search/index_test.go
git commit -m "feat: add search index loader with int16 binary format"
```

---

## Task 4: KNN Search

**Files:**
- Create: `internal/search/knn.go`
- Create: `internal/search/knn_test.go`

Brute-force L2 KNN, k=5. Maintains a max-heap of size k so the farthest current neighbor is O(1) to find and replace. For k=5, a linear scan of the 5-element slice is used instead of a heap (simpler, same asymptotic for small k).

At query time, int16 vectors are converted back to float32: `float32(v) / 10000.0`. The sentinel -10000 becomes -1.0 in float, which matches the query's -1.0 for null last_transaction — correct behavior.

- [ ] **Step 1: Write the failing test**

Create `internal/search/knn_test.go`:

```go
package search

import (
	"testing"
)

func makeTestIndex(nFraud, nLegit int, fraudVal, legitVal int16) *Index {
	n := nFraud + nLegit
	idx := &Index{
		N:       n,
		Vectors: make([]int16, n*14),
		Labels:  make([]uint8, n),
	}
	for i := 0; i < nFraud; i++ {
		for j := 0; j < 14; j++ {
			idx.Vectors[i*14+j] = fraudVal
		}
		idx.Labels[i] = 1
	}
	for i := nFraud; i < n; i++ {
		for j := 0; j < 14; j++ {
			idx.Vectors[i*14+j] = legitVal
		}
		idx.Labels[i] = 0
	}
	return idx
}

func TestKNN_AllFraud(t *testing.T) {
	// 5 fraud vectors at 1.0 (int16=10000), 5 legit at 0.0
	// Query at 1.0 → nearest 5 are all fraud
	idx := makeTestIndex(5, 5, 10000, 0)
	query := [14]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	got := idx.KNN(query, 5)
	if got != 5 {
		t.Errorf("AllFraud: got %d fraud, want 5", got)
	}
}

func TestKNN_AllLegit(t *testing.T) {
	// 5 fraud at 1.0, 5 legit at 0.0
	// Query at 0.0 → nearest 5 are all legit
	idx := makeTestIndex(5, 5, 10000, 0)
	query := [14]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	got := idx.KNN(query, 5)
	if got != 0 {
		t.Errorf("AllLegit: got %d fraud, want 0", got)
	}
}

func TestKNN_Mixed(t *testing.T) {
	// 3 fraud at 1.0, 7 legit at 0.0
	// Query at 0.9 → nearest 5 are 3 fraud + 2 legit
	idx := makeTestIndex(3, 7, 10000, 0)
	query := [14]float32{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}
	got := idx.KNN(query, 5)
	if got != 3 {
		t.Errorf("Mixed: got %d fraud, want 3", got)
	}
}

func TestKNN_SentinelHandling(t *testing.T) {
	// Both query and reference have -1 at dims 5,6 → distance at those dims = 0
	idx := &Index{
		N:       2,
		Vectors: make([]int16, 2*14),
		Labels:  make([]uint8, 2),
	}
	// Reference 0: -10000 at dims 5,6 (sentinel), 0 elsewhere; legit
	idx.Vectors[5] = -10000
	idx.Vectors[6] = -10000
	// Reference 1: 10000 everywhere; fraud
	for j := 0; j < 14; j++ {
		idx.Vectors[14+j] = 10000
	}
	idx.Labels[1] = 1

	query := [14]float32{0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0}
	got := idx.KNN(query, 2)
	if got != 0 {
		t.Errorf("SentinelHandling: got %d fraud, want 0 (legit is nearest)", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/search/... -run TestKNN -v
```

Expected: `FAIL` — `idx.KNN undefined`

- [ ] **Step 3: Create `internal/search/knn.go`**

```go
package search

func (idx *Index) KNN(query [14]float32, k int) int {
	type entry struct {
		dist  float32
		label uint8
	}

	top := make([]entry, 0, k)
	maxDist := float32(0)
	maxPos := 0

	for i := 0; i < idx.N; i++ {
		base := i * 14
		var dist float32
		for j := 0; j < 14; j++ {
			ref := float32(idx.Vectors[base+j]) / 10000.0
			d := query[j] - ref
			dist += d * d
		}

		if len(top) < k {
			top = append(top, entry{dist, idx.Labels[i]})
			if len(top) == k {
				maxDist, maxPos = findMax(top)
			}
		} else if dist < maxDist {
			top[maxPos] = entry{dist, idx.Labels[i]}
			maxDist, maxPos = findMax(top)
		}
	}

	fraudCount := 0
	for _, e := range top {
		if e.label == 1 {
			fraudCount++
		}
	}
	return fraudCount
}

func findMax(entries []entry) (maxDist float32, maxPos int) {
	maxDist = entries[0].dist
	maxPos = 0
	for i := 1; i < len(entries); i++ {
		if entries[i].dist > maxDist {
			maxDist = entries[i].dist
			maxPos = i
		}
	}
	return
}

type entry = struct {
	dist  float32
	label uint8
}
```

Wait — `entry` is defined twice (in the function and as a package-level type alias). Fix: define `entry` at package level, use it in both `KNN` and `findMax`.

Correct version of `internal/search/knn.go`:

```go
package search

type knnEntry struct {
	dist  float32
	label uint8
}

func (idx *Index) KNN(query [14]float32, k int) int {
	top := make([]knnEntry, 0, k)
	maxDist := float32(0)
	maxPos := 0

	for i := 0; i < idx.N; i++ {
		base := i * 14
		var dist float32
		for j := 0; j < 14; j++ {
			ref := float32(idx.Vectors[base+j]) / 10000.0
			d := query[j] - ref
			dist += d * d
		}

		if len(top) < k {
			top = append(top, knnEntry{dist, idx.Labels[i]})
			if len(top) == k {
				maxDist, maxPos = knnFindMax(top)
			}
		} else if dist < maxDist {
			top[maxPos] = knnEntry{dist, idx.Labels[i]}
			maxDist, maxPos = knnFindMax(top)
		}
	}

	fraudCount := 0
	for _, e := range top {
		if e.label == 1 {
			fraudCount++
		}
	}
	return fraudCount
}

func knnFindMax(entries []knnEntry) (maxDist float32, maxPos int) {
	maxDist = entries[0].dist
	maxPos = 0
	for i := 1; i < len(entries); i++ {
		if entries[i].dist > maxDist {
			maxDist = entries[i].dist
			maxPos = i
		}
	}
	return
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
go test ./internal/search/... -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add internal/search/knn.go internal/search/knn_test.go
git commit -m "feat: add brute-force L2 KNN over int16 index"
```

---

## Task 5: Vectorizer

**Files:**
- Create: `internal/vectorizer/vectorizer.go`
- Create: `internal/vectorizer/vectorizer_test.go`

Implements the 14-dim transformation from `references/rules/DETECTION_RULES.md`.

Key notes:
- `time.Weekday()` returns 0=Sunday … 6=Saturday. Spec is Mon=0 … Sun=6. Conversion: `(int(wd) + 6) % 7`.
- `clamp(x)` keeps x in [0.0, 1.0]. Only indices 5 and 6 can be -1 (null last_transaction).
- MCC risk default is 0.5 for unknown codes.
- `known_merchants` is a slice that may contain duplicates — build a set for O(1) lookup.

- [ ] **Step 1: Write the failing test**

Create `internal/vectorizer/vectorizer_test.go`:

```go
package vectorizer

import (
	"math"
	"testing"

	"gopher-fraud-detection/internal/dto"
)

var testNorm = Normalization{
	MaxAmount:            10000,
	MaxInstallments:      12,
	AmountVsAvgRatio:     10,
	MaxMinutes:           1440,
	MaxKm:                1000,
	MaxTxCount24h:        20,
	MaxMerchantAvgAmount: 10000,
}

var testMcc = map[string]float32{
	"5411": 0.15,
	"7802": 0.75,
}

func approxEqual(a, b float32, tol float64) bool {
	return math.Abs(float64(a-b)) <= tol
}

func checkVec(t *testing.T, got [14]float32, want [14]float32) {
	t.Helper()
	for i := range want {
		if !approxEqual(got[i], want[i], 1e-3) {
			t.Errorf("dim[%d]: got %.4f, want %.4f", i, got[i], want[i])
		}
	}
}

// Example 1 from DETECTION_RULES.md — legit, last_transaction null
// Expected vector: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
func TestVectorize_LegitNullLastTx(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{
			Amount:      41.12,
			Installments: 2,
			RequestedAt: "2026-03-11T18:45:53Z",
		},
		Customer: dto.Customer{
			AvgAmount:      82.24,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: dto.Merchant{ID: "MERC-016", MCC: "5411", AvgAmount: 60.25},
		Terminal: dto.Terminal{IsOnline: false, CardPresent: true, KmFromHome: 29.2331036248},
		LastTx:   nil,
	}
	want := [14]float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	checkVec(t, v.Vectorize(req), want)
}

// Example 2 from DETECTION_RULES.md — fraud, last_transaction null
// Expected vector: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
func TestVectorize_FraudNullLastTx(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{
			Amount:      9505.97,
			Installments: 10,
			RequestedAt: "2026-03-14T05:15:12Z",
		},
		Customer: dto.Customer{
			AvgAmount:      81.28,
			TxCount24h:     20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: dto.Merchant{ID: "MERC-068", MCC: "7802", AvgAmount: 54.86},
		Terminal: dto.Terminal{IsOnline: false, CardPresent: true, KmFromHome: 952.2745933273},
		LastTx:   nil,
	}
	want := [14]float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	checkVec(t, v.Vectorize(req), want)
}

// Known merchant → unknown_merchant = 0
func TestVectorize_KnownMerchant(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 100, Installments: 1, RequestedAt: "2026-01-01T12:00:00Z"},
		Customer:    dto.Customer{AvgAmount: 100, TxCount24h: 1, KnownMerchants: []string{"MERC-A", "MERC-B"}},
		Merchant:    dto.Merchant{ID: "MERC-A", MCC: "5411", AvgAmount: 100},
		Terminal:    dto.Terminal{},
		LastTx:      nil,
	}
	got := v.Vectorize(req)
	if got[11] != 0 {
		t.Errorf("known merchant: dim[11] got %.1f, want 0", got[11])
	}
}

// Unknown merchant → unknown_merchant = 1
func TestVectorize_UnknownMerchant(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 100, Installments: 1, RequestedAt: "2026-01-01T12:00:00Z"},
		Customer:    dto.Customer{AvgAmount: 100, TxCount24h: 1, KnownMerchants: []string{"MERC-A", "MERC-B"}},
		Merchant:    dto.Merchant{ID: "MERC-Z", MCC: "9999", AvgAmount: 100},
		Terminal:    dto.Terminal{},
		LastTx:      nil,
	}
	got := v.Vectorize(req)
	if got[11] != 1 {
		t.Errorf("unknown merchant: dim[11] got %.1f, want 1", got[11])
	}
	if got[12] != 0.5 {
		t.Errorf("unknown mcc default: dim[12] got %.1f, want 0.5", got[12])
	}
}

// last_transaction present → dims 5,6 computed
func TestVectorize_WithLastTx(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{
			Amount:      384.88,
			Installments: 3,
			RequestedAt: "2026-03-11T20:23:35Z",
		},
		Customer: dto.Customer{
			AvgAmount:      769.76,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-009", "MERC-009", "MERC-001", "MERC-001"},
		},
		Merchant: dto.Merchant{ID: "MERC-001", MCC: "5912", AvgAmount: 298.95},
		Terminal: dto.Terminal{IsOnline: false, CardPresent: true, KmFromHome: 13.7090520965},
		LastTx: &dto.LastTransaction{
			Timestamp:     "2026-03-11T14:58:35Z",
			KmFromCurrent: 18.8626479774,
		},
	}
	got := v.Vectorize(req)
	// minutes between 14:58:35 and 20:23:35 = 5h25m = 325 minutes
	// clamp(325/1440) = 0.2257
	if got[5] < 0.224 || got[5] > 0.228 {
		t.Errorf("minutes_since_last_tx: got %.4f, want ~0.2257", got[5])
	}
	// clamp(18.8626/1000) = 0.0189
	if got[6] < 0.018 || got[6] > 0.020 {
		t.Errorf("km_from_last_tx: got %.4f, want ~0.0189", got[6])
	}
	// dims 5,6 must not be -1
	if got[5] < 0 || got[6] < 0 {
		t.Errorf("dims 5,6 should not be -1 when last_tx present")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/vectorizer/... -v
```

Expected: `FAIL` — compile error (package doesn't exist yet)

- [ ] **Step 3: Create `internal/vectorizer/vectorizer.go`**

```go
package vectorizer

import (
	"encoding/json"
	"os"
	"time"

	"gopher-fraud-detection/internal/dto"
)

type Normalization struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKm                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

type Vectorizer struct {
	Norm    Normalization
	MccRisk map[string]float32
}

func Load(normPath, mccPath string) (*Vectorizer, error) {
	normData, err := os.ReadFile(normPath)
	if err != nil {
		return nil, err
	}
	var norm Normalization
	if err := json.Unmarshal(normData, &norm); err != nil {
		return nil, err
	}

	mccData, err := os.ReadFile(mccPath)
	if err != nil {
		return nil, err
	}
	var mccRisk map[string]float32
	if err := json.Unmarshal(mccData, &mccRisk); err != nil {
		return nil, err
	}

	return &Vectorizer{Norm: norm, MccRisk: mccRisk}, nil
}

func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func (v *Vectorizer) Vectorize(req dto.FraudRequest) [14]float32 {
	var vec [14]float32
	n := v.Norm

	vec[0] = clamp(float32(req.Transaction.Amount) / n.MaxAmount)
	vec[1] = clamp(float32(req.Transaction.Installments) / n.MaxInstallments)
	vec[2] = clamp((float32(req.Transaction.Amount) / float32(req.Customer.AvgAmount)) / n.AmountVsAvgRatio)

	t, _ := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	t = t.UTC()
	vec[3] = float32(t.Hour()) / 23.0
	wd := int(t.Weekday())
	vec[4] = float32((wd+6)%7) / 6.0

	if req.LastTx == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		lastT, _ := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		minutes := float32(t.Sub(lastT).Minutes())
		vec[5] = clamp(minutes / n.MaxMinutes)
		vec[6] = clamp(float32(req.LastTx.KmFromCurrent) / n.MaxKm)
	}

	vec[7] = clamp(float32(req.Terminal.KmFromHome) / n.MaxKm)
	vec[8] = clamp(float32(req.Customer.TxCount24h) / n.MaxTxCount24h)

	if req.Terminal.IsOnline {
		vec[9] = 1
	}
	if req.Terminal.CardPresent {
		vec[10] = 1
	}

	known := make(map[string]struct{}, len(req.Customer.KnownMerchants))
	for _, m := range req.Customer.KnownMerchants {
		known[m] = struct{}{}
	}
	if _, ok := known[req.Merchant.ID]; !ok {
		vec[11] = 1
	}

	if risk, ok := v.MccRisk[req.Merchant.MCC]; ok {
		vec[12] = risk
	} else {
		vec[12] = 0.5
	}

	vec[13] = clamp(float32(req.Merchant.AvgAmount) / n.MaxMerchantAvgAmount)

	return vec
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
go test ./internal/vectorizer/... -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add internal/vectorizer/vectorizer.go internal/vectorizer/vectorizer_test.go
git commit -m "feat: add 14-dim vectorizer with normalization and MCC risk lookup"
```

---

## Task 6: Wire Service

**Files:**
- Modify: `internal/service/fraud_detection.go`

Replace the stub with real vectorizer + KNN calls. The index and vectorizer are set as package-level pointers by `main.go` at startup (set before `Serve` is called, so no race condition).

- [ ] **Step 1: Update `internal/service/fraud_detection.go`**

```go
package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	Idx  *search.Index
	Vec  *vectorizer.Vectorizer
)

func CalculateFraudScore(req dto.FraudRequest) dto.FraudResponse {
	vec := Vec.Vectorize(req)
	fraudCount := Idx.KNN(vec, 5)
	fraudScore := float64(fraudCount) / 5.0
	return dto.FraudResponse{
		Approved:   fraudScore < 0.6,
		FraudScore: fraudScore,
	}
}
```

- [ ] **Step 2: Verify it compiles**

```bash
go build ./...
```

Expected: success (no output). If it fails, check that all import paths and type names match.

- [ ] **Step 3: Commit**

```bash
git add internal/service/fraud_detection.go
git commit -m "feat: wire service to real vectorizer and KNN index"
```

---

## Task 7: Wire Main

**Files:**
- Modify: `cmd/api/main.go`

Load index and resources at startup (fatal on error). Set `service.Idx` and `service.Vec` before serving. Read paths from env vars with defaults for local dev.

The index loading is synchronous — the HTTP server only starts after everything is loaded. This means `GET /ready` always returns 200 once the server is up (the current `ready.go` handler is correct as-is).

- [ ] **Step 1: Update `cmd/api/main.go`**

```go
package main

import (
	"gopher-fraud-detection/internal/router"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/service"
	"gopher-fraud-detection/internal/vectorizer"
	"log"
	"net"
	"net/http"
	"os"
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	indexPath := envOr("INDEX_PATH", "index/references.bin")
	normPath := envOr("NORM_PATH", "resources/normalization.json")
	mccPath := envOr("MCC_PATH", "resources/mcc_risk.json")

	vec, err := vectorizer.Load(normPath, mccPath)
	if err != nil {
		log.Fatalf("load vectorizer: %v", err)
	}

	idx, err := search.LoadIndex(indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}

	service.Vec = vec
	service.Idx = idx

	log.Printf("loaded %d vectors", idx.N)

	sock := envOr("SOCK", "")
	if sock == "" {
		log.Fatal("SOCK environment variable is required")
	}

	_ = os.Remove(sock)

	listener, err := net.Listen("unix", sock)
	if err != nil {
		log.Fatal(err)
	}

	server := &http.Server{Handler: router.New()}
	log.Fatal(server.Serve(listener))
}
```

- [ ] **Step 2: Build the binary**

```bash
go build ./cmd/api
```

Expected: produces `fraud-api` binary (or `./fraud-api`), no errors.

- [ ] **Step 3: Commit**

```bash
git add cmd/api/main.go
git commit -m "feat: load index and resources at startup, wire service globals"
```

---

## Task 8: Makefile and .gitignore

**Files:**
- Create: `Makefile`
- Modify: `.gitignore`

- [ ] **Step 1: Update `.gitignore`**

Append to the existing `.gitignore`:

```
index/
ml/
```

- [ ] **Step 2: Create `Makefile`**

```makefile
IMAGE_REPO     := ghcr.io/abeswz/gopher-fraud-detection
GIT_SHA        := $(shell git rev-parse --short HEAD)
IMAGE          := $(IMAGE_REPO):$(GIT_SHA)
PORT           := 9999
READY_TIMEOUT  := 300
PARTICIPANT    := abeswz-gopher
RINHA_REPO     := zanfranceschi/rinha-de-backend-2026

.PHONY: index bench submission

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

- [ ] **Step 3: Verify full build pipeline**

```bash
# 1. Build index (requires references.json.gz in resources/)
make index
# Expected: "3000000 vectors written, ~87 MB → .../index/references.bin"

# 2. Check that Go compiles cleanly
go build ./...
# Expected: no output

# 3. Run all Go tests
go test ./...
# Expected: all PASS
```

- [ ] **Step 4: Commit**

```bash
git add Makefile .gitignore
git commit -m "chore: add Makefile (index/bench/submission) and gitignore index/ ml/"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Covered by |
|-----------------|-----------|
| Binary format: uint32 N + N×(14×int16 + 1 byte label) | Task 2 (build_index.py) + Task 3 (LoadIndex) |
| int16 encoding: float×10000, sentinel -1→-10000 | Task 2 + Task 4 (KNN conversion at query time) |
| 87 MB memory budget | Binary format design (14×2+1=29 bytes/record × 3M = 87MB) |
| Vectorize 14 dims per DETECTION_RULES.md | Task 5 |
| day_of_week Mon=0 Sun=6 (vs Go's Sun=0) | Task 5: `(int(wd)+6)%7` |
| KNN brute-force L2 k=5 | Task 4 |
| fraud_score = fraudCount/5.0, approved = score < 0.6 | Task 6 |
| /ready returns 200 once loaded | Handled by sync startup in Task 7 (existing handler correct) |
| Malformed JSON → 400 | Existing handler (no change needed) |
| Index load failure → fatal | Task 7 (`log.Fatalf`) |
| ENV vars: INDEX_PATH, NORM_PATH, MCC_PATH | Task 1 (Dockerfile) + Task 7 (main.go) |
| GOMAXPROCS=2 | Task 1 (Dockerfile) |
| make index, make bench, make submission | Task 8 |
| index/ and ml/ gitignored | Task 8 |
| Unit tests: vectorizer, KNN, index | Tasks 3, 4, 5 |

### Placeholder scan

No TBDs, no "similar to task N", no "add error handling" without showing how, no forward references to undefined types.

### Type consistency

- `*search.Index` — defined in Task 3, used in Tasks 4, 6, 7 ✓
- `*vectorizer.Vectorizer` — defined in Task 5, used in Tasks 6, 7 ✓
- `service.Idx` and `service.Vec` — defined in Task 6, set in Task 7 ✓
- `[14]float32` — KNN query type in Task 4, Vectorize return type in Task 5, consistent ✓

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-30-gopher-fraud-detection-impl.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration
Use superpowers:subagent-driven-development

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints
Use superpowers:executing-plans

**Which approach?**
