# Hybrid Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-path 100%-k-NN pipeline with 3-tier hybrid: fast_path rules → decision_tree → k-NN, plus shared mmap to halve index RAM usage from ~174 MB to ~87 MB.

**Architecture:** Fast path operates on raw request fields (no vectorization, ~0μs) and exits early for ~79% of obviously safe/risky transactions. Decision tree operates on the normalized 14-dim vector and handles the gray area. k-NN is the exact fallback. Both instances mmap the same index file (OS shares physical pages → ~87 MB instead of 174 MB).

**Tech Stack:** Go stdlib (`syscall.Mmap`, `unsafe`), sklearn DecisionTreeClassifier (offline training), Python code generator emitting Go arrays, uv.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `internal/service/fast_path.go` | Create | Fast path rules: raw request → (count, ok) |
| `internal/service/fast_path_test.go` | Create | Tests for fast path |
| `internal/search/index.go` | Modify | Replace os.ReadFile with mmap; zero-copy vector/label slices via unsafe |
| `internal/search/decision_tree.go` | Create (generated) | Decision tree arrays + iterative traversal function |
| `ml/train_decision_tree.py` | Create | Read references.json.gz → train sklearn tree → save model JSON |
| `ml/gen_tree_go.py` | Create | Read model JSON → emit internal/search/decision_tree.go |
| `internal/service/fraud_detection.go` | Modify | Wire: fast_path → vectorize → decision_tree → k-NN |

---

## Task 1: Fast Path Rules

**Files:**
- Create: `internal/service/fast_path.go`
- Create: `internal/service/fast_path_test.go`

Safe spend (ALL conditions → count=0, ok=true):
- `transaction.amount ≤ 500`
- `transaction.amount ≤ 0.5 × customer.avg_amount`
- `transaction.installments ≤ 3`
- `customer.tx_count_24h ≤ 5`
- merchant is known (`merchant.id` ∈ `customer.known_merchants`)
- `terminal.km_from_home ≤ 50`
- `merchant.mcc` ∈ {5411, 5812, 5912, 5311}

Risky spend (ALL conditions → count=5, ok=true):
- `transaction.amount ≥ 5000`
- `transaction.installments ≥ 5`
- `customer.tx_count_24h ≥ 6`
- merchant is unknown
- `terminal.km_from_home ≥ 150`
- `merchant.mcc` ∈ {7995, 7801, 7802}

- [ ] **Step 1.1: Write failing tests**

```go
// internal/service/fast_path_test.go
package service

import (
	"testing"

	"gopher-fraud-detection/internal/dto"
)

func TestFastPath_SafeSpend(t *testing.T) {
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 80, Installments: 2},
		Customer:    dto.Customer{AvgAmount: 200, TxCount24h: 3, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-01", MCC: "5411"},
		Terminal:    dto.Terminal{KmFromHome: 10},
	}
	count, ok := fastPath(req)
	if !ok {
		t.Fatal("expected fast path hit")
	}
	if count != 0 {
		t.Errorf("safe spend: got count=%d, want 0", count)
	}
}

func TestFastPath_RiskySpend(t *testing.T) {
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 8000, Installments: 7},
		Customer:    dto.Customer{AvgAmount: 100, TxCount24h: 10, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-99", MCC: "7995"},
		Terminal:    dto.Terminal{KmFromHome: 300},
	}
	count, ok := fastPath(req)
	if !ok {
		t.Fatal("expected fast path hit")
	}
	if count != 5 {
		t.Errorf("risky spend: got count=%d, want 5", count)
	}
}

func TestFastPath_SafeMissOneCondition(t *testing.T) {
	// safe in all ways except amount is 600 (> 500)
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 600, Installments: 2},
		Customer:    dto.Customer{AvgAmount: 2000, TxCount24h: 3, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-01", MCC: "5411"},
		Terminal:    dto.Terminal{KmFromHome: 10},
	}
	_, ok := fastPath(req)
	if ok {
		t.Fatal("should not hit fast path when amount > 500")
	}
}

func TestFastPath_RiskyMissOneCondition(t *testing.T) {
	// risky in all ways except installments is 4 (< 5)
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 8000, Installments: 4},
		Customer:    dto.Customer{AvgAmount: 100, TxCount24h: 10, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-99", MCC: "7995"},
		Terminal:    dto.Terminal{KmFromHome: 300},
	}
	_, ok := fastPath(req)
	if ok {
		t.Fatal("should not hit fast path when installments < 5")
	}
}

func TestFastPath_SafeHighAmountVsAvg(t *testing.T) {
	// amount=400 but > 50% of avg_amount=600 (400 > 300)
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 400, Installments: 2},
		Customer:    dto.Customer{AvgAmount: 600, TxCount24h: 3, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-01", MCC: "5411"},
		Terminal:    dto.Terminal{KmFromHome: 10},
	}
	_, ok := fastPath(req)
	if ok {
		t.Fatal("should not hit safe path when amount > 50%% of avg_amount")
	}
}
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
cd /home/snow/workspace/rinha-backend/gopher-fraud-detection
go test ./internal/service/ -run TestFastPath -v 2>&1 | head -20
```
Expected: compile error (fastPath undefined).

- [ ] **Step 1.3: Implement fast_path.go**

```go
// internal/service/fast_path.go
package service

import "gopher-fraud-detection/internal/dto"

var safeMCCs = map[string]struct{}{
	"5411": {}, "5812": {}, "5912": {}, "5311": {},
}

var riskyMCCs = map[string]struct{}{
	"7995": {}, "7801": {}, "7802": {},
}

// fastPath checks deterministic rules before vectorization and k-NN.
// Returns (fraudCount, true) when the result is certain; (0, false) otherwise.
// Safe: ALL conditions met → 0 fraud neighbors.
// Risky: ALL conditions met → 5 fraud neighbors.
func fastPath(req dto.FraudRequest) (int, bool) {
	tx := req.Transaction
	cust := req.Customer
	merch := req.Merchant
	term := req.Terminal

	isKnown := false
	for _, m := range cust.KnownMerchants {
		if m == merch.ID {
			isKnown = true
			break
		}
	}

	_, isSafe := safeMCCs[merch.MCC]
	if tx.Amount <= 500 &&
		tx.Amount <= 0.5*cust.AvgAmount &&
		tx.Installments <= 3 &&
		cust.TxCount24h <= 5 &&
		isKnown &&
		term.KmFromHome <= 50 &&
		isSafe {
		return 0, true
	}

	_, isRisky := riskyMCCs[merch.MCC]
	if tx.Amount >= 5000 &&
		tx.Installments >= 5 &&
		cust.TxCount24h >= 6 &&
		!isKnown &&
		term.KmFromHome >= 150 &&
		isRisky {
		return 5, true
	}

	return 0, false
}
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
go test ./internal/service/ -run TestFastPath -v
```
Expected: all 5 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add internal/service/fast_path.go internal/service/fast_path_test.go
git commit -m "feat(service): add fast path rules for obvious safe/risky transactions"
```

---

## Task 2: Wire Fast Path into Service

**Files:**
- Modify: `internal/service/fraud_detection.go`

- [ ] **Step 2.1: Update CalculateFraudScore to call fastPath first**

Replace the entire body of `fraud_detection.go`:

```go
// internal/service/fraud_detection.go
package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	Idx *search.IVFIndex
	Vec *vectorizer.Vectorizer
)

// CalculateFraudScore returns fraudCount (0–5): number of fraud neighbors among k=5.
// k=5 and threshold=0.6 are fixed by spec — do not change.
func CalculateFraudScore(req dto.FraudRequest) int {
	if count, ok := fastPath(req); ok {
		return count
	}
	vec := Vec.Vectorize(req)
	return Idx.KNN(vec, 5)
}
```

- [ ] **Step 2.2: Run all tests**

```bash
go test ./... -v 2>&1 | tail -30
```
Expected: all tests PASS (21 tests).

- [ ] **Step 2.3: Commit**

```bash
git add internal/service/fraud_detection.go
git commit -m "feat(service): wire fast path before vectorization and k-NN"
```

---

## Task 3: Shared mmap for Index Loading

**Files:**
- Modify: `internal/search/index.go`

Replace `os.ReadFile` + heap copies with `syscall.Mmap` + zero-copy `unsafe` slices for vectors and labels (the large data). Parse small structures (header, centroids, starts, sizes) normally.

**Why zero-copy matters:** Vectors (N×14×2 bytes ≈ 84 MB) and labels (N bytes ≈ 3 MB) account for ~87 MB. If we copy to heap, both api1 and api2 each have their own heap copy → 174 MB used. With mmap slices pointing into the mapped file, the OS shares physical pages → ~87 MB total (Linux COW page sharing for MAP_SHARED|PROT_READ).

- [ ] **Step 3.1: Verify existing index tests pass before touching anything**

```bash
go test ./internal/search/ -run TestLoad -v
```
Expected: TestLoadIVFIndex_Basic PASS.

- [ ] **Step 3.2: Replace LoadIVFIndex with mmap-based implementation**

Full replacement of `internal/search/index.go`:

```go
package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
	"unsafe"
)

const ivfMagic = "IVF1"
const dims = 14

// IVFIndex stores vectors grouped by cluster for fast approximate KNN.
// Vectors and Labels are zero-copy views into mmap'd file data.
// Binary format (little-endian):
//
//	[4]       "IVF1" magic
//	[4]       uint32 C (number of clusters)
//	[4]       uint32 N (total vectors)
//	[C×14×4]  float32 centroids
//	[C×4]     uint32 cluster starts (index into Vectors/Labels)
//	[C×4]     uint32 cluster sizes
//	[N×14×2]  int16 vectors (all vectors contiguous, not interleaved with labels)
//	[N×1]     uint8 labels
type IVFIndex struct {
	C         int
	N         int
	Centroids []float32
	Starts    []uint32
	Sizes     []uint32
	Vectors   []int16
	Labels    []uint8
	mmap      []byte // retains reference to prevent GC of mmap'd region
}

// Close unmaps the index file. Call at process shutdown.
func (idx *IVFIndex) Close() {
	if idx.mmap != nil {
		_ = syscall.Munmap(idx.mmap)
		idx.mmap = nil
	}
}

func LoadIVFIndex(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(fi.Size())

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	if err := parseIVF(data); err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}

	c := int(binary.LittleEndian.Uint32(data[4:8]))
	n := int(binary.LittleEndian.Uint32(data[8:12]))

	off := 12

	centroids := make([]float32, c*dims)
	for i := range centroids {
		centroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	starts := make([]uint32, c)
	for i := range starts {
		starts[i] = binary.LittleEndian.Uint32(data[off:])
		off += 4
	}

	sizes := make([]uint32, c)
	for i := range sizes {
		sizes[i] = binary.LittleEndian.Uint32(data[off:])
		off += 4
	}

	// Zero-copy: reinterpret mmap bytes as []int16 and []uint8.
	// Safe on little-endian (x86/x86-64): file is LE, host is LE.
	vecBytes := data[off : off+n*dims*2]
	labelsBytes := data[off+n*dims*2 : off+n*dims*2+n]

	vectors := unsafe.Slice((*int16)(unsafe.Pointer(&vecBytes[0])), n*dims)
	labels := labelsBytes

	return &IVFIndex{
		C: c, N: n,
		Centroids: centroids,
		Starts:    starts,
		Sizes:     sizes,
		Vectors:   vectors,
		Labels:    labels,
		mmap:      data,
	}, nil
}

func parseIVF(data []byte) error {
	if len(data) < 12 {
		return fmt.Errorf("index too small: %d bytes", len(data))
	}
	if string(data[0:4]) != ivfMagic {
		return fmt.Errorf("bad magic: %q (want %q)", data[0:4], ivfMagic)
	}
	c := int(binary.LittleEndian.Uint32(data[4:8]))
	n := int(binary.LittleEndian.Uint32(data[8:12]))
	centSize := c * dims * 4
	startsSize := c * 4
	sizesSize := c * 4
	vecsSize := n * dims * 2
	expected := 12 + centSize + startsSize + sizesSize + vecsSize + n
	if len(data) != expected {
		return fmt.Errorf("size mismatch: got %d, want %d", len(data), expected)
	}
	return nil
}
```

- [ ] **Step 3.3: Run index tests**

```bash
go test ./internal/search/ -run TestLoad -v
```
Expected: TestLoadIVFIndex_Basic PASS.

Note: `writeIVFBinary` in the test constructs an `IVFIndex` directly (not via `LoadIVFIndex`) then writes it to a temp file. `LoadIVFIndex` mmaps that temp file. The test doesn't call `Close()` — that's fine, the OS reclaims the mmap at process exit, and `t.TempDir()` cleans the file after the test.

- [ ] **Step 3.4: Run full test suite**

```bash
go test ./... -v 2>&1 | tail -30
```
Expected: all tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add internal/search/index.go
git commit -m "perf(search): mmap index with zero-copy vector/label slices"
```

---

## Task 4: Decision Tree Training Script

**Files:**
- Create: `ml/train_decision_tree.py`

Reads `resources/references.json.gz` (already has pre-computed 14-dim vectors), trains sklearn `DecisionTreeClassifier`, saves model parameters as JSON.

**Note:** This step requires `resources/references.json.gz` (3M records). Training takes ~2-5 minutes on a modern CPU.

- [ ] **Step 4.1: Create training script**

```python
# ml/train_decision_tree.py
"""
Train a decision tree classifier on the reference dataset.
Reads resources/references.json.gz (vectors already pre-computed).
Outputs ml/decision_tree_model.json for Go code generation.

Usage:
    uv run ml/train_decision_tree.py [--depth 20] [--min-leaf 50] [--confidence 0.95]
"""
import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20, help="max tree depth")
    ap.add_argument("--min-leaf", type=int, default=50, help="min samples per leaf")
    ap.add_argument("--confidence", type=float, default=0.95,
                    help="min majority class fraction to mark leaf as confident")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    src = root / "resources" / "references.json.gz"
    dst = root / "ml" / "decision_tree_model.json"

    print(f"Loading {src}...")
    with gzip.open(src) as f:
        records = json.load(f)

    n = len(records)
    print(f"Loaded {n} records")

    vectors = np.array([rec["vector"] for rec in records], dtype=np.float32)
    labels = np.array(
        [1 if rec["label"] == "fraud" else 0 for rec in records], dtype=np.int32
    )

    fraud_count = int(labels.sum())
    print(f"Labels: {fraud_count} fraud ({100*fraud_count/n:.1f}%), {n-fraud_count} legit")

    print(f"Training DecisionTreeClassifier(max_depth={args.depth}, min_samples_leaf={args.min_leaf})...")
    clf = DecisionTreeClassifier(
        max_depth=args.depth,
        min_samples_leaf=args.min_leaf,
        random_state=42,
    )
    clf.fit(vectors, labels)

    tree = clf.tree_
    n_nodes = tree.node_count
    print(f"Tree: {n_nodes} nodes, {tree.max_depth} depth")

    # Determine confidence for each node:
    # value[node] has shape [1, n_classes] = [[n_legit, n_fraud]]
    confident = []
    leaf_class = []
    for i in range(n_nodes):
        v = tree.value[i][0]
        total = float(v.sum())
        majority = float(v.max())
        frac = majority / total if total > 0 else 0.0
        is_leaf = tree.children_left[i] == -1
        is_confident = is_leaf and frac >= args.confidence
        confident.append(bool(is_confident))
        if is_leaf:
            leaf_class.append(int(np.argmax(v)))  # 0=legit, 1=fraud
        else:
            leaf_class.append(-1)

    conf_count = sum(confident)
    print(f"Confident leaves: {conf_count} of {n_nodes} nodes "
          f"(confidence threshold={args.confidence})")

    model = {
        "n_nodes": n_nodes,
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": [float(x) for x in tree.threshold],
        "confident": confident,
        "leaf_class": leaf_class,
    }

    with open(dst, "w") as f:
        json.dump(model, f)
    print(f"Model saved to {dst} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run training**

```bash
cd /home/snow/workspace/rinha-backend/gopher-fraud-detection
uv run ml/train_decision_tree.py
```
Expected output (approximate):
```
Loading resources/references.json.gz...
Loaded 3000000 records
Labels: XXXXXX fraud (XX.X%), XXXXXX legit
Training DecisionTreeClassifier(max_depth=20, min_samples_leaf=50)...
Tree: XXXX nodes, XX depth
Confident leaves: XXX of XXXX nodes (confidence threshold=0.95)
Model saved to ml/decision_tree_model.json (XXX KB)
```

- [ ] **Step 4.3: Commit**

```bash
git add ml/train_decision_tree.py ml/decision_tree_model.json
git commit -m "feat(ml): train decision tree classifier from reference dataset"
```

---

## Task 5: Go Code Generator

**Files:**
- Create: `ml/gen_tree_go.py`
- Create: `internal/search/decision_tree.go` (generated output)

Reads `ml/decision_tree_model.json`, emits `internal/search/decision_tree.go` with flat arrays and an iterative traversal.

- [ ] **Step 5.1: Create code generator**

```python
# ml/gen_tree_go.py
"""
Generate internal/search/decision_tree.go from ml/decision_tree_model.json.

Usage:
    uv run ml/gen_tree_go.py
"""
import json
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    src = root / "ml" / "decision_tree_model.json"
    dst = root / "internal" / "search" / "decision_tree.go"

    with open(src) as f:
        m = json.load(f)

    n = m["n_nodes"]
    left = m["children_left"]
    right = m["children_right"]
    feature = m["feature"]
    threshold = m["threshold"]
    confident = m["confident"]
    leaf_class = m["leaf_class"]

    conf_count = sum(confident)
    print(f"Generating tree: {n} nodes, {conf_count} confident leaves")

    lines = []
    lines.append("package search")
    lines.append("")
    lines.append("// Code generated by ml/gen_tree_go.py — DO NOT EDIT.")
    lines.append(f"// Nodes: {n}, Confident leaves: {conf_count}")
    lines.append("")

    def fmt_int32_arr(name, values):
        body = ", ".join(str(v) for v in values)
        return f"var {name} = [{n}]int32{{{body}}}"

    def fmt_int8_arr(name, values):
        body = ", ".join(str(v) for v in values)
        return f"var {name} = [{n}]int8{{{body}}}"

    def fmt_float32_arr(name, values):
        body = ", ".join(f"{v:.9f}" for v in values)
        return f"var {name} = [{n}]float32{{{body}}}"

    def fmt_bool_arr(name, values):
        body = ", ".join("true" if v else "false" for v in values)
        return f"var {name} = [{n}]bool{{{body}}}"

    lines.append(fmt_int32_arr("dtLeft", left))
    lines.append(fmt_int32_arr("dtRight", right))
    lines.append(fmt_int8_arr("dtFeature", feature))
    lines.append(fmt_float32_arr("dtThreshold", threshold))
    lines.append(fmt_bool_arr("dtConfident", confident))
    lines.append(fmt_int8_arr("dtClass", leaf_class))
    lines.append("")
    lines.append("// DecisionTree classifies vec using the pre-trained fraud decision tree.")
    lines.append("// Returns (fraudCount, true) only when the reached leaf is confident")
    lines.append("// (majority class >= confidence threshold during training).")
    lines.append("// Returns (0, false) for uncertain leaves — caller should fall through to k-NN.")
    lines.append("func DecisionTree(vec [14]float32) (int, bool) {")
    lines.append("\tnode := int32(0)")
    lines.append("\tfor dtLeft[node] != -1 {")
    lines.append("\t\tif vec[dtFeature[node]] <= dtThreshold[node] {")
    lines.append("\t\t\tnode = dtLeft[node]")
    lines.append("\t\t} else {")
    lines.append("\t\t\tnode = dtRight[node]")
    lines.append("\t\t}")
    lines.append("\t}")
    lines.append("\tif !dtConfident[node] {")
    lines.append("\t\treturn 0, false")
    lines.append("\t}")
    lines.append("\tif dtClass[node] == 1 {")
    lines.append("\t\treturn 5, true")
    lines.append("\t}")
    lines.append("\treturn 0, true")
    lines.append("}")
    lines.append("")

    dst.write_text("\n".join(lines))
    size_kb = dst.stat().st_size // 1024
    print(f"Written {dst} ({size_kb} KB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run generator**

```bash
uv run ml/gen_tree_go.py
```
Expected: writes `internal/search/decision_tree.go`.

- [ ] **Step 5.3: Verify generated file compiles**

```bash
go build ./internal/search/
```
Expected: no errors.

- [ ] **Step 5.4: Commit**

```bash
git add ml/gen_tree_go.py internal/search/decision_tree.go
git commit -m "feat(search): generate decision tree Go code from trained model"
```

---

## Task 6: Wire Decision Tree into Pipeline + Tests

**Files:**
- Modify: `internal/service/fraud_detection.go`

Pipeline after this task:
1. `fastPath(req)` → (count, ok) — raw request, ~0μs
2. `Vec.Vectorize(req)` → [14]float32
3. `search.DecisionTree(vec)` → (count, ok) — zero alloc
4. `Idx.KNN(vec, 5)` → count — approximate fallback

- [ ] **Step 6.1: Write test for CalculateFraudScore with obviously safe request (fast path)**

Add to `internal/service/fast_path_test.go` (append):

```go
// TestCalculateFraudScore_FastPath verifies obviously safe/risky requests
// bypass vectorization and k-NN. Uses nil Idx/Vec to prove they're not called.
func TestCalculateFraudScore_FastPath_NoKNN(t *testing.T) {
	// Save and nil out globals to prove they are not touched.
	origIdx := Idx
	origVec := Vec
	Idx = nil
	Vec = nil
	defer func() { Idx = origIdx; Vec = origVec }()

	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 80, Installments: 2},
		Customer:    dto.Customer{AvgAmount: 200, TxCount24h: 3, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-01", MCC: "5411"},
		Terminal:    dto.Terminal{KmFromHome: 10},
	}

	// Should not panic (Idx/Vec are nil), fast path intercepts.
	count := CalculateFraudScore(req)
	if count != 0 {
		t.Errorf("got count=%d, want 0", count)
	}
}
```

- [ ] **Step 6.2: Run to confirm test fails**

```bash
go test ./internal/service/ -run TestCalculateFraudScore_FastPath_NoKNN -v
```
Expected: PASS already (fast path already wired in Task 2). Confirm it doesn't panic.

- [ ] **Step 6.3: Wire decision tree into CalculateFraudScore**

Replace body in `internal/service/fraud_detection.go`:

```go
// internal/service/fraud_detection.go
package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	Idx *search.IVFIndex
	Vec *vectorizer.Vectorizer
)

// CalculateFraudScore returns fraudCount (0–5): number of fraud neighbors among k=5.
// k=5 and threshold=0.6 are fixed by spec — do not change.
// Pipeline: fast_path → decision_tree → k-NN (each step only runs if previous returns ok=false).
func CalculateFraudScore(req dto.FraudRequest) int {
	if count, ok := fastPath(req); ok {
		return count
	}
	vec := Vec.Vectorize(req)
	if count, ok := search.DecisionTree(vec); ok {
		return count
	}
	return Idx.KNN(vec, 5)
}
```

- [ ] **Step 6.4: Run full test suite**

```bash
go test ./... -v 2>&1 | tail -40
```
Expected: all tests PASS.

- [ ] **Step 6.5: Build binary**

```bash
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o fraud-api ./cmd/api
```
Expected: no errors.

- [ ] **Step 6.6: Commit**

```bash
git add internal/service/fraud_detection.go internal/service/fast_path_test.go
git commit -m "feat(service): wire decision tree into fraud pipeline (fast_path → tree → k-NN)"
```

---

## Task 7: Local Benchmark

Verify the hybrid pipeline improves local p99 (should be similar or better than before, with ~79% of requests skipping k-NN).

- [ ] **Step 7.1: Run local bench**

```bash
make bench
```
Expected: p99 ≤ 0.65ms (same or better than before; fast_path reduces k-NN load). Note FP/FN counts — they should be ≤ current (19 FP, 5 FN). If FP/FN increase, review decision tree confidence threshold.

- [ ] **Step 7.2: If FP/FN increase — retrain with lower confidence**

```bash
uv run ml/train_decision_tree.py --confidence 0.99
uv run ml/gen_tree_go.py
go build ./internal/search/
make bench
```

Repeat with decreasing confidence (0.99 → 0.97 → 0.95) until FP/FN stabilize.

- [ ] **Step 7.3: Update PROGRESS.md** with new benchmark results.

---

## Self-Review

**Spec coverage:**
- Fast path rules ✓ (Task 1)
- Fast path wired ✓ (Task 2)
- Shared mmap ✓ (Task 3)
- Decision tree training ✓ (Task 4)
- Go code generation ✓ (Task 5)
- Decision tree wired ✓ (Task 6)
- Local verification ✓ (Task 7)

**Not in scope:** FD-pass (SCM_RIGHTS), AVX2 SIMD.

**Risks:**
- Decision tree may add FP/FN vs pure k-NN. Mitigation: confidence threshold + fallback to k-NN for uncertain leaves.
- mmap `unsafe` slice cast is only correct on little-endian (x86). Remote judge runs x86 Linux — safe.
- The generated `decision_tree.go` will be large (potentially 500+ KB for 1000+ nodes). This is expected and OK.
