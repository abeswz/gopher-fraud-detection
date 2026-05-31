# VP-Tree Exact Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace IVF approximate search with VP-tree exact search to eliminate FP/FN errors and reduce remote p99 from 30ms to <1ms.

**Architecture:** VP-tree partitions space by actual Euclidean distance from a pivot; queries use branch-and-bound pruning via the triangle inequality for exact k-NN. Both index types share a common `Index` interface; `cmd/api/main.go` auto-detects format by reading the first 4 magic bytes at startup. The Python builder gains an `--algo` flag defaulting to `vptree`.

**Tech Stack:** Go 1.26 stdlib only, Python + numpy (offline builder), little-endian binary format `VPT1`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `internal/search/index.go` | Add `Index` interface alongside existing `IVFIndex` |
| Modify | `internal/service/fraud_detection.go` | Change `Idx` type from `*search.IVFIndex` → `search.Index` |
| Create | `internal/search/vp_index.go` | `VPNode`, `VPIndex`, `LoadVPIndex` |
| Create | `internal/search/vp_index_test.go` | Loader round-trip + error cases |
| Create | `internal/search/vp_knn.go` | `VPIndex.KNN` (iterative DFS, branch-and-bound) |
| Create | `internal/search/vp_knn_test.go` | Correctness vs brute force (200 queries, N=500) |
| Modify | `cmd/api/main.go` | Magic-byte dispatch + `readMagic` helper |
| Modify | `ml/build_index.py` | `--algo {vptree,ivf}` flag, vptree default |

---

## Task 1: Index interface + service refactor

**Files:**
- Modify: `internal/search/index.go`
- Modify: `internal/service/fraud_detection.go`

- [ ] **Step 1: Add `Index` interface to `internal/search/index.go`**

Add at the end of the file (after `LoadIVFIndex`):

```go
// Index is the common interface for both IVFIndex and VPIndex.
type Index interface {
	KNN(query [14]float32, k int) int
}
```

- [ ] **Step 2: Update `internal/service/fraud_detection.go`**

Change `Idx` from `*search.IVFIndex` to `search.Index`:

```go
package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	Idx search.Index
	Vec *vectorizer.Vectorizer
)

// CalculateFraudScore returns fraudCount (0–5): number of fraud neighbors among k=5.
// k=5 and threshold=0.6 are fixed by spec — do not change.
func CalculateFraudScore(req dto.FraudRequest) int {
	vec := Vec.Vectorize(req)
	return Idx.KNN(vec, 5)
}
```

- [ ] **Step 3: Verify existing tests still pass**

```
go test ./...
```

Expected: all existing tests pass. `IVFIndex` already has `KNN([14]float32, int) int` so it satisfies `Index` automatically.

- [ ] **Step 4: Commit**

```bash
git add internal/search/index.go internal/service/fraud_detection.go
git commit -m "refactor(search): introduce Index interface for IVF and future VP-tree"
```

---

## Task 2: VPIndex binary loader (TDD)

**Files:**
- Create: `internal/search/vp_index.go`
- Create: `internal/search/vp_index_test.go`

### Binary format reference

```
[4B]   "VPT1" magic
[4B]   uint32 N          — total vectors
[4B]   uint32 nodeCount  — number of nodes
[4B]   uint32 leafSize   — leaf threshold (stored, not used by loader)

[nodeCount × 40B] node array:
  [4B]  float32 tau       — split radius; 0 for leaves
  [4B]  uint32  childOff  — right child node index (internal) | vec array start (leaf)
  [2B]  uint16  count     — 0 = internal; >0 = leaf (# vecs ≤ 16)
  [2B]  _pad
  [28B] int16[14] vec     — pivot ×10000; zeroed for leaves

[N × 28B]  int16[14] vectors — DFS-reordered, ×10000
[N × 1B]   uint8     labels  — DFS-reordered
```

- [ ] **Step 1: Write failing tests in `internal/search/vp_index_test.go`**

```go
package search

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// writeVPBinary serializes a VPIndex to the VPT1 wire format.
func writeVPBinary(nodes []VPNode, vectors []int16, labels []uint8) []byte {
	n := len(labels)
	nodeCount := len(nodes)
	var buf bytes.Buffer
	buf.Write([]byte(vpMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(n))
	binary.Write(&buf, binary.LittleEndian, uint32(nodeCount))
	binary.Write(&buf, binary.LittleEndian, uint32(16)) // leafSize
	for _, nd := range nodes {
		binary.Write(&buf, binary.LittleEndian, nd.Tau)
		binary.Write(&buf, binary.LittleEndian, nd.ChildOff)
		binary.Write(&buf, binary.LittleEndian, nd.Count)
		buf.Write([]byte{0, 0}) // _pad
		for _, v := range nd.Vec {
			binary.Write(&buf, binary.LittleEndian, v)
		}
	}
	for _, v := range vectors {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	buf.Write(labels)
	return buf.Bytes()
}

func TestLoadVPIndex(t *testing.T) {
	// 1 internal node (root) + 2 leaves, 10 vectors total (5 per leaf).
	// Root: tau=0.5, childOff=2 (right leaf), pivot dim0=5000 rest 0.
	// Node 1 (left leaf):  childOff=0, count=5.
	// Node 2 (right leaf): childOff=5, count=5.
	var pivotVec [14]int16
	pivotVec[0] = 5000

	nodes := []VPNode{
		{Tau: 0.5, ChildOff: 2, Count: 0, Vec: pivotVec}, // internal
		{Tau: 0.0, ChildOff: 0, Count: 5},                // left leaf
		{Tau: 0.0, ChildOff: 5, Count: 5},                // right leaf
	}

	vectors := make([]int16, 10*14)
	// vectors 0-4 (left leaf): dim0=1000
	for i := 0; i < 5; i++ {
		vectors[i*14] = 1000
	}
	// vectors 5-9 (right leaf): dim0=8000
	for i := 5; i < 10; i++ {
		vectors[i*14] = 8000
	}

	labels := []uint8{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}

	tmp := t.TempDir() + "/vp.bin"
	if err := os.WriteFile(tmp, writeVPBinary(nodes, vectors, labels), 0644); err != nil {
		t.Fatal(err)
	}

	got, err := LoadVPIndex(tmp)
	if err != nil {
		t.Fatalf("LoadVPIndex error: %v", err)
	}

	if len(got.Nodes) != 3 {
		t.Errorf("nodeCount: got %d, want 3", len(got.Nodes))
	}
	if len(got.Labels) != 10 {
		t.Errorf("N: got %d, want 10", len(got.Labels))
	}

	// Root node fields
	root := got.Nodes[0]
	if math.Abs(float64(root.Tau-0.5)) > 1e-6 {
		t.Errorf("root.Tau: got %v, want 0.5", root.Tau)
	}
	if root.ChildOff != 2 {
		t.Errorf("root.ChildOff: got %d, want 2", root.ChildOff)
	}
	if root.Count != 0 {
		t.Errorf("root.Count: got %d, want 0 (internal)", root.Count)
	}
	if root.Vec[0] != 5000 {
		t.Errorf("root.Vec[0]: got %d, want 5000", root.Vec[0])
	}

	// Left leaf fields
	left := got.Nodes[1]
	if left.Count != 5 {
		t.Errorf("left.Count: got %d, want 5", left.Count)
	}
	if left.ChildOff != 0 {
		t.Errorf("left.ChildOff: got %d, want 0", left.ChildOff)
	}

	// Vectors
	if got.Vectors[0] != 1000 {
		t.Errorf("Vectors[0]: got %d, want 1000", got.Vectors[0])
	}
	if got.Vectors[5*14] != 8000 {
		t.Errorf("Vectors[5*14]: got %d, want 8000", got.Vectors[5*14])
	}

	// Labels
	if got.Labels[1] != 1 {
		t.Errorf("Labels[1]: got %d, want 1", got.Labels[1])
	}
}

func TestLoadVPIndexBadMagic(t *testing.T) {
	nodes := []VPNode{{Tau: 0, ChildOff: 0, Count: 1}}
	vectors := make([]int16, 14)
	labels := []uint8{0}
	data := writeVPBinary(nodes, vectors, labels)
	copy(data[0:4], []byte("BAD!"))

	tmp := t.TempDir() + "/bad.bin"
	os.WriteFile(tmp, data, 0644)

	_, err := LoadVPIndex(tmp)
	if err == nil {
		t.Fatal("expected error for bad magic, got nil")
	}
}

func TestLoadVPIndexSizeMismatch(t *testing.T) {
	nodes := []VPNode{{Tau: 0, ChildOff: 0, Count: 5}}
	vectors := make([]int16, 5*14)
	labels := []uint8{0, 0, 0, 0, 0}
	data := writeVPBinary(nodes, vectors, labels)

	tmp := t.TempDir() + "/trunc.bin"
	os.WriteFile(tmp, data[:len(data)-10], 0644) // truncate

	_, err := LoadVPIndex(tmp)
	if err == nil {
		t.Fatal("expected error for truncated file, got nil")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```
go test ./internal/search/ -run "TestLoadVP" -v
```

Expected: `FAIL` — `vpMagic undefined`, `VPNode undefined`, `LoadVPIndex undefined`.

- [ ] **Step 3: Implement `internal/search/vp_index.go`**

```go
package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const vpMagic = "VPT1"
const vpNodeSize = 40 // 4+4+2+2+28

// VPNode is one node in the VP-tree, stored in DFS order.
// Layout matches the VPT1 binary format exactly (40 bytes, little-endian).
type VPNode struct {
	Tau      float32    // split radius (actual Euclidean distance); 0 for leaves
	ChildOff uint32     // right child node index (internal) | vec array start offset (leaf)
	Count    uint16     // 0 = internal node; >0 = leaf (number of vectors in leaf, ≤ 16)
	_pad     [2]byte
	Vec      [14]int16  // pivot vector scaled ×10000; zeroed for leaf nodes
}

// VPIndex is the in-memory VP-tree exact-search index.
// Vectors and Labels are DFS-reordered to match the node tree layout.
type VPIndex struct {
	Nodes   []VPNode
	Vectors []int16 // N×14 int16, DFS-reordered, scaled ×10000
	Labels  []uint8 // N uint8, DFS-reordered (0=legit, 1=fraud)
}

// LoadVPIndex reads a VPT1-format binary file and returns a ready-to-query VPIndex.
func LoadVPIndex(path string) (*VPIndex, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 16 {
		return nil, fmt.Errorf("vp index too small: %d bytes", len(data))
	}

	if string(data[0:4]) != vpMagic {
		return nil, fmt.Errorf("bad magic: %q (want %q)", data[0:4], vpMagic)
	}

	n := int(binary.LittleEndian.Uint32(data[4:8]))
	nodeCount := int(binary.LittleEndian.Uint32(data[8:12]))
	// data[12:16] is leafSize — stored but not needed at query time

	expected := 16 + nodeCount*vpNodeSize + n*dims*2 + n
	if len(data) != expected {
		return nil, fmt.Errorf("size mismatch: got %d, want %d", len(data), expected)
	}

	off := 16

	nodes := make([]VPNode, nodeCount)
	for i := range nodes {
		nodes[i].Tau = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		nodes[i].ChildOff = binary.LittleEndian.Uint32(data[off+4:])
		nodes[i].Count = binary.LittleEndian.Uint16(data[off+8:])
		// [off+10 : off+12] is _pad — skip
		for j := 0; j < dims; j++ {
			nodes[i].Vec[j] = int16(binary.LittleEndian.Uint16(data[off+12+j*2:]))
		}
		off += vpNodeSize
	}

	vectors := make([]int16, n*dims)
	for i := range vectors {
		vectors[i] = int16(binary.LittleEndian.Uint16(data[off:]))
		off += 2
	}

	labels := make([]uint8, n)
	copy(labels, data[off:off+n])

	return &VPIndex{
		Nodes:   nodes,
		Vectors: vectors,
		Labels:  labels,
	}, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
go test ./internal/search/ -run "TestLoadVP" -v
```

Expected: `PASS` — all three tests pass.

- [ ] **Step 5: Run full test suite to check no regressions**

```
go test ./...
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add internal/search/vp_index.go internal/search/vp_index_test.go
git commit -m "feat(search): add VPIndex binary loader (VPT1 format)"
```

---

## Task 3: VP-tree KNN (TDD)

**Files:**
- Create: `internal/search/vp_knn.go`
- Create: `internal/search/vp_knn_test.go`

### Algorithm summary

Iterative DFS with explicit stack. Prune subtrees when `minDist >= maxRadius` (triangle inequality). Heap stores squared distances (like IVF); `maxRadius = sqrt(maxDistSq)` maintained for subtree pruning that uses actual distances. One `sqrt` per internal node visited (~50–200 total per query).

Node addressing:
- Left child of node `i`: always `i+1` (implicit, no storage)
- Right child of node `i`: `node[i].ChildOff`
- Leaf: `node[i].Count > 0`

- [ ] **Step 1: Write failing tests in `internal/search/vp_knn_test.go`**

```go
package search

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// buildVPIndexForTest recursively builds a VPIndex from float32 vectors and labels.
// Only used in tests — not part of the production API.
func buildVPIndexForTest(rawVecs [][]float32, labels []uint8, leafSize int) *VPIndex {
	rng := rand.New(rand.NewSource(42))
	n := len(rawVecs)

	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	type nodeEntry struct {
		tau      float32
		childOff uint32
		count    uint16
		vec      [14]int16
	}

	var nodeArr []nodeEntry
	var vecOrder []int

	var buildTree func(idx []int)
	buildTree = func(idx []int) {
		if len(idx) <= leafSize {
			vecStart := len(vecOrder)
			vecOrder = append(vecOrder, idx...)
			nodeArr = append(nodeArr, nodeEntry{
				count:    uint16(len(idx)),
				childOff: uint32(vecStart),
			})
			return
		}

		pivotPos := rng.Intn(len(idx))
		pivotIdx := idx[pivotPos]
		pivotVec := rawVecs[pivotIdx]

		dists := make([]float32, len(idx))
		for i, vi := range idx {
			dists[i] = euclidDistVP(rawVecs[vi], pivotVec)
		}

		sorted := make([]float32, len(dists))
		copy(sorted, dists)
		sort.Slice(sorted, func(a, b int) bool { return sorted[a] < sorted[b] })
		tau := sorted[len(sorted)/2]

		var inner, outer []int
		for i, vi := range idx {
			if dists[i] <= tau {
				inner = append(inner, vi)
			} else {
				outer = append(outer, vi)
			}
		}
		if len(inner) == 0 || len(outer) == 0 {
			mid := len(idx) / 2
			inner, outer = idx[:mid], idx[mid:]
		}

		ni := len(nodeArr)
		nodeArr = append(nodeArr, nodeEntry{}) // placeholder for internal node
		buildTree(inner)                       // left child always at ni+1
		rightNi := len(nodeArr)

		var vec [14]int16
		for j, f := range pivotVec {
			vec[j] = int16(math.Round(float64(f) * 10000))
		}
		nodeArr[ni] = nodeEntry{
			tau:      tau,
			childOff: uint32(rightNi),
			count:    0,
			vec:      vec,
		}
		buildTree(outer)
	}

	buildTree(indices)

	result := &VPIndex{
		Nodes:   make([]VPNode, len(nodeArr)),
		Vectors: make([]int16, n*dims),
		Labels:  make([]uint8, n),
	}
	for i, e := range nodeArr {
		result.Nodes[i] = VPNode{
			Tau:      e.tau,
			ChildOff: e.childOff,
			Count:    e.count,
			Vec:      e.vec,
		}
	}
	for newIdx, origIdx := range vecOrder {
		for j := 0; j < dims; j++ {
			result.Vectors[newIdx*dims+j] = int16(math.Round(float64(rawVecs[origIdx][j]) * 10000))
		}
		result.Labels[newIdx] = labels[origIdx]
	}
	return result
}

// euclidDistVP computes Euclidean distance between two float32 vectors (test helper).
func euclidDistVP(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

// bruteForceVPKNN computes exact KNN directly over the int16 Vectors in idx.
// This uses the same dequantization as VPIndex.KNN, so results must match exactly.
func bruteForceVPKNN(idx *VPIndex, query [14]float32, k int) int {
	n := len(idx.Labels)
	var topArr [5]knnEntry
	top := topArr[:0]
	maxDist := float32(0)
	maxPos := 0

	for vi := 0; vi < n; vi++ {
		base := vi * dims
		_ = idx.Vectors[base+13]
		d0 := query[0] - float32(idx.Vectors[base])*invScale
		d1 := query[1] - float32(idx.Vectors[base+1])*invScale
		d2 := query[2] - float32(idx.Vectors[base+2])*invScale
		d3 := query[3] - float32(idx.Vectors[base+3])*invScale
		d4 := query[4] - float32(idx.Vectors[base+4])*invScale
		d5 := query[5] - float32(idx.Vectors[base+5])*invScale
		d6 := query[6] - float32(idx.Vectors[base+6])*invScale
		d7 := query[7] - float32(idx.Vectors[base+7])*invScale
		d8 := query[8] - float32(idx.Vectors[base+8])*invScale
		d9 := query[9] - float32(idx.Vectors[base+9])*invScale
		d10 := query[10] - float32(idx.Vectors[base+10])*invScale
		d11 := query[11] - float32(idx.Vectors[base+11])*invScale
		d12 := query[12] - float32(idx.Vectors[base+12])*invScale
		d13 := query[13] - float32(idx.Vectors[base+13])*invScale
		distSq := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

		if len(top) < k {
			top = append(top, knnEntry{distSq, idx.Labels[vi]})
			if len(top) == k {
				maxDist, maxPos = knnFindMax(top)
			}
		} else if distSq < maxDist {
			top[maxPos] = knnEntry{distSq, idx.Labels[vi]}
			maxDist, maxPos = knnFindMax(top)
		}
	}

	count := 0
	for _, e := range top {
		if e.label == 1 {
			count++
		}
	}
	return count
}

func TestVPKNNLeafOnly(t *testing.T) {
	// N=10 → single leaf, no tree traversal.
	rawVecs := make([][]float32, 10)
	labels := make([]uint8, 10)
	rng := rand.New(rand.NewSource(1))
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = rng.Float32()
		}
		if i < 3 {
			labels[i] = 1
		}
	}

	idx := buildVPIndexForTest(rawVecs, labels, 16)
	if len(idx.Nodes) != 1 || idx.Nodes[0].Count == 0 {
		t.Fatalf("expected single leaf node, got %d nodes, count=%d", len(idx.Nodes), idx.Nodes[0].Count)
	}

	var query [14]float32
	for j := range query {
		query[j] = rng.Float32()
	}

	got := idx.KNN(query, 5)
	want := bruteForceVPKNN(idx, query, 5)
	if got != want {
		t.Errorf("LeafOnly: got %d, want %d", got, want)
	}
}

func TestVPKNNMatchesBruteForce(t *testing.T) {
	// N=500, ~30% fraud labels, 200 random queries — acceptance bar from spec.
	const n = 500
	rawVecs := make([][]float32, n)
	labels := make([]uint8, n)
	rng := rand.New(rand.NewSource(99))

	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = rng.Float32()*2 - 1 // [-1, 1]
		}
		if rng.Float32() < 0.3 {
			labels[i] = 1
		}
	}

	idx := buildVPIndexForTest(rawVecs, labels, 16)

	failures := 0
	for q := 0; q < 200; q++ {
		var query [14]float32
		for j := range query {
			query[j] = rng.Float32()*2 - 1
		}
		got := idx.KNN(query, 5)
		want := bruteForceVPKNN(idx, query, 5)
		if got != want {
			t.Errorf("query %d: got fraudCount=%d, want %d", q, got, want)
			failures++
			if failures >= 5 {
				t.Fatal("too many mismatches, stopping")
			}
		}
	}
}

func TestVPKNNExactPivotHit(t *testing.T) {
	// Query identical to a stored vector → dist=0, must appear in top-5.
	rawVecs := make([][]float32, 30)
	labels := make([]uint8, 30)
	rng := rand.New(rand.NewSource(7))

	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = rng.Float32()
		}
	}
	// Make index 0 a fraud vector with a known value.
	for j := range rawVecs[0] {
		rawVecs[0][j] = 0.5
	}
	labels[0] = 1

	idx := buildVPIndexForTest(rawVecs, labels, 16)

	// Build query from the stored int16 representation of vector 0.
	// Find where vector 0 landed in DFS order.
	var query [14]float32
	// Search for the stored vector that equals round(0.5*10000)=5000 in all dims.
	for vi := 0; vi < len(labels); vi++ {
		base := vi * dims
		match := true
		for j := 0; j < dims; j++ {
			if idx.Vectors[base+j] != 5000 {
				match = false
				break
			}
		}
		if match {
			for j := 0; j < dims; j++ {
				query[j] = float32(idx.Vectors[base+j]) * invScale
			}
			break
		}
	}

	got := idx.KNN(query, 5)
	want := bruteForceVPKNN(idx, query, 5)
	if got != want {
		t.Errorf("ExactPivotHit: got %d, want %d", got, want)
	}
}

func TestVPKNNDegenerateSplit(t *testing.T) {
	// All vectors equidistant from pivot → triggers midpoint split path.
	// 32 vectors arranged at equal radius (0.5) from origin in dim 0 only.
	// Query at [0.9,...] → distinct distances, no ties.
	const n = 32
	rawVecs := make([][]float32, n)
	labels := make([]uint8, n)
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		rawVecs[i][0] = float32(i) * (1.0 / float32(n)) // unique dim0 values
		// dims 1-13 identical → equidistant from an origin pivot in dims 1-13
		for j := 1; j < dims; j++ {
			rawVecs[i][j] = 0.5
		}
	}
	for i := 0; i < 16; i++ {
		labels[i] = 1
	}

	idx := buildVPIndexForTest(rawVecs, labels, 16)

	var query [14]float32
	query[0] = 0.9
	for j := 1; j < dims; j++ {
		query[j] = 0.5
	}

	got := idx.KNN(query, 5)
	want := bruteForceVPKNN(idx, query, 5)
	if got != want {
		t.Errorf("DegenerateSplit: got %d, want %d", got, want)
	}
}

func TestVPKNNAllFraud(t *testing.T) {
	rawVecs := make([][]float32, 10)
	labels := make([]uint8, 10)
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = 1.0
		}
		labels[i] = 1
	}
	idx := buildVPIndexForTest(rawVecs, labels, 16)

	var query [14]float32
	for j := range query {
		query[j] = 0.9
	}
	got := idx.KNN(query, 5)
	if got != 5 {
		t.Errorf("AllFraud: got %d, want 5", got)
	}
}

func TestVPKNNAllLegit(t *testing.T) {
	rawVecs := make([][]float32, 10)
	labels := make([]uint8, 10) // all zero (legit)
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
	}
	idx := buildVPIndexForTest(rawVecs, labels, 16)

	var query [14]float32
	for j := range query {
		query[j] = 0.1
	}
	got := idx.KNN(query, 5)
	if got != 0 {
		t.Errorf("AllLegit: got %d, want 0", got)
	}
}

func BenchmarkVPKNN(b *testing.B) {
	const n = 3000
	rawVecs := make([][]float32, n)
	labels := make([]uint8, n)
	rng := rand.New(rand.NewSource(42))
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = rng.Float32()
		}
		if rng.Float32() < 0.3 {
			labels[i] = 1
		}
	}
	idx := buildVPIndexForTest(rawVecs, labels, 16)
	var query [14]float32
	for j := range query {
		query[j] = rng.Float32()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.KNN(query, 5)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```
go test ./internal/search/ -run "TestVPKNN" -v
```

Expected: `FAIL` — `KNN` method undefined on `*VPIndex`.

- [ ] **Step 3: Implement `internal/search/vp_knn.go`**

```go
package search

import "math"

type vpStackEntry struct {
	nodeIdx uint32
	minDist float32 // min possible actual distance from query to any point in this subtree
}

// KNN finds the k nearest neighbors in the VP-tree using iterative DFS with
// branch-and-bound pruning (triangle inequality). Returns the fraud label count
// among the top-k neighbors. Zero heap allocations per call.
func (idx *VPIndex) KNN(query [14]float32, k int) int {
	var stackArr [40]vpStackEntry // depth ≤ 18×2 + headroom; stack-allocated
	stack := stackArr[:0]

	var topArr [5]knnEntry // k=5 fixed by spec; stack-allocated
	top := topArr[:0]
	maxDistSq := float32(math.MaxFloat32)
	maxRadius := float32(math.MaxFloat32) // sqrt(maxDistSq); updated on each heap change
	maxPos := 0

	q0 := query[0]
	q1 := query[1]
	q2 := query[2]
	q3 := query[3]
	q4 := query[4]
	q5 := query[5]
	q6 := query[6]
	q7 := query[7]
	q8 := query[8]
	q9 := query[9]
	q10 := query[10]
	q11 := query[11]
	q12 := query[12]
	q13 := query[13]

	stack = append(stack, vpStackEntry{0, 0})

	nodes := idx.Nodes
	vecs := idx.Vectors
	labs := idx.Labels

	for len(stack) > 0 {
		e := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if len(top) == k && e.minDist >= maxRadius {
			continue // entire subtree pruned by triangle inequality
		}

		node := nodes[e.nodeIdx]

		if node.Count > 0 {
			// LEAF: unrolled 14-dim linear scan with early exit at dim 0 and dim 7.
			base := int(node.ChildOff) * dims
			end := int(node.ChildOff) + int(node.Count)
			for vi := int(node.ChildOff); vi < end; vi, base = vi+1, base+dims {
				_ = vecs[base+13] // bounds-check hint: proves all 14 accesses in-bounds

				d0 := q0 - float32(vecs[base])*invScale
				distSq := d0 * d0
				if len(top) == k && distSq >= maxDistSq {
					continue
				}

				d1 := q1 - float32(vecs[base+1])*invScale
				d2 := q2 - float32(vecs[base+2])*invScale
				d3 := q3 - float32(vecs[base+3])*invScale
				d4 := q4 - float32(vecs[base+4])*invScale
				d5 := q5 - float32(vecs[base+5])*invScale
				d6 := q6 - float32(vecs[base+6])*invScale
				d7 := q7 - float32(vecs[base+7])*invScale
				distSq += d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7
				if len(top) == k && distSq >= maxDistSq {
					continue
				}

				d8 := q8 - float32(vecs[base+8])*invScale
				d9 := q9 - float32(vecs[base+9])*invScale
				d10 := q10 - float32(vecs[base+10])*invScale
				d11 := q11 - float32(vecs[base+11])*invScale
				d12 := q12 - float32(vecs[base+12])*invScale
				d13 := q13 - float32(vecs[base+13])*invScale
				distSq += d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

				if len(top) < k {
					top = append(top, knnEntry{distSq, labs[vi]})
					if len(top) == k {
						maxDistSq, maxPos = knnFindMax(top)
						maxRadius = float32(math.Sqrt(float64(maxDistSq)))
					}
				} else if distSq < maxDistSq {
					top[maxPos] = knnEntry{distSq, labs[vi]}
					maxDistSq, maxPos = knnFindMax(top)
					maxRadius = float32(math.Sqrt(float64(maxDistSq)))
				}
			}
			continue
		}

		// INTERNAL: compute actual distance to pivot (1 sqrt per internal node visited).
		v := node.Vec
		_ = v[13]
		d0 := q0 - float32(v[0])*invScale
		d1 := q1 - float32(v[1])*invScale
		d2 := q2 - float32(v[2])*invScale
		d3 := q3 - float32(v[3])*invScale
		d4 := q4 - float32(v[4])*invScale
		d5 := q5 - float32(v[5])*invScale
		d6 := q6 - float32(v[6])*invScale
		d7 := q7 - float32(v[7])*invScale
		d8 := q8 - float32(v[8])*invScale
		d9 := q9 - float32(v[9])*invScale
		d10 := q10 - float32(v[10])*invScale
		d11 := q11 - float32(v[11])*invScale
		d12 := q12 - float32(v[12])*invScale
		d13 := q13 - float32(v[13])*invScale
		distSq := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
		dist := float32(math.Sqrt(float64(distSq)))
		tau := node.Tau

		leftChild := e.nodeIdx + 1
		rightChild := node.ChildOff

		if dist <= tau {
			// Query inside ball: left (inner) is closer.
			// Push right (farther) first so left is popped first.
			outerMin := tau - dist
			if outerMin < maxRadius {
				stack = append(stack, vpStackEntry{rightChild, outerMin})
			}
			stack = append(stack, vpStackEntry{leftChild, 0})
		} else {
			// Query outside ball: right (outer) is closer.
			// Push left (farther) first so right is popped first.
			innerMin := dist - tau
			if innerMin < maxRadius {
				stack = append(stack, vpStackEntry{leftChild, innerMin})
			}
			stack = append(stack, vpStackEntry{rightChild, 0})
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
go test ./internal/search/ -run "TestVPKNN" -v
```

Expected: all 6 tests pass. `TestVPKNNMatchesBruteForce` must pass all 200 queries — any failure is a correctness bug.

- [ ] **Step 5: Run benchmark to verify performance**

```
go test ./internal/search/ -bench BenchmarkVPKNN -benchtime=5s
```

Expected: significantly faster than `BenchmarkKNN` (IVF); target <5μs/op on a 3000-vector test index.

- [ ] **Step 6: Run full test suite**

```
go test ./...
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add internal/search/vp_knn.go internal/search/vp_knn_test.go
git commit -m "feat(search): implement VP-tree KNN with branch-and-bound pruning"
```

---

## Task 4: Magic-byte dispatch in `cmd/api/main.go`

**Files:**
- Modify: `cmd/api/main.go`

- [ ] **Step 1: Rewrite `cmd/api/main.go` with magic-byte auto-detection**

```go
package main

import (
	"gopher-fraud-detection/internal/router"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/service"
	"gopher-fraud-detection/internal/vectorizer"
	"io"
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

func readMagic(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var buf [4]byte
	_, err = io.ReadFull(f, buf[:])
	return string(buf[:]), err
}

func main() {
	indexPath := envOr("INDEX_PATH", "index/references.bin")
	normPath := envOr("NORM_PATH", "resources/normalization.json")
	mccPath := envOr("MCC_PATH", "resources/mcc_risk.json")

	vec, err := vectorizer.Load(normPath, mccPath)
	if err != nil {
		log.Fatalf("load vectorizer: %v", err)
	}

	magic, err := readMagic(indexPath)
	if err != nil {
		log.Fatalf("read index magic: %v", err)
	}

	var idx search.Index
	var n int

	switch magic {
	case "IVF1":
		ivf, err := search.LoadIVFIndex(indexPath)
		if err != nil {
			log.Fatalf("load IVF index: %v", err)
		}
		idx = ivf
		n = ivf.N
	case "VPT1":
		vp, err := search.LoadVPIndex(indexPath)
		if err != nil {
			log.Fatalf("load VP index: %v", err)
		}
		idx = vp
		n = len(vp.Labels)
	default:
		log.Fatalf("unknown index format: %q (want IVF1 or VPT1)", magic)
	}

	service.Vec = vec
	service.Idx = idx

	log.Printf("loaded %d vectors (format: %s)", n, magic)

	sock := envOr("SOCK", "")
	if sock == "" {
		log.Fatal("SOCK environment variable is required")
	}

	_ = os.Remove(sock)

	listener, err := net.Listen("unix", sock)
	if err != nil {
		log.Fatal(err)
	}

	if err := os.Chmod(sock, 0666); err != nil {
		log.Fatal(err)
	}

	server := &http.Server{Handler: router.New()}
	log.Fatal(server.Serve(listener))
}
```

- [ ] **Step 2: Build to verify no compile errors**

```
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /tmp/fraud-api ./cmd/api
```

Expected: binary produced with no errors.

- [ ] **Step 3: Run full test suite**

```
go test ./...
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add cmd/api/main.go
git commit -m "feat(api): auto-detect index format by magic bytes (IVF1/VPT1)"
```

---

## Task 5: Python VP-tree builder

**Files:**
- Modify: `ml/build_index.py`

- [ ] **Step 1: Rewrite `ml/build_index.py` with `--algo` flag**

```python
import argparse
import gzip
import json
import struct
import sys
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

N_CLUSTERS = 4000
NPROBE_DEFAULT = 20
LEAF_SIZE = 16


def build_ivf(vectors, labels, dst):
    n = len(labels)
    print(f"Fitting MiniBatchKMeans with {N_CLUSTERS} clusters...")
    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=3,
        batch_size=100_000,
        verbose=0,
    )
    assignments = km.fit_predict(vectors)
    centroids = km.cluster_centers_.astype(np.float32)
    print(f"K-means done. Centroids: {centroids.shape}")

    sort_idx = np.argsort(assignments, kind="stable")
    vectors_sorted = vectors[sort_idx]
    labels_sorted = labels[sort_idx]

    cluster_sizes = np.bincount(assignments, minlength=N_CLUSTERS).astype(np.uint32)
    cluster_starts = np.zeros(N_CLUSTERS, dtype=np.uint32)
    cluster_starts[1:] = np.cumsum(cluster_sizes[:-1])

    vectors_int16 = np.clip(np.round(vectors_sorted * 10000), -32768, 32767).astype(np.int16)

    print("Writing IVF index...")
    with open(dst, "wb") as out:
        out.write(b"IVF1")
        out.write(struct.pack("<II", N_CLUSTERS, n))
        out.write(centroids.astype("<f4").tobytes())
        out.write(cluster_starts.astype("<u4").tobytes())
        out.write(cluster_sizes.astype("<u4").tobytes())
        out.write(vectors_int16.astype("<i2").tobytes())
        out.write(labels_sorted.astype("u1").tobytes())

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"{n} vectors, {N_CLUSTERS} clusters → {dst} ({size_mb:.1f} MB)")
    avg = cluster_sizes.mean()
    print(f"Avg cluster size: {avg:.0f}, nprobe={NPROBE_DEFAULT} → ~{avg*NPROBE_DEFAULT:.0f} vecs/query")


def build_vptree(vectors, labels, dst):
    n = len(labels)
    rng = np.random.default_rng(42)
    vectors_int16 = np.clip(np.round(vectors * 10000), -32768, 32767).astype(np.int16)

    # --- Phase 1: recursive tree construction ---
    # Returns a nested tuple tree: ('leaf', indices) | ('node', pivot_idx, tau, left, right)
    def build(indices):
        if len(indices) <= LEAF_SIZE:
            return ("leaf", indices)

        pivot_pos = int(rng.integers(len(indices)))
        pivot_idx = indices[pivot_pos]
        pivot_vec = vectors[pivot_idx]

        diffs = vectors[indices] - pivot_vec
        dists_sq = np.einsum("ij,ij->i", diffs, diffs)
        dists = np.sqrt(dists_sq)
        tau = float(np.median(dists))

        mask = dists <= tau
        if mask.all() or (~mask).all():
            mid = len(indices) // 2
            inner = indices[:mid]
            outer = indices[mid:]
        else:
            inner = indices[mask]
            outer = indices[~mask]

        return ("node", pivot_idx, tau, build(inner), build(outer))

    # --- Phase 2: DFS serialization into flat arrays ---
    nodes = []      # list of dicts, packed to binary in order
    vec_order = []  # original indices in DFS traversal order

    def serialize(tree):
        if tree[0] == "leaf":
            _, indices = tree
            vec_start = len(vec_order)
            vec_order.extend(indices.tolist())
            nodes.append({
                "leaf": True,
                "childOff": vec_start,
                "count": len(indices),
                "tau": 0.0,
                "vec": np.zeros(14, dtype=np.int16),
            })
            return

        _, pivot_idx, tau, left, right = tree
        ni = len(nodes)
        nodes.append(None)      # placeholder; filled after left subtree serialized
        serialize(left)         # left child always at ni+1 (implicit)
        right_ni = len(nodes)
        vec_i16 = np.clip(np.round(vectors[pivot_idx] * 10000), -32768, 32767).astype(np.int16)
        nodes[ni] = {
            "leaf": False,
            "tau": tau,
            "childOff": right_ni,
            "count": 0,
            "vec": vec_i16,
        }
        serialize(right)

    print("Building VP-tree (recursive)...")
    sys.setrecursionlimit(50000)
    all_indices = np.arange(n)
    tree = build(all_indices)
    print(f"Tree built. Serializing {n} vectors...")
    serialize(tree)

    # --- Phase 3: reorder vectors/labels and write VPT1 ---
    vec_order_arr = np.array(vec_order, dtype=np.int64)
    vectors_dfs = vectors_int16[vec_order_arr]
    labels_dfs = labels[vec_order_arr]

    node_count = len(nodes)
    print(f"Writing VPT1 index ({node_count} nodes)...")
    with open(dst, "wb") as out:
        out.write(b"VPT1")
        out.write(struct.pack("<III", n, node_count, LEAF_SIZE))
        for nd in nodes:
            out.write(struct.pack("<fIHH", nd["tau"], nd["childOff"], nd["count"], 0))
            out.write(nd["vec"].astype("<i2").tobytes())
        out.write(vectors_dfs.astype("<i2").tobytes())
        out.write(labels_dfs.astype("u1").tobytes())

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"{n} vectors, {node_count} nodes → {dst} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build fraud detection search index")
    parser.add_argument(
        "--algo",
        choices=["vptree", "ivf"],
        default="vptree",
        help="Index algorithm (default: vptree)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    src = root / "resources" / "references.json.gz"
    dst = root / "index" / "references.bin"
    dst.parent.mkdir(exist_ok=True)

    print("Loading records...")
    with gzip.open(src) as f:
        records = json.load(f)

    n = len(records)
    print(f"Loaded {n} records")

    vectors = np.array([rec["vector"] for rec in records], dtype=np.float32)
    labels = np.array([1 if rec["label"] == "fraud" else 0 for rec in records], dtype=np.uint8)

    if args.algo == "vptree":
        build_vptree(vectors, labels, dst)
    else:
        build_ivf(vectors, labels, dst)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script syntax**

```
uv run python -c "import ml.build_index" 2>/dev/null || uv run python -m py_compile ml/build_index.py && echo "OK"
```

Expected: `OK` — no syntax errors.

- [ ] **Step 3: Verify help text**

```
uv run ml/build_index.py --help
```

Expected: shows `--algo {vptree,ivf}` with `default: vptree`.

- [ ] **Step 4: Commit**

```bash
git add ml/build_index.py
git commit -m "feat(ml): add VP-tree index builder with --algo flag (vptree default)"
```

---

## Final Verification

- [ ] **Build index with vptree**

```
uv run ml/build_index.py
```

Expected: `VPT1` file written to `index/references.bin`. Build takes 5–15 minutes on 3M vectors.

- [ ] **Build binary and spot-check startup**

```bash
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /tmp/fraud-api ./cmd/api
INDEX_PATH=index/references.bin SOCK=/tmp/test.sock /tmp/fraud-api &
sleep 1
curl --unix-socket /tmp/test.sock http://localhost/health
kill %1
```

Expected: server starts and logs `loaded 3000000 vectors (format: VPT1)`.

- [ ] **Run full test suite one last time**

```
go test ./...
```

Expected: all tests pass.

- [ ] **Use superpowers:finishing-a-development-branch to wrap up**
