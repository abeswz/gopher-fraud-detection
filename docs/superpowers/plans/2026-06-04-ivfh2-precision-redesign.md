# IVF_H2 Precision Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-level IVF (C=8000, nprobe=25) with a 2-level hierarchical IVF (IVF_H2) and dual index split (first_tx / subsequent_tx) to eliminate FN=1, reduce FP≤5, and cut p99 latency.

**Architecture:** Two `.ivfh` files (first_tx.ivfh, subsequent_tx.ivfh) replace `references.bin`. Each uses a 2-level KMeans (K1 macro × K2 micro clusters), boundary oversampling during index build, cluster radii for exact centroid pruning, and D_safe for fast-exit on clear-legit queries. Service routes by `req.LastTransaction == nil`.

**Tech Stack:** Go 1.26 (stdlib only), cuML/cuPy for GPU KMeans, numpy, uv

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `ml/build_index.py` | Full rewrite: data split, boundary oversampling, 2-level KMeans, balanced k-means, D_safe, cluster_radius, write IVFH |
| Modify | `internal/search/index.go` | Add `IVFHIndex` struct + `LoadIVFHIndex` + `Close`; keep existing `IVFIndex` and `Index` interface |
| Create | `internal/search/index_ivfh_test.go` | Tests for IVFH binary load/parse |
| Modify | `internal/search/knn.go` | Add `sqrt32` helper + `(*IVFHIndex).KNN` (Phase 1→3 with adaptive nprobe) |
| Create | `internal/search/knn_ivfh_test.go` | Tests for IVFHIndex.KNN (all-fraud, all-legit, mixed, fast-exit, centroid pruning) |
| Modify | `internal/service/fraud_detection.go` | Swap global `Idx *IVFIndex` for `FirstTxIdx, SubseqIdx search.Index`; route by nil LastTransaction |
| Modify | `cmd/api/main.go` | Load two IVFH files; set both service globals |

---

## Task 1: IVFHIndex Struct + Loader

**Files:**
- Modify: `internal/search/index.go`
- Create: `internal/search/index_ivfh_test.go`

### Binary format recap

```
Offset   Size           Field
0        4B             magic "IVFH"
4        4B float32     D_safe (L2 distance, not squared)
8        4B uint32      K1 (macro cluster count)
12       4B uint32      K2 (micro clusters per macro)
16       4B uint32      N  (total vectors)
20       K1×16×4B       macro_centroids (float32, row-major)
+        K1×K2×16×4B    micro_centroids (float32, indexed [macro*K2+micro]*16)
+        K1×K2×4B       cluster_starts (uint32)
+        K1×K2×4B       cluster_sizes  (uint32)
+        K1×K2×4B       cluster_radii  (float32)
+        N×16×2B        flat_vectors   (int16 ×10000)
+        N×1B           flat_labels    (uint8: 0=legit, 1=fraud)
```

- [ ] **Step 1: Write the failing test**

Create `internal/search/index_ivfh_test.go`:

```go
package search

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// writeIVFHBinary serializes an IVFHIndex to the IVFH on-disk format.
// Used only in tests — mirrors the format that build_index.py writes.
func writeIVFHBinary(idx *IVFHIndex) []byte {
	K1, K2, N := idx.K1, idx.K2, idx.N
	var buf bytes.Buffer
	buf.Write([]byte("IVFH"))
	binary.Write(&buf, binary.LittleEndian, idx.DSafe)
	binary.Write(&buf, binary.LittleEndian, uint32(K1))
	binary.Write(&buf, binary.LittleEndian, uint32(K2))
	binary.Write(&buf, binary.LittleEndian, uint32(N))
	for _, v := range idx.MacroCentroids {
		binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	for _, v := range idx.MicroCentroids {
		binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	for _, v := range idx.Starts {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	for _, v := range idx.Sizes {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	for _, v := range idx.Radii {
		binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	for _, v := range idx.Vectors {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	buf.Write(idx.Labels)
	return buf.Bytes()
}

func makeMinimalIVFHIndex() *IVFHIndex {
	// K1=2, K2=2 → 4 leaf clusters; N=4 (one vector per leaf)
	K1, K2, N := 2, 2, 4
	idx := &IVFHIndex{
		K1: K1, K2: K2, N: N,
		DSafe:          0.5,
		NCoarseProbe:   2,
		MacroCentroids: make([]float32, K1*16),
		MicroCentroids: make([]float32, K1*K2*16),
		Starts:         []uint32{0, 1, 2, 3},
		Sizes:          []uint32{1, 1, 1, 1},
		Radii:          []float32{0.1, 0.1, 0.1, 0.1},
		Vectors:        make([]int16, N*16),
		Labels:         []uint8{0, 1, 0, 1},
	}
	// macro 0 centroid at (0,0,...), macro 1 at (1,0,...)
	idx.MacroCentroids[16] = 1.0 // macro 1, dim 0
	// set vector 1 (leaf 1) to 1.0 in all dims → int16 10000
	for j := 0; j < 16; j++ {
		idx.Vectors[1*16+j] = 10000
	}
	return idx
}

func TestLoadIVFHIndex_Fields(t *testing.T) {
	idx := makeMinimalIVFHIndex()
	data := writeIVFHBinary(idx)

	tmp := t.TempDir() + "/test.ivfh"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		t.Fatal(err)
	}

	got, err := LoadIVFHIndex(tmp, 2)
	if err != nil {
		t.Fatalf("LoadIVFHIndex: %v", err)
	}
	defer got.Close()

	if got.K1 != 2 {
		t.Errorf("K1: got %d, want 2", got.K1)
	}
	if got.K2 != 2 {
		t.Errorf("K2: got %d, want 2", got.K2)
	}
	if got.N != 4 {
		t.Errorf("N: got %d, want 4", got.N)
	}
	if got.DSafe != 0.5 {
		t.Errorf("DSafe: got %f, want 0.5", got.DSafe)
	}
	if got.DSafeSq != 0.25 {
		t.Errorf("DSafeSq: got %f, want 0.25", got.DSafeSq)
	}
	if got.NCoarseProbe != 2 {
		t.Errorf("NCoarseProbe: got %d, want 2", got.NCoarseProbe)
	}
	if got.MacroCentroids[16] != 1.0 {
		t.Errorf("MacroCentroids[16]: got %f, want 1.0", got.MacroCentroids[16])
	}
	if got.Radii[0] != 0.1 {
		t.Errorf("Radii[0]: got %f, want 0.1", got.Radii[0])
	}
	if got.Vectors[1*16] != 10000 {
		t.Errorf("Vectors[16]: got %d, want 10000", got.Vectors[1*16])
	}
	if got.Labels[1] != 1 {
		t.Errorf("Labels[1]: got %d, want 1", got.Labels[1])
	}
}

func TestLoadIVFHIndex_BadMagic(t *testing.T) {
	data := []byte("IVF2\x00\x00\x00\x00")
	tmp := t.TempDir() + "/bad.ivfh"
	os.WriteFile(tmp, data, 0644)
	_, err := LoadIVFHIndex(tmp, 1)
	if err == nil {
		t.Error("expected error for bad magic, got nil")
	}
}

func TestLoadIVFHIndex_SizeMismatch(t *testing.T) {
	idx := makeMinimalIVFHIndex()
	data := writeIVFHBinary(idx)
	data = data[:len(data)-10] // truncate

	tmp := t.TempDir() + "/trunc.ivfh"
	os.WriteFile(tmp, data, 0644)
	_, err := LoadIVFHIndex(tmp, 1)
	if err == nil {
		t.Error("expected error for truncated file, got nil")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/snow/workspace/rinha-backend/gopher-fraud-detection
go test ./internal/search/ -run TestLoadIVFHIndex -v 2>&1 | head -20
```

Expected: `undefined: IVFHIndex` or `undefined: LoadIVFHIndex`

- [ ] **Step 3: Add IVFHIndex struct and LoadIVFHIndex to index.go**

Append to `internal/search/index.go` (after the existing `Index` interface):

```go
const ivfhMagic = "IVFH"

// IVFHIndex is a 2-level hierarchical IVF index (IVF_H2 format).
// Binary format: see docs/superpowers/specs/2026-06-04-ivfh2-precision-redesign.md
// MacroCentroids, MicroCentroids, Radii are parsed into Go slices at load time.
// Vectors and Labels are zero-copy views into the mmap'd file.
type IVFHIndex struct {
	K1, K2, N    int
	DSafe        float32 // L2 dist threshold (stored as L2, not L2²)
	DSafeSq      float32 // DSafe*DSafe — pre-computed at load time
	NCoarseProbe int     // top macro clusters to probe
	MacroCentroids []float32 // K1×16
	MicroCentroids []float32 // K1×K2×16, indexed by [macro*K2+micro]*16
	Starts         []uint32  // K1×K2
	Sizes          []uint32  // K1×K2
	Radii          []float32 // K1×K2, max L2 dist from centroid to any vector
	Vectors        []int16   // N×16 (zero-copy mmap view)
	Labels         []uint8   // N    (zero-copy mmap view)
	mmap           []byte
}

// Close unmaps the index file. Call at process shutdown.
func (idx *IVFHIndex) Close() {
	if idx.mmap != nil {
		_ = syscall.Munmap(idx.mmap)
		idx.mmap = nil
	}
}

// LoadIVFHIndex mmaps path and parses the IVF_H2 binary format.
// nCoarseProbe is the number of macro clusters to probe per query.
func LoadIVFHIndex(path string, nCoarseProbe int) (*IVFHIndex, error) {
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

	if err := parseIVFH(data); err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}

	dSafe := math.Float32frombits(binary.LittleEndian.Uint32(data[4:8]))
	k1 := int(binary.LittleEndian.Uint32(data[8:12]))
	k2 := int(binary.LittleEndian.Uint32(data[12:16]))
	n := int(binary.LittleEndian.Uint32(data[16:20]))
	nLeaf := k1 * k2

	off := 20

	macroCentroids := make([]float32, k1*dims)
	for i := range macroCentroids {
		macroCentroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	microCentroids := make([]float32, nLeaf*dims)
	for i := range microCentroids {
		microCentroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	starts := make([]uint32, nLeaf)
	for i := range starts {
		starts[i] = binary.LittleEndian.Uint32(data[off:])
		off += 4
	}

	sizes := make([]uint32, nLeaf)
	for i := range sizes {
		sizes[i] = binary.LittleEndian.Uint32(data[off:])
		off += 4
	}

	radii := make([]float32, nLeaf)
	for i := range radii {
		radii[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	vecBytes := data[off : off+n*dims*2]
	labelsBytes := data[off+n*dims*2 : off+n*dims*2+n]

	vectors := unsafe.Slice((*int16)(unsafe.Pointer(&vecBytes[0])), n*dims)
	labels := labelsBytes

	return &IVFHIndex{
		K1: k1, K2: k2, N: n,
		DSafe:          dSafe,
		DSafeSq:        dSafe * dSafe,
		NCoarseProbe:   nCoarseProbe,
		MacroCentroids: macroCentroids,
		MicroCentroids: microCentroids,
		Starts:         starts,
		Sizes:          sizes,
		Radii:          radii,
		Vectors:        vectors,
		Labels:         labels,
		mmap:           data,
	}, nil
}

func parseIVFH(data []byte) error {
	if len(data) < 20 {
		return fmt.Errorf("ivfh: file too small: %d bytes", len(data))
	}
	if string(data[0:4]) != ivfhMagic {
		return fmt.Errorf("ivfh: bad magic: %q (want %q)", data[0:4], ivfhMagic)
	}
	k1 := int(binary.LittleEndian.Uint32(data[8:12]))
	k2 := int(binary.LittleEndian.Uint32(data[12:16]))
	n := int(binary.LittleEndian.Uint32(data[16:20]))
	if n == 0 {
		return fmt.Errorf("ivfh: index has zero vectors")
	}
	nLeaf := k1 * k2
	expected := 20 +
		k1*dims*4 +    // macro_centroids
		nLeaf*dims*4 + // micro_centroids
		nLeaf*4 +      // starts
		nLeaf*4 +      // sizes
		nLeaf*4 +      // radii
		n*dims*2 +     // vectors (int16)
		n              // labels (uint8)
	if len(data) != expected {
		return fmt.Errorf("ivfh: size mismatch: got %d, want %d", len(data), expected)
	}
	return nil
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
go test ./internal/search/ -run TestLoadIVFHIndex -v
```

Expected: all three tests PASS

- [ ] **Step 5: Commit**

```bash
git add internal/search/index.go internal/search/index_ivfh_test.go
git commit -m "feat(search): add IVFHIndex struct and LoadIVFHIndex for IVF_H2 format"
```

---

## Task 2: IVFHIndex.KNN — Phase 1 + Phase 2 (centroid scanning)

**Files:**
- Modify: `internal/search/knn.go`
- Create: `internal/search/knn_ivfh_test.go`

Goal: implement macro centroid scan (Phase 1) + micro centroid accumulation (Phase 2), sorted ascending for centroid pruning. Phase 3 is stubbed as `return 0`.

- [ ] **Step 1: Write the failing tests**

Create `internal/search/knn_ivfh_test.go`:

```go
package search

import (
	"testing"
)

// makeTestIVFHIndex builds a minimal 2-level index for unit testing.
// K1=2 macro clusters, K2=2 micro clusters each → 4 leaf clusters.
// Layout:
//   macro 0: centroid at (0,0,...,0)
//     micro 0,0: centroid (0,0,...), 1 vector: labels[0]=legit
//     micro 0,1: centroid (0.2,...), 1 vector: labels[1]=fraud
//   macro 1: centroid at (1,0,...,0)
//     micro 1,0: centroid (0.8,...), 1 vector: labels[2]=fraud
//     micro 1,1: centroid (1.0,...), 1 vector: labels[3]=fraud
// DSafe=0.05 (tight), so DSafeSq=0.0025
func makeTestIVFHIndex() *IVFHIndex {
	K1, K2, N := 2, 2, 4
	macroCentroids := make([]float32, K1*16)
	// macro 0: (0,...), macro 1: (1,0,...)
	macroCentroids[1*16] = 1.0

	microCentroids := make([]float32, K1*K2*16)
	// micro 0,0 → cluster 0: (0,...) — already zero
	// micro 0,1 → cluster 1: (0.2,0,...)
	microCentroids[1*16] = 0.2
	// micro 1,0 → cluster 2: (0.8,0,...)
	microCentroids[2*16] = 0.8
	// micro 1,1 → cluster 3: (1.0,0,...)
	microCentroids[3*16] = 1.0

	vectors := make([]int16, N*16)
	// vec 0 → (0,0,...) → int16 all zeros
	// vec 1 → (0.2,0,...) → int16: 2000 at dim 0
	vectors[1*16] = 2000
	// vec 2 → (0.8,0,...) → 8000
	vectors[2*16] = 8000
	// vec 3 → (1.0,0,...) → 10000
	vectors[3*16] = 10000

	dSafe := float32(0.05)
	return &IVFHIndex{
		K1: K1, K2: K2, N: N,
		DSafe:          dSafe,
		DSafeSq:        dSafe * dSafe,
		NCoarseProbe:   1, // probe only top-1 macro cluster
		MacroCentroids: macroCentroids,
		MicroCentroids: microCentroids,
		Starts:         []uint32{0, 1, 2, 3},
		Sizes:          []uint32{1, 1, 1, 1},
		Radii:          []float32{0.01, 0.01, 0.01, 0.01},
		Vectors:        vectors,
		Labels:         []uint8{0, 1, 1, 1},
	}
}

func TestIVFHKNN_AllFraud(t *testing.T) {
	// NCoarseProbe=2 → probes both macros → sees all 4 vecs
	idx := makeTestIVFHIndex()
	idx.NCoarseProbe = 2
	// query near (1,0,...) → top-5 from N=4 → all 4, 3 fraud
	query := [16]float32{0.9}
	got := idx.KNN(query, 5)
	// 3 of 4 vecs are fraud (labels 1,2,3)
	if got != 3 {
		t.Errorf("got fraudCount=%d, want 3", got)
	}
}

func TestIVFHKNN_AllLegit(t *testing.T) {
	idx := makeTestIVFHIndex()
	idx.NCoarseProbe = 2
	// query near (0,...) → nearest vecs are 0 (legit), then 1 (fraud), etc.
	// With k=5 but only N=4 total: 1 legit out of 4
	query := [16]float32{0.0}
	got := idx.KNN(query, 5)
	// vec order by dist to (0,...): vec0 d²=0, vec1 d²=0.04, vec2 d²=0.64, vec3 d²=1.0
	// top-4 = {0=legit,1=fraud,2=fraud,3=fraud} → fraudCount=3
	if got != 3 {
		t.Errorf("got fraudCount=%d, want 3", got)
	}
}

func TestIVFHKNN_MacroRouting(t *testing.T) {
	// NCoarseProbe=1 → only top macro cluster is probed
	// query near macro 0 → only probes macro 0 → sees vecs 0,1
	// vec 0=legit, vec 1=fraud → k=2 → fraudCount=1
	idx := makeTestIVFHIndex()
	idx.NCoarseProbe = 1
	query := [16]float32{0.1}
	got := idx.KNN(query, 2)
	if got != 1 {
		t.Errorf("MacroRouting: got %d, want 1", got)
	}
}

func TestIVFHKNN_FastExitAllFraud(t *testing.T) {
	// fraudCount==5 fast exit: build index with 5 identical fraud vecs
	K1, K2 := 1, 1
	vecs := make([]int16, 5*16)
	for i := range vecs {
		vecs[i] = 10000
	}
	idx := &IVFHIndex{
		K1: K1, K2: K2, N: 5,
		DSafe: 0.01, DSafeSq: 0.0001,
		NCoarseProbe:   1,
		MacroCentroids: make([]float32, 16),
		MicroCentroids: make([]float32, 16),
		Starts:         []uint32{0},
		Sizes:          []uint32{5},
		Radii:          []float32{0.01},
		Vectors:        vecs,
		Labels:         []uint8{1, 1, 1, 1, 1},
	}
	query := [16]float32{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}
	got := idx.KNN(query, 5)
	if got != 5 {
		t.Errorf("FastExitAllFraud: got %d, want 5", got)
	}
}

func TestIVFHKNN_FastExitDSafe(t *testing.T) {
	// DSafe fast exit: query is clearly legit (maxDist < DSafeSq after nprobeInit clusters)
	K1, K2 := 1, 1
	vecs := make([]int16, 5*16)
	// 5 legit vecs at (0,...) → dist to query (0,...) = 0
	idx := &IVFHIndex{
		K1: K1, K2: K2, N: 5,
		DSafe: 10.0, DSafeSq: 100.0, // very large DSafe → always exits
		NCoarseProbe:   1,
		MacroCentroids: make([]float32, 16),
		MicroCentroids: make([]float32, 16),
		Starts:         []uint32{0},
		Sizes:          []uint32{5},
		Radii:          []float32{0.01},
		Vectors:        vecs,
		Labels:         []uint8{0, 0, 0, 0, 0},
	}
	query := [16]float32{}
	got := idx.KNN(query, 5)
	if got != 0 {
		t.Errorf("FastExitDSafe: got %d, want 0", got)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./internal/search/ -run TestIVFHKNN -v 2>&1 | head -10
```

Expected: `*IVFHIndex has no field or method KNN`

- [ ] **Step 3: Add KNN stub (Phase 1+2 only) to knn.go**

Add at the bottom of `internal/search/knn.go`:

```go
const (
	nCoarseProbeSubseq = 4  // macro clusters to probe for subsequent_tx
	nCoarseProbeFirst  = 3  // macro clusters to probe for first_tx
	nprobeInit         = 8  // micro clusters: fast path
	nprobeMax          = 20 // micro clusters: repair path
)

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// KNN finds k nearest neighbors in the IVF_H2 hierarchical index.
// Phase 1: scan K1 macro centroids, select NCoarseProbe best.
// Phase 2: for each top macro, scan K2 micro centroids; accumulate nprobeMax best.
// Phase 3: scan vectors in topMicro[:nprobeInit] (fast path), then adaptive repair.
func (idx *IVFHIndex) KNN(query [16]float32, k int) int {
	// Extract query to locals — avoids repeated bounds checks.
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
	q14 := query[14]
	q15 := query[15]

	nCoarse := min(idx.NCoarseProbe, idx.K1)
	nMicro := min(nprobeMax, idx.K1*idx.K2)

	// --- Phase 1: top-nCoarse macro clusters ---
	var topMacroArr [nCoarseProbeSubseq]centEntry // max NCoarseProbe is nCoarseProbeSubseq=4
	topMacro := topMacroArr[:0]
	maxMacroD := float32(0)
	maxMacroP := 0

	cents := idx.MacroCentroids
	for c, base := 0, 0; c < idx.K1; c, base = c+1, base+dims {
		_ = cents[base+15]
		d0 := q0 - cents[base]
		d1 := q1 - cents[base+1]
		d2 := q2 - cents[base+2]
		d3 := q3 - cents[base+3]
		d4 := q4 - cents[base+4]
		d5 := q5 - cents[base+5]
		d6 := q6 - cents[base+6]
		d7 := q7 - cents[base+7]
		d8 := q8 - cents[base+8]
		d9 := q9 - cents[base+9]
		d10 := q10 - cents[base+10]
		d11 := q11 - cents[base+11]
		d12 := q12 - cents[base+12]
		d13 := q13 - cents[base+13]
		d14 := q14 - cents[base+14]
		d15 := q15 - cents[base+15]
		d := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13 +
			d14*d14 + d15*d15

		if len(topMacro) < nCoarse {
			topMacro = append(topMacro, centEntry{d, c})
			if len(topMacro) == nCoarse {
				maxMacroD, maxMacroP = centFindMax(topMacro)
			}
		} else if d < maxMacroD {
			topMacro[maxMacroP] = centEntry{d, c}
			maxMacroD, maxMacroP = centFindMax(topMacro)
		}
	}

	// --- Phase 2: top-nMicro micro clusters across selected macros ---
	var topMicroArr [nprobeMax]centEntry
	topMicro := topMicroArr[:0]
	maxMicroD := float32(0)
	maxMicroP := 0

	mcents := idx.MicroCentroids
	for _, me := range topMacro {
		macroBase := me.id * idx.K2
		for j := 0; j < idx.K2; j++ {
			base := (macroBase + j) * dims
			_ = mcents[base+15]
			d0 := q0 - mcents[base]
			d1 := q1 - mcents[base+1]
			d2 := q2 - mcents[base+2]
			d3 := q3 - mcents[base+3]
			d4 := q4 - mcents[base+4]
			d5 := q5 - mcents[base+5]
			d6 := q6 - mcents[base+6]
			d7 := q7 - mcents[base+7]
			d8 := q8 - mcents[base+8]
			d9 := q9 - mcents[base+9]
			d10 := q10 - mcents[base+10]
			d11 := q11 - mcents[base+11]
			d12 := q12 - mcents[base+12]
			d13 := q13 - mcents[base+13]
			d14 := q14 - mcents[base+14]
			d15 := q15 - mcents[base+15]
			d := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
				d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13 +
				d14*d14 + d15*d15

			leafID := macroBase + j
			if len(topMicro) < nMicro {
				topMicro = append(topMicro, centEntry{d, leafID})
				if len(topMicro) == nMicro {
					maxMicroD, maxMicroP = centFindMax(topMicro)
				}
			} else if d < maxMicroD {
				topMicro[maxMicroP] = centEntry{d, leafID}
				maxMicroD, maxMicroP = centFindMax(topMicro)
			}
		}
	}

	// Sort topMicro ascending by dist (needed for centroid pruning in Phase 3).
	// nprobeMax=20 entries: insertion sort is optimal for n≤20.
	n := len(topMicro)
	for i := 1; i < n; i++ {
		key := topMicro[i]
		j := i - 1
		for j >= 0 && topMicro[j].dist > key.dist {
			topMicro[j+1] = topMicro[j]
			j--
		}
		topMicro[j+1] = key
	}

	// --- Phase 3: vector scan ---
	return ivfhScanVectors(idx, topMicro, &query, k, q0)
}

// ivfhScanVectors scans the micro clusters in topMicro[:nprobeInit] (fast path),
// then applies adaptive nprobe and centroid pruning for the repair path.
func ivfhScanVectors(idx *IVFHIndex, topMicro []centEntry, query *[16]float32, k int, q0 float32) int {
	var topArr [5]knnEntry
	top := topArr[:0]
	maxDist := float32(0)
	maxPos := 0

	vecs := idx.Vectors
	labs := idx.Labels

	fastEnd := min(nprobeInit, len(topMicro))

	for _, ce := range topMicro[:fastEnd] {
		start := int(idx.Starts[ce.id])
		size := int(idx.Sizes[ce.id])
		base := start * dims

		for vi := start; vi < start+size; vi, base = vi+1, base+dims {
			_ = vecs[base+15]
			d0 := q0 - float32(vecs[base])*invScale
			if len(top) == k && d0*d0 >= maxDist {
				continue
			}
			dist := distL2i16_16(vecs, base, query)
			if len(top) < k {
				top = append(top, knnEntry{dist, labs[vi]})
				if len(top) == k {
					maxDist, maxPos = knnFindMax(top)
				}
			} else if dist < maxDist {
				top[maxPos] = knnEntry{dist, labs[vi]}
				maxDist, maxPos = knnFindMax(top)
			}
		}
	}

	// Fast exit after nprobeInit clusters.
	if len(top) == k {
		fraudCount := countFraudH(top)
		if fraudCount == 5 {
			return 5 // all fraud — can't change with more probes
		}
		if fraudCount == 0 && maxDist < idx.DSafeSq {
			return 0 // clearly legit — within DSafe confidence radius
		}
	}

	// Repair path: probe topMicro[nprobeInit:] with centroid pruning.
	for _, ce := range topMicro[fastEnd:] {
		dCentroid := sqrt32(ce.dist)
		radius := idx.Radii[ce.id]
		lowerBound := dCentroid - radius
		if lowerBound > 0 && lowerBound*lowerBound > maxDist {
			break // triangle inequality: no vector in this cluster can improve top-k
		}

		start := int(idx.Starts[ce.id])
		size := int(idx.Sizes[ce.id])
		base := start * dims

		for vi := start; vi < start+size; vi, base = vi+1, base+dims {
			_ = vecs[base+15]
			d0 := q0 - float32(vecs[base])*invScale
			if len(top) == k && d0*d0 >= maxDist {
				continue
			}
			dist := distL2i16_16(vecs, base, query)
			if len(top) < k {
				top = append(top, knnEntry{dist, labs[vi]})
				if len(top) == k {
					maxDist, maxPos = knnFindMax(top)
				}
			} else if dist < maxDist {
				top[maxPos] = knnEntry{dist, labs[vi]}
				maxDist, maxPos = knnFindMax(top)
			}
		}
	}

	return countFraudH(top)
}

func countFraudH(entries []knnEntry) int {
	n := 0
	for _, e := range entries {
		if e.label == 1 {
			n++
		}
	}
	return n
}
```

Note: you must add `"math"` to imports in `knn.go` if not already present.

- [ ] **Step 4: Run tests**

```bash
go test ./internal/search/ -run TestIVFHKNN -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Verify existing IVFIndex tests still pass**

```bash
go test ./internal/search/ -v 2>&1 | tail -20
```

Expected: all existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add internal/search/knn.go internal/search/knn_ivfh_test.go
git commit -m "feat(search): implement IVFHIndex.KNN with 3-phase hierarchical search"
```

---

## Task 3: Service Routing — Dual Index

**Files:**
- Modify: `internal/service/fraud_detection.go`
- Create: `internal/service/fraud_detection_ivfh_test.go`

The service currently uses a single global `Idx *search.IVFIndex`. Replace it with two `search.Index` globals (`FirstTxIdx`, `SubseqIdx`) and route by `req.LastTransaction == nil`.

Preserve `fastPath` and `RawTreePredict` — these are not removed by the spec; they short-circuit expensive calls.

- [ ] **Step 1: Write the failing test**

Create `internal/service/fraud_detection_ivfh_test.go`:

```go
package service

import (
	"testing"

	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
)

// stubIndex returns a fixed fraudCount regardless of query.
type stubIndex struct{ count int }

func (s *stubIndex) KNN(_ [16]float32, _ int) int { return s.count }

func TestRouting_NilLastTx_UsesFirstIdx(t *testing.T) {
	FirstTxIdx = &stubIndex{count: 4} // fraud
	SubseqIdx = &stubIndex{count: 0}  // legit
	_ = search.Index(FirstTxIdx)

	req := dto.FraudRequest{}
	// LastTx is nil (zero value for *LastTransaction) → routes to FirstTxIdx → fraudCount=4
	got := CalculateFraudScore(req)
	if got != 4 {
		t.Errorf("nil LastTx routing: got %d, want 4", got)
	}
}

func TestRouting_NonNilLastTx_UsesSubseqIdx(t *testing.T) {
	FirstTxIdx = &stubIndex{count: 0}  // legit
	SubseqIdx = &stubIndex{count: 3}   // fraud

	lastTx := &dto.LastTransaction{}
	req := dto.FraudRequest{LastTx: lastTx}
	// LastTx is non-nil → should route to SubseqIdx → fraudCount=3
	got := CalculateFraudScore(req)
	if got != 3 {
		t.Errorf("nonNil LastTx routing: got %d, want 3", got)
	}
}
```

- [ ] **Step 2: Check dto.LastTransaction type**

```bash
grep -n "LastTransaction" /home/snow/workspace/rinha-backend/gopher-fraud-detection/internal/dto/fraud.go | head -5
```

Verify `LastTransaction` field type and adjust the stub test if needed (it may be a value type embedded in the request, not a pointer — check the actual dto definition).

- [ ] **Step 3: Run test to verify it fails**

```bash
go test ./internal/service/ -run TestRouting -v 2>&1 | head -15
```

Expected: compile error — `FirstTxIdx` undefined or wrong type.

- [ ] **Step 4: Rewrite fraud_detection.go**

Replace `internal/service/fraud_detection.go` entirely:

```go
package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	FirstTxIdx search.Index
	SubseqIdx  search.Index
	Vec        *vectorizer.Vectorizer
)

// CalculateFraudScore returns fraudCount (0–5): fraud neighbors among k=5.
// Routes by LastTx: nil → firstTx index, non-nil → subsequent index.
// fastPath and RawTreePredict short-circuit before the expensive IVF call.
func CalculateFraudScore(req dto.FraudRequest) int {
	if count, ok := fastPath(req); ok {
		return count
	}
	vec := Vec.Vectorize(req)
	if count, ok := RawTreePredict(vec); ok {
		return count
	}
	if req.LastTx == nil {
		return FirstTxIdx.KNN(vec, 5)
	}
	return SubseqIdx.KNN(vec, 5)
}
```

Note: `Idx` global is removed. If other files reference `service.Idx`, update them too (check `cmd/api/main.go` — Task 4 handles it).

- [ ] **Step 5: Run tests**

```bash
go test ./internal/service/ -run TestRouting -v
```

Expected: both routing tests PASS

- [ ] **Step 6: Verify all service tests still pass**

```bash
go test ./internal/service/ -v 2>&1 | tail -20
```

Expected: all existing tests still PASS (fastPath, rawTree, etc.)

- [ ] **Step 7: Commit**

```bash
git add internal/service/fraud_detection.go internal/service/fraud_detection_ivfh_test.go
git commit -m "feat(service): route by LastTransaction to dual IVFH indexes"
```

---

## Task 4: main.go — Load Two IVFH Indexes

**Files:**
- Modify: `cmd/api/main.go`

- [ ] **Step 1: Update main.go**

Replace `cmd/api/main.go`:

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
	firstIdxPath := envOr("FIRST_TX_INDEX_PATH", "index/first_tx.ivfh")
	subseqIdxPath := envOr("SUBSEQ_INDEX_PATH", "index/subsequent_tx.ivfh")
	normPath := envOr("NORM_PATH", "resources/normalization.json")
	mccPath := envOr("MCC_PATH", "resources/mcc_risk.json")

	vec, err := vectorizer.Load(normPath, mccPath)
	if err != nil {
		log.Fatalf("load vectorizer: %v", err)
	}

	firstIdx, err := search.LoadIVFHIndex(firstIdxPath, search.NCoarseProbeFirst)
	if err != nil {
		log.Fatalf("load first_tx index: %v", err)
	}
	defer firstIdx.Close()

	subseqIdx, err := search.LoadIVFHIndex(subseqIdxPath, search.NCoarseProbeSubseq)
	if err != nil {
		log.Fatalf("load subsequent_tx index: %v", err)
	}
	defer subseqIdx.Close()

	service.Vec = vec
	service.FirstTxIdx = firstIdx
	service.SubseqIdx = subseqIdx

	log.Printf("loaded first_tx: %d vectors, subsequent_tx: %d vectors", firstIdx.N, subseqIdx.N)

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

- [ ] **Step 2: Export the nCoarseProbe constants from knn.go**

The `main.go` references `search.NCoarseProbeFirst` and `search.NCoarseProbeSubseq`. Make them exported in `internal/search/knn.go`:

Change in `knn.go`:
```go
// Before (unexported):
const (
	nCoarseProbeSubseq = 4
	nCoarseProbeFirst  = 3
	...
)

// After (exported):
const (
	NCoarseProbeSubseq = 4  // macro clusters to probe for subsequent_tx
	NCoarseProbeFirst  = 3  // macro clusters to probe for first_tx
	nprobeInit         = 8
	nprobeMax          = 20
)
```

Also update the reference inside `KNN`:
```go
// Change:
nCoarse := min(idx.NCoarseProbe, idx.K1)
// (no change needed there — NCoarseProbe is read from the struct, not the const)
```

And the array size in `KNN` Phase 1 uses `nCoarseProbeSubseq` (now `NCoarseProbeSubseq`):
```go
var topMacroArr [NCoarseProbeSubseq]centEntry
```

- [ ] **Step 3: Build to verify compilation**

```bash
cd /home/snow/workspace/rinha-backend/gopher-fraud-detection
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /tmp/fraud-api ./cmd/api 2>&1
```

Expected: successful build, no errors.

- [ ] **Step 4: Run full test suite**

```bash
go test ./... 2>&1
```

Expected: all packages PASS (indexes don't exist yet — startup-only code isn't tested by unit tests)

- [ ] **Step 5: Commit**

```bash
git add cmd/api/main.go internal/search/knn.go
git commit -m "feat(cmd): load dual IVFH indexes for first_tx and subsequent_tx routing"
```

---

## Task 5: Python build_index.py Rewrite

**Files:**
- Modify: `ml/build_index.py`

This is the GPU index builder. No TDD. Output: `index/first_tx.ivfh` and `index/subsequent_tx.ivfh`.

- [ ] **Step 1: Rewrite ml/build_index.py**

Replace `ml/build_index.py` entirely:

```python
import gzip
import json
import struct
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.cluster import KMeans
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

# 2-level IVF hyperparameters
K1_FIRST = 64    # macro clusters, first_tx index
K2_FIRST = 32    # micro clusters per macro, first_tx index
K1_SUBSEQ = 128  # macro clusters, subsequent_tx index
K2_SUBSEQ = 32   # micro clusters per macro, subsequent_tx index

DIMS = 16        # padded feature dims (14 features + 2 zero-padding)
SCALE = 10000    # int16 quantization factor


def detect_null_tx_mask(vectors: np.ndarray) -> np.ndarray:
    """True where both dim5 and dim6 equal -1.0 (sentinel for null last_transaction)."""
    return (vectors[:, 5] == -1.0) & (vectors[:, 6] == -1.0)


def boundary_oversample(vectors: np.ndarray, labels: np.ndarray,
                        sample_size: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """Return augmented (vectors, labels) with boundary vectors duplicated 3x."""
    n = len(labels)
    sample_idx = np.random.choice(n, min(sample_size, n), replace=False)
    sample = vectors[sample_idx]
    sample_labels = labels[sample_idx]

    sample_gpu = cp.asarray(sample, dtype=cp.float32)
    vectors_gpu = cp.asarray(vectors, dtype=cp.float32)

    nbrs = cuNearestNeighbors(n_neighbors=5, algorithm='brute',
                               metric='euclidean', output_type='numpy')
    nbrs.fit(vectors_gpu)
    _, indices = nbrs.kneighbors(sample_gpu)

    fraud_counts = labels[indices].sum(axis=1)
    boundary_mask = (fraud_counts == 2) | (fraud_counts == 3)
    boundary_vecs = sample[boundary_mask]
    boundary_labs = sample_labels[boundary_mask]

    print(f"  Boundary vectors (fc=2 or 3): {boundary_mask.sum()} → duplicated 3×")

    vectors_aug = np.concatenate([vectors, boundary_vecs, boundary_vecs, boundary_vecs])
    labels_aug = np.concatenate([labels, boundary_labs, boundary_labs, boundary_labs])
    return vectors_aug, labels_aug


def two_level_kmeans(vectors: np.ndarray, labels: np.ndarray,
                     vectors_aug: np.ndarray,
                     K1: int, K2: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit 2-level KMeans on augmented vectors, assign original vectors.
    Returns: macro_centroids (K1×DIMS), micro_centroids (K1*K2×DIMS),
             micro_assignments (N,) — flat leaf index for each original vector.
    """
    N = len(labels)
    vectors_gpu = cp.asarray(vectors, dtype=cp.float32)
    vectors_gpu_aug = cp.asarray(vectors_aug, dtype=cp.float32)

    print(f"  Level-1 KMeans: K1={K1}, fit on {len(vectors_aug)} aug vecs...")
    km1 = KMeans(n_clusters=K1, init='scalable-k-means++', n_init=10,
                 max_iter=300, random_state=42, output_type='numpy')
    km1.fit(vectors_gpu_aug)
    macro_centroids = km1.cluster_centers_.astype(np.float32)  # (K1, DIMS)

    macro_assignments_orig = km1.predict(vectors_gpu).astype(np.int32)  # (N,)

    micro_centroids = np.zeros((K1, K2, DIMS), dtype=np.float32)
    micro_assignments = np.zeros(N, dtype=np.int32)

    aug_labels_aug = km1.predict(vectors_gpu_aug).astype(np.int32)

    for i in range(K1):
        mask_aug = aug_labels_aug == i
        vecs_aug_i = vectors_gpu_aug[mask_aug]
        if len(vecs_aug_i) < K2:
            # degenerate macro cluster: assign all to micro 0
            micro_assignments[macro_assignments_orig == i] = i * K2
            continue
        km2 = KMeans(n_clusters=K2, init='scalable-k-means++', n_init=5,
                     max_iter=200, random_state=42, output_type='numpy')
        km2.fit(vecs_aug_i)
        micro_centroids[i] = km2.cluster_centers_

        mask_orig = macro_assignments_orig == i
        orig_in_macro = cp.asarray(vectors[mask_orig], dtype=cp.float32)
        if len(orig_in_macro) > 0:
            local_assign = km2.predict(orig_in_macro).astype(np.int32)
            micro_assignments[mask_orig] = i * K2 + local_assign

        if (i + 1) % 16 == 0:
            print(f"    macro {i+1}/{K1} done")

    micro_centroids_flat = micro_centroids.reshape(K1 * K2, DIMS)
    return macro_centroids, micro_centroids_flat, micro_assignments


def balanced_kmeans(vectors: np.ndarray, micro_centroids_flat: np.ndarray,
                    micro_assignments: np.ndarray, K1: int, K2: int) -> np.ndarray:
    """Reassign overflow vectors (>1.5× avg leaf size) to nearest sibling with capacity."""
    N_leaves = K1 * K2
    cluster_sizes = np.bincount(micro_assignments, minlength=N_leaves).astype(np.int32)
    avg_size = len(vectors) / N_leaves
    max_size = int(avg_size * 1.5)
    print(f"  Balanced k-means: avg={avg_size:.0f}, cap={max_size}")

    for macro_id in range(K1):
        for micro_local in range(K2):
            c = macro_id * K2 + micro_local
            if cluster_sizes[c] <= max_size:
                continue
            idxs = np.where(micro_assignments == c)[0]
            cent = micro_centroids_flat[c]
            dists = np.linalg.norm(vectors[idxs] - cent, axis=1)
            n_overflow = cluster_sizes[c] - max_size
            overflow_idxs = idxs[np.argsort(-dists)[:n_overflow]]

            for v in overflow_idxs:
                siblings = [macro_id * K2 + j for j in range(K2)
                            if j != micro_local and cluster_sizes[macro_id * K2 + j] < max_size]
                if not siblings:
                    continue
                sib_cents = micro_centroids_flat[[s for s in siblings]]
                nearest_sib = siblings[int(np.argmin(
                    np.linalg.norm(sib_cents - vectors[v], axis=1)))]
                micro_assignments[v] = nearest_sib
                cluster_sizes[c] -= 1
                cluster_sizes[nearest_sib] += 1

    return micro_assignments


def compute_dsafe(vectors: np.ndarray, labels: np.ndarray) -> float:
    """
    D_safe = 99th percentile of L2 dist-to-5th-neighbor for samples whose
    brute-force k=5 gives fraudCount==0. Stored as L2 (not L2²).
    """
    legit_idxs = np.where(labels == 0)[0]
    sample_idx = legit_idxs[:min(10_000, len(legit_idxs))]
    sample_vecs = cp.asarray(vectors[sample_idx], dtype=cp.float32)

    nbrs = cuNearestNeighbors(n_neighbors=5, algorithm='brute',
                               metric='euclidean', output_type='numpy')
    nbrs.fit(cp.asarray(vectors, dtype=cp.float32))
    dists_sq, neighbor_idx = nbrs.kneighbors(sample_vecs)
    # dists_sq shape: (sample_size, 5) — squared euclidean from cuML

    neighbor_labels = labels[neighbor_idx]
    fraud_counts = neighbor_labels.sum(axis=1)
    truly_legit = fraud_counts == 0

    max_dists = np.sqrt(dists_sq[truly_legit, 4])  # L2 to 5th neighbor
    d_safe = float(np.percentile(max_dists, 99))
    print(f"  D_safe = {d_safe:.6f} (99th pct of dist-to-5th-neighbor for fraudCount==0)")
    return d_safe


def build_ivfh(vectors: np.ndarray, labels: np.ndarray,
               K1: int, K2: int, dst: Path) -> None:
    N = len(labels)
    print(f"Building IVFH: N={N}, K1={K1}, K2={K2} → {dst.name}")

    # Boundary oversampling
    vectors_aug, _ = boundary_oversample(vectors, labels)

    # 2-level KMeans
    macro_centroids, micro_centroids_flat, micro_assignments = \
        two_level_kmeans(vectors, labels, vectors_aug, K1, K2)

    # Balanced k-means post-processing
    micro_assignments = balanced_kmeans(
        vectors, micro_centroids_flat, micro_assignments, K1, K2)

    # Sort vectors by leaf assignment
    sort_idx = np.argsort(micro_assignments, kind='stable')
    vectors_sorted = vectors[sort_idx]
    labels_sorted = labels[sort_idx]
    micro_assignments_sorted = micro_assignments[sort_idx]

    N_leaves = K1 * K2
    cluster_sizes = np.bincount(micro_assignments_sorted, minlength=N_leaves).astype(np.uint32)
    cluster_starts = np.zeros(N_leaves, dtype=np.uint32)
    cluster_starts[1:] = np.cumsum(cluster_sizes[:-1])

    # Cluster radii: max L2 dist from centroid to any vector in cluster
    cluster_radius = np.zeros(N_leaves, dtype=np.float32)
    for c in range(N_leaves):
        s, sz = int(cluster_starts[c]), int(cluster_sizes[c])
        if sz == 0:
            continue
        vecs_in_c = vectors_sorted[s:s+sz]
        cent = micro_centroids_flat[c]
        dists = np.linalg.norm(vecs_in_c - cent, axis=1)
        cluster_radius[c] = float(dists.max())

    # D_safe
    d_safe = compute_dsafe(vectors, labels)

    # Quantize vectors to int16
    vectors_int16 = np.clip(
        np.round(vectors_sorted * SCALE), -32768, 32767
    ).astype(np.int16)

    # Write IVFH binary
    dst.parent.mkdir(exist_ok=True)
    with open(dst, 'wb') as out:
        out.write(b'IVFH')
        out.write(struct.pack('<f', d_safe))
        out.write(struct.pack('<II', K1, K2))
        out.write(struct.pack('<I', N))
        out.write(macro_centroids.astype('<f4').tobytes())
        out.write(micro_centroids_flat.astype('<f4').tobytes())
        out.write(cluster_starts.astype('<u4').tobytes())
        out.write(cluster_sizes.astype('<u4').tobytes())
        out.write(cluster_radius.astype('<f4').tobytes())
        out.write(vectors_int16.astype('<i2').tobytes())
        out.write(labels_sorted.astype('u1').tobytes())

    size_mb = dst.stat().st_size / 1024 / 1024
    fraud_pct = labels.mean() * 100
    print(f"  → {dst} ({size_mb:.1f} MB), fraud={fraud_pct:.1f}%, D_safe={d_safe:.4f}")


def main():
    root = Path(__file__).parent.parent
    src = root / 'resources' / 'references.json.gz'
    dst_dir = root / 'index'

    print('Loading records...')
    with gzip.open(src) as f:
        records = json.load(f)
    n = len(records)
    print(f'Loaded {n} records')

    # Pad 14-dim features to 16 dims (zeros in dims 14, 15)
    vectors = np.zeros((n, DIMS), dtype=np.float32)
    vectors[:, :14] = np.array([rec['vector'] for rec in records], dtype=np.float32)
    labels = np.array(
        [1 if rec['label'] == 'fraud' else 0 for rec in records], dtype=np.uint8)

    # Split by null last_transaction sentinel (dims 5 and 6 == -1.0)
    null_mask = detect_null_tx_mask(vectors)
    assert not (null_mask & ~null_mask).any(), "overlap check"
    vectors_first, labels_first = vectors[null_mask], labels[null_mask]
    vectors_subseq, labels_subseq = vectors[~null_mask], labels[~null_mask]
    print(f'Split: first_tx={len(labels_first)} ({null_mask.mean()*100:.1f}%), '
          f'subsequent_tx={len(labels_subseq)}')

    build_ivfh(vectors_first, labels_first, K1_FIRST, K2_FIRST,
               dst_dir / 'first_tx.ivfh')
    build_ivfh(vectors_subseq, labels_subseq, K1_SUBSEQ, K2_SUBSEQ,
               dst_dir / 'subsequent_tx.ivfh')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/snow/workspace/rinha-backend/gopher-fraud-detection
uv run python -c "import ast; ast.parse(open('ml/build_index.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add ml/build_index.py
git commit -m "feat(ml): rewrite build_index.py for IVF_H2 dual-split hierarchical index"
```

- [ ] **Step 4: Run index build (requires GPU)**

```bash
uv run ml/build_index.py
```

Expected: creates `index/first_tx.ivfh` and `index/subsequent_tx.ivfh`. Check sizes:
```bash
ls -lh index/*.ivfh
```

Both files should exist and be non-zero. Total size should be ≤ 87MB combined (was 87MB for single unified index; split reduces per-index size but keeps total similar).

---

## Task 6: Update docker-compose.yml and Dockerfile

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile` (if it references INDEX_PATH)

- [ ] **Step 1: Check current env vars**

```bash
grep -n "INDEX_PATH\|index/" /home/snow/workspace/rinha-backend/gopher-fraud-detection/docker-compose.yml
grep -n "INDEX_PATH\|index/" /home/snow/workspace/rinha-backend/gopher-fraud-detection/Dockerfile
```

- [ ] **Step 2: Update docker-compose.yml**

Replace any `INDEX_PATH=index/references.bin` occurrences with:

```yaml
- FIRST_TX_INDEX_PATH=index/first_tx.ivfh
- SUBSEQ_INDEX_PATH=index/subsequent_tx.ivfh
```

The exact lines depend on the current file structure — run the grep in Step 1 first.

- [ ] **Step 3: Update Dockerfile**

If Dockerfile copies the index file, update the COPY path or ADD instruction from `index/references.bin` to `index/*.ivfh` (or keep it generic with `index/`).

- [ ] **Step 4: Build Docker image locally**

```bash
docker build -t fraud-api-test . 2>&1 | tail -10
```

Expected: successful build.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml Dockerfile
git commit -m "chore(docker): update env vars for dual IVFH index paths"
```

---

## Task 7: Update References Bench + PROGRESS.md

**Files:**
- Modify: `references/bench/result.csv` (add new result row after bench run)
- Modify: `PROGRESS.md` (rewrite with current state per CLAUDE.md requirement)

- [ ] **Step 1: Run bench (requires built indexes)**

```bash
make bench
```

Record p99, FP count, FN count, and score from output.

- [ ] **Step 2: Update references/bench/result.csv**

Add a new row with the IVFH result. Format matches existing rows.

- [ ] **Step 3: Rewrite PROGRESS.md**

Rewrite (not append) PROGRESS.md with the current state: index type, K1/K2, D_safe value, NCoarseProbe settings, FP/FN, score, p99. Per CLAUDE.md: "One file, always current — not a changelog, not appended."

- [ ] **Step 4: Commit**

```bash
git add references/bench/result.csv PROGRESS.md
git commit -m "docs(progress): update after IVF_H2 dual-split hierarchical index"
```

---

## Self-Review Against Spec

### Spec coverage check

| Spec section | Covered by |
|---|---|
| Dual index split by null_mask | Task 5 (build_index.py) + Task 3 (service routing) |
| 2-level hierarchical IVF (K1/K2) | Task 5 (build), Task 1 (IVFHIndex struct), Task 2 (KNN Phase 1+2) |
| Boundary oversampling | Task 5 (`boundary_oversample`) |
| Balanced k-means | Task 5 (`balanced_kmeans`) |
| IVF_H2 binary format | Task 1 (parseIVFH + LoadIVFHIndex + writeIVFHBinary) |
| D_safe computation | Task 5 (`compute_dsafe`) |
| cluster_radius | Task 5 (Step 1 in build loop) |
| DSafeSq pre-computed at load | Task 1 (`DSafeSq = dSafe * dSafe`) |
| Phase 1 macro scan | Task 2 (KNN Phase 1) |
| Phase 2 micro scan | Task 2 (KNN Phase 2) |
| Phase 3 fast path (nprobeInit) | Task 2 (`ivfhScanVectors`) |
| Fast exit fraudCount==5 | Task 2 (`if fraudCount == 5`) |
| Fast exit DSafe | Task 2 (`if fraudCount == 0 && maxDist < DSafeSq`) |
| Repair path (nprobeMax) | Task 2 (`for _, ce := range topMicro[fastEnd:]`) |
| Centroid pruning (triangle inequality) | Task 2 (`lowerBound > 0 && lowerBound² > maxDist`) |
| NCoarseProbeFirst=3, NCoarseProbeSubseq=4 | Task 4 (exported constants) |
| Service routing | Task 3 |
| main.go dual load | Task 4 |
| docker-compose env vars | Task 6 |

### Type consistency check

- `IVFHIndex.Radii []float32` ✓ used in `ivfhScanVectors` as `idx.Radii[ce.id]`
- `IVFHIndex.DSafeSq float32` ✓ set to `dSafe * dSafe` in LoadIVFHIndex, read in `ivfhScanVectors`
- `centEntry{dist, id}` ✓ reused from existing knn.go — `id` = leaf index `macro*K2+micro`
- `NCoarseProbeSubseq = 4` ✓ used as array bound `[NCoarseProbeSubseq]centEntry`
- `search.Index` interface: `KNN(query [16]float32, k int) int` ✓ both `IVFIndex` and `IVFHIndex` implement it

### dto.LastTx field (already verified)

The field is `LastTx *LastTransaction` (pointer) in `dto.FraudRequest`. The nil check `req.LastTx == nil` in `CalculateFraudScore` is correct. The spec uses `req.LastTransaction` which is a naming mismatch — the plan uses the actual field name `req.LastTx`.
