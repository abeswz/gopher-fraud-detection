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
