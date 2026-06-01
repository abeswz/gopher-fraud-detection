package search

import (
	"testing"
)

// makeTestIVFIndex creates a single-cluster IVFIndex (equivalent to brute-force).
func makeTestIVFIndex(nFraud, nLegit int, fraudVal, legitVal int16) *IVFIndex {
	n := nFraud + nLegit
	vectors := make([]int16, n*16)
	labels := make([]uint8, n)

	for i := 0; i < nFraud; i++ {
		for j := 0; j < 16; j++ {
			vectors[i*16+j] = fraudVal
		}
		labels[i] = 1
	}
	for i := nFraud; i < n; i++ {
		for j := 0; j < 16; j++ {
			vectors[i*16+j] = legitVal
		}
		labels[i] = 0
	}

	return &IVFIndex{
		C:         1,
		N:         n,
		Centroids: make([]float32, 16),
		Starts:    []uint32{0},
		Sizes:     []uint32{uint32(n)},
		Vectors:   vectors,
		Labels:    labels,
	}
}

func TestKNN_AllFraud(t *testing.T) {
	idx := makeTestIVFIndex(5, 5, 10000, 0)
	query := [16]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	got := idx.KNN(query, 5)
	if got != 5 {
		t.Errorf("AllFraud: got %d fraud, want 5", got)
	}
}

func TestKNN_AllLegit(t *testing.T) {
	idx := makeTestIVFIndex(5, 5, 10000, 0)
	query := [16]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	got := idx.KNN(query, 5)
	if got != 0 {
		t.Errorf("AllLegit: got %d fraud, want 0", got)
	}
}

func TestKNN_Mixed(t *testing.T) {
	idx := makeTestIVFIndex(3, 7, 10000, 0)
	query := [16]float32{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}
	got := idx.KNN(query, 5)
	if got != 3 {
		t.Errorf("Mixed: got %d fraud, want 3", got)
	}
}

// TestKNN_IntegerArithmeticPrecision verifies that integer-scaled distance
// computation selects the same nearest neighbors as float32 dequantization.
func TestKNN_IntegerArithmeticPrecision(t *testing.T) {
	// Three clusters: near-zero (legit), near-0.5 (legit), near-1.0 (fraud).
	// Query at 0.95 — should pick fraud cluster as all 5 nearest.
	idx := makeTestIVFIndex(5, 5, 9500, 0)
	query := [16]float32{0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95}
	got := idx.KNN(query, 5)
	if got != 5 {
		t.Errorf("IntegerArithmeticPrecision: got %d fraud, want 5", got)
	}
}

func TestKNN_SentinelHandling(t *testing.T) {
	idx := &IVFIndex{
		C:         1,
		N:         2,
		Centroids: make([]float32, 16),
		Starts:    []uint32{0},
		Sizes:     []uint32{2},
		Vectors:   make([]int16, 2*16),
		Labels:    make([]uint8, 2),
	}
	// Vector 0: -10000 at dims 5,6 (sentinel), 0 elsewhere; legit
	idx.Vectors[5] = -10000
	idx.Vectors[6] = -10000
	// Vector 1: 10000 everywhere; fraud
	for j := 0; j < 16; j++ {
		idx.Vectors[16+j] = 10000
	}
	idx.Labels[1] = 1

	query := [16]float32{0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0}
	got := idx.KNN(query, 1)
	if got != 0 {
		t.Errorf("SentinelHandling: got %d fraud, want 0 (legit nearest)", got)
	}
}

func BenchmarkKNN(b *testing.B) {
	// 3000 vectors: realistic cluster size for nprobe=20 over 3M vectors / 256 clusters
	idx := makeTestIVFIndex(1500, 1500, 8000, 2000)
	query := [16]float32{0.7, 0.3, 0.5, 0.9, 0.1, -1.0, -1.0, 0.4, 0.6, 0.8, 0.2, 0.55, 0.45, 0.65}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.KNN(query, 5)
	}
}
