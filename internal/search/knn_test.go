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
	got := idx.KNN(query, 1)
	if got != 0 {
		t.Errorf("SentinelHandling: got %d fraud, want 0 (legit is nearest)", got)
	}
}
