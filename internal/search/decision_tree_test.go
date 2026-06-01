package search

import (
	"testing"
)

func TestDecisionTree_UncertainLeaf(t *testing.T) {
	// Zero vector — guaranteed to reach a leaf; with only 8 confident leaves
	// out of 2069 nodes, almost all paths hit uncertain leaves.
	var vec [16]float32
	_, ok := DecisionTree(vec)
	// We don't assert the class — just that the function runs without panic
	// and the zero vec (which represents an invalid/edge-case transaction)
	// returns a result. If ok=true, the tree is confident; if ok=false, it defers to k-NN.
	_ = ok
}

func TestDecisionTree_ReturnsValidFraudCount(t *testing.T) {
	// Traverse with various inputs and verify invariants:
	// - count must be 0 or 5 when ok=true
	// - count must be 0 when ok=false
	tests := []struct {
		name string
		vec  [16]float32
	}{
		{"zero vector", [16]float32{}},
		{"all ones", [16]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
		{"mid values", [16]float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}},
		{"max legit sentinel", [16]float32{0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			count, ok := DecisionTree(tc.vec)
			if ok {
				if count != 0 && count != 5 {
					t.Errorf("confident leaf returned count=%d, want 0 or 5", count)
				}
			} else {
				if count != 0 {
					t.Errorf("uncertain leaf returned count=%d, want 0", count)
				}
			}
		})
	}
}
