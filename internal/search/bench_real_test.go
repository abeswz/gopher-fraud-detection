package search

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func realIndexPath() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "index", "references.bin")
}

func BenchmarkKNN_RealIndex(b *testing.B) {
	path := realIndexPath()
	if _, err := os.Stat(path); err != nil {
		b.Skipf("real index not found at %s", path)
	}
	idx, err := LoadIVFIndex(path)
	if err != nil {
		b.Skipf("not IVF1 or load error: %v", err)
	}
	b.Logf("C=%d N=%d avg_cluster=%.0f nprobe=%d vecs_per_query=~%.0f",
		idx.C, idx.N, float64(idx.N)/float64(idx.C), nprobe,
		float64(idx.N)/float64(idx.C)*nprobe)

	query := [14]float32{0.7, 0.3, 0.5, 0.9, 0.1, -1.0, -1.0, 0.4, 0.6, 0.8, 0.2, 0.55, 0.45, 0.65}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.KNN(query, 5)
	}
}

func BenchmarkVPKNN_RealIndex(b *testing.B) {
	path := realIndexPath()
	if _, err := os.Stat(path); err != nil {
		b.Skipf("real index not found at %s", path)
	}
	idx, err := LoadVPIndex(path)
	if err != nil {
		b.Skipf("not VPT1 or load error: %v", err)
	}
	b.Logf("N=%d nodes=%d depth_est=%d",
		len(idx.Labels), len(idx.Nodes), treeDepth(len(idx.Nodes)))

	query := [14]float32{0.7, 0.3, 0.5, 0.9, 0.1, -1.0, -1.0, 0.4, 0.6, 0.8, 0.2, 0.55, 0.45, 0.65}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.KNN(query, 5)
	}
}

func treeDepth(nodes int) int {
	d := 0
	for n := nodes + 1; n > 1; n >>= 1 {
		d++
	}
	return d
}
