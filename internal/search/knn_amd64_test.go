//go:build amd64

package search

import (
	"math/rand"
	"testing"
)

// scalarDistL2i16q is the reference for distL2i16q.
func scalarDistL2i16q(vecs []int16, base int, q *[16]int16) int32 {
	_ = vecs[base+15]
	var dist int32
	for i := 0; i < 16; i++ {
		d := int32(vecs[base+i]) - int32(q[i])
		dist += d * d
	}
	return dist
}

func TestDistL2i16q_ZeroQuery(t *testing.T) {
	vecs := make([]int16, 16)
	for i := range vecs {
		vecs[i] = 5000
	}
	var q [16]int16
	got := distL2i16q(vecs, 0, &q)
	want := scalarDistL2i16q(vecs, 0, &q)
	if got != want {
		t.Errorf("ZeroQuery: got %d, want %d", got, want)
	}
}

func TestDistL2i16q_MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	vecs := make([]int16, 32*16)
	for i := range vecs {
		vecs[i] = int16(rng.Intn(20001) - 10000)
	}
	var q [16]int16
	for i := range q {
		q[i] = int16(rng.Intn(20001) - 10000)
	}
	for base := 0; base < 32*16; base += 16 {
		got := distL2i16q(vecs, base, &q)
		want := scalarDistL2i16q(vecs, base, &q)
		if got != want {
			t.Errorf("base=%d: got %d, want %d", base, got, want)
		}
	}
}

func TestDistL2i16q_SentinelMinus10000(t *testing.T) {
	vecs := make([]int16, 16)
	for i := range vecs {
		vecs[i] = -10000
	}
	q := [16]int16{-10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000,
		-10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000}
	got := distL2i16q(vecs, 0, &q)
	if got != 0 {
		t.Errorf("Sentinel same: got %d, want 0", got)
	}
}

func TestDistL2i16q_RealisticMaxDiff(t *testing.T) {
	// Worst realistic case: 2 sentinel dims (5,6) at max diff, 12 normal dims, 2 padding (14,15)=0.
	// Max total: 2×(20000)²+12×(10000)² = 2e9 < int32_max. Must not overflow.
	vecs := make([]int16, 16)
	var q [16]int16
	for i := 0; i < 14; i++ {
		vecs[i] = 10000
		q[i] = 0
	}
	// sentinel dims 5,6: push to max diff
	vecs[5] = 10000
	q[5] = -10000
	vecs[6] = 10000
	q[6] = -10000

	got := distL2i16q(vecs, 0, &q)
	want := scalarDistL2i16q(vecs, 0, &q)
	if got != want {
		t.Errorf("RealisticMaxDiff: got %d, want %d", got, want)
	}
	// Verify no overflow: sum should be 2×(20000)²+12×(10000)² = 2000000000
	if got != 2000000000 {
		t.Errorf("RealisticMaxDiff: expected 2000000000, got %d", got)
	}
}

func TestComputeClusterBatch8_Zero(t *testing.T) {
	// Query inside all bboxes → all lbs = 0.
	minSoA := make([]int16, NPairs*16)
	maxSoA := make([]int16, NPairs*16)
	for p := 0; p < NPairs; p++ {
		for l := 0; l < 8; l++ {
			minSoA[p*16+l*2] = 0
			minSoA[p*16+l*2+1] = 0
			maxSoA[p*16+l*2] = 5000
			maxSoA[p*16+l*2+1] = 5000
		}
	}
	var q [16]int16
	for i := 0; i < 14; i++ {
		q[i] = 2500 // inside [0, 5000]
	}
	var lbs [8]int32
	computeClusterBatch8(&minSoA[0], &maxSoA[0], &q, &lbs)
	for l := 0; l < 8; l++ {
		if lbs[l] != 0 {
			t.Errorf("cluster %d: got lb=%d, want 0", l, lbs[l])
		}
	}
}

func TestComputeClusterBatch8_BelowBbox(t *testing.T) {
	// q=0, bbox=[5000,5000] for all dims → gap=5000 per dim.
	// lb per cluster = 7 pairs × (5000²+5000²) = 7×50_000_000 = 350_000_000.
	minSoA := make([]int16, NPairs*16)
	maxSoA := make([]int16, NPairs*16)
	for p := 0; p < NPairs; p++ {
		for l := 0; l < 8; l++ {
			minSoA[p*16+l*2] = 5000
			minSoA[p*16+l*2+1] = 5000
			maxSoA[p*16+l*2] = 5000
			maxSoA[p*16+l*2+1] = 5000
		}
	}
	var q [16]int16 // all zeros
	var lbs [8]int32
	computeClusterBatch8(&minSoA[0], &maxSoA[0], &q, &lbs)
	want := int32(NPairs * (5000*5000 + 5000*5000))
	for l := 0; l < 8; l++ {
		if lbs[l] != want {
			t.Errorf("cluster %d: got %d, want %d", l, lbs[l], want)
		}
	}
}

func TestComputeClusterBatch8_MatchesScalar(t *testing.T) {
	// Build a bpsoaMin/bpsoaMax for 8 clusters with varied bboxes.
	// Compare against scalar reference.
	rng := rand.New(rand.NewSource(99))
	minSoA := make([]int16, NPairs*16)
	maxSoA := make([]int16, NPairs*16)
	for i := range minSoA {
		lo := int16(rng.Intn(10001) - 5000)
		hi := lo + int16(rng.Intn(3000))
		minSoA[i] = lo
		maxSoA[i] = hi
	}
	var q [16]int16
	for i := 0; i < 14; i++ {
		q[i] = int16(rng.Intn(20001) - 10000)
	}

	var lbs [8]int32
	computeClusterBatch8(&minSoA[0], &maxSoA[0], &q, &lbs)

	// Scalar reference
	for l := 0; l < 8; l++ {
		var lb int32
		for p := 0; p < NPairs; p++ {
			for d := 0; d < 2; d++ {
				dim := p*2 + d
				qd := int32(q[dim])
				lo := int32(minSoA[p*16+l*2+d])
				hi := int32(maxSoA[p*16+l*2+d])
				var gap int32
				if qd < lo {
					gap = lo - qd
				} else if qd > hi {
					gap = qd - hi
				}
				lb += gap * gap
			}
		}
		if lbs[l] != lb {
			t.Errorf("cluster %d: AVX2=%d scalar=%d", l, lbs[l], lb)
		}
	}
}

func BenchmarkComputeClusterBatch8(b *testing.B) {
	minSoA := make([]int16, NPairs*16)
	maxSoA := make([]int16, NPairs*16)
	for i := range minSoA {
		minSoA[i] = int16(i * 100)
		maxSoA[i] = int16(i*100 + 500)
	}
	var q [16]int16
	for i := range q {
		q[i] = int16(i * 300)
	}
	var lbs [8]int32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		computeClusterBatch8(&minSoA[0], &maxSoA[0], &q, &lbs)
	}
}

func BenchmarkDistL2i16q_AVX2(b *testing.B) {
	vecs := make([]int16, 16)
	for i := range vecs {
		vecs[i] = int16(i * 500)
	}
	q := [16]int16{5000, 3000, 7000, 1000, 9000, -10000, -10000, 4000, 6000, 8000, 2000, 5500, 4500, 6500}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = distL2i16q(vecs, 0, &q)
	}
}
