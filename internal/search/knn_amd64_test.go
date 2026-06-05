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
