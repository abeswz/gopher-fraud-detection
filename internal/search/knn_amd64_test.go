//go:build amd64

package search

import (
	"math"
	"math/rand"
	"testing"
)

// scalarDistL2i16_16 is the reference implementation for testing.
func scalarDistL2i16_16(vecs []int16, base int, q *[16]float32) float32 {
	_ = vecs[base+15]
	var dist float32
	for i := 0; i < 16; i++ {
		d := float32(vecs[base+i])*(1.0/10000.0) - q[i]
		dist += d * d
	}
	return dist
}

func TestDistL2i16_16_ZeroQuery(t *testing.T) {
	vecs := make([]int16, 16)
	for i := range vecs {
		vecs[i] = 5000 // dequantizes to 0.5
	}
	var q [16]float32
	got := distL2i16_16(vecs, 0, &q)
	want := scalarDistL2i16_16(vecs, 0, &q)
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("ZeroQuery: got %v, want %v", got, want)
	}
}

func TestDistL2i16_16_MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	vecs := make([]int16, 32*16) // 32 vectors
	for i := range vecs {
		vecs[i] = int16(rng.Intn(20001) - 10000)
	}
	var q [16]float32
	for i := range q {
		q[i] = float32(rng.Float64()*2 - 1)
	}

	for base := 0; base < 32*16; base += 16 {
		got := distL2i16_16(vecs, base, &q)
		want := scalarDistL2i16_16(vecs, base, &q)
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Errorf("base=%d: got %v, want %v (delta %v)", base, got, want, got-want)
		}
	}
}

func TestDistL2i16_16_SentinelMinus1(t *testing.T) {
	vecs := make([]int16, 16)
	for i := range vecs {
		vecs[i] = -10000 // sentinel: dequantizes to -1.0
	}
	q := [16]float32{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	got := distL2i16_16(vecs, 0, &q)
	want := scalarDistL2i16_16(vecs, 0, &q)
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("Sentinel: got %v, want %v", got, want)
	}
}

func BenchmarkDistL2i16_16_AVX2(b *testing.B) {
	vecs := make([]int16, 16)
	for i := range vecs {
		vecs[i] = int16(i * 500)
	}
	q := [16]float32{0.5, 0.3, 0.7, 0.1, 0.9, -1, -1, 0.4, 0.6, 0.8, 0.2, 0.55, 0.45, 0.65}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = distL2i16_16(vecs, 0, &q)
	}
}

func BenchmarkDistL2i16_16_Scalar(b *testing.B) {
	vecs := make([]int16, 16)
	for i := range vecs {
		vecs[i] = int16(i * 500)
	}
	q := [16]float32{0.5, 0.3, 0.7, 0.1, 0.9, -1, -1, 0.4, 0.6, 0.8, 0.2, 0.55, 0.45, 0.65}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = scalarDistL2i16_16(vecs, 0, &q)
	}
}
