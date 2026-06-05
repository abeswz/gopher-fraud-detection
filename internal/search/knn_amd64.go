//go:build amd64

package search

// distL2i16_16 computes L2² between vecs[base:base+16] (int16 ×10000) and query q.
// Implemented in knn_amd64.s via AVX2. Caller must ensure base+15 < len(vecs).
//
//go:noescape
func distL2i16_16(vecs []int16, base int, q *[16]float32) float32

// distL2i16q computes L2² entirely in int16/int32 space — no float conversion.
// Returns sum of (vecs[base+i]-q[i])² as int32. Caller ensures base+15 < len(vecs).
//
//go:noescape
func distL2i16q(vecs []int16, base int, q *[16]int16) int32
