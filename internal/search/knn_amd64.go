//go:build amd64

package search

// distL2i16_16 computes L2² between vecs[base:base+16] dequantized as float32(x)*(1/10000)
// and query q. Implemented in knn_amd64.s using AVX2 (two YMM passes).
// Caller must ensure base+15 < len(vecs).
//
//go:noescape
func distL2i16_16(vecs []int16, base int, q *[16]float32) float32
