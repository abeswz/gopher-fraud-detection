//go:build amd64

package search

// distL2i16q computes L2² between 16 int16s starting at vecs[base] and query q.
// Implemented in knn_amd64.s via AVX2 VPMADDWD. Caller ensures base+15 < len(vecs).
//
//go:noescape
func distL2i16q(vecs []int16, base int, q *[16]int16) int32

// computeClusterBatch8 computes bbox lower bounds for 8 clusters in one SIMD pass.
// minSoA and maxSoA point to the start of one group in bpsoaMin/bpsoaMax (NPairs*16 int16s each).
// lbs receives 8 int32 lower bounds; values may overflow (negative) for very distant clusters.
//
//go:noescape
func computeClusterBatch8(minSoA, maxSoA *int16, q *[16]int16, lbs *[8]int32)
