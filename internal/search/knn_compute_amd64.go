//go:build amd64

package search

import "math"

// computeClusterPacked fills out[c] = (bbox_lb << CidBits) | c for c in [0, NClusters).
// Processes 8 clusters per iteration using AVX2 via bpsoaMin/bpsoaMax.
func (ix *IvfIndex) computeClusterPacked(q *[16]int16, out []int64) {
	nGroups := (ix.NClusters + 7) / 8
	var lbs [8]int32
	for g := 0; g < nGroups; g++ {
		off := g * NPairs * 16
		computeClusterBatch8(&ix.bpsoaMin[off], &ix.bpsoaMax[off], q, &lbs)
		base := g * 8
		for l := 0; l < 8; l++ {
			c := base + l
			if c >= ix.NClusters {
				break
			}
			lb := int64(lbs[l])
			if lbs[l] < 0 {
				// int32 overflow means very far cluster; cap to prevent sign-bit confusion.
				lb = math.MaxInt64 >> CidBits
			}
			out[c] = (lb << CidBits) | int64(c)
		}
	}
}

func (ix *IvfIndex) scanClusterGather(bestC int, q *[16]int16, topkK *[5]int64, topkL *[5]uint8, worstKey *int64) {
	start := int(ix.clusterOffsets[bestC])
	end := int(ix.clusterOffsets[bestC+1])
	wk := *worstKey

	for vi := start; vi < end; vi++ {
		base := vi * 16
		d0 := int32(ix.flatVec[base]) - int32(q[0])
		if wk != math.MaxInt64 && int64(d0*d0) >= (wk>>IdxBits) {
			continue
		}
		// AoS: 16 contiguous int16s at base; dims 14,15 are 0 matching q[14,15]=0.
		dist := distL2i16q(ix.flatVec, base, q)
		key := (int64(uint32(dist)) << IdxBits) | int64(vi)
		if key >= wk {
			continue
		}
		wi, mx := 0, topkK[0]
		for t := 1; t < KNeighbors; t++ {
			if topkK[t] > mx {
				mx, wi = topkK[t], t
			}
		}
		topkK[wi] = key
		topkL[wi] = ix.labels[vi]
		wk = topkK[0]
		for t := 1; t < KNeighbors; t++ {
			if topkK[t] > wk {
				wk = topkK[t]
			}
		}
	}
	*worstKey = wk
}
