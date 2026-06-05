//go:build !amd64

package search

import "math"

func (ix *IvfIndex) computeClusterPacked(q *[16]int16, out []int64) {
	for c := 0; c < ix.NClusters; c++ {
		base := c * 16
		var lb int64
		for d := 0; d < NDims; d++ {
			qd := int32(q[d])
			lo := int32(ix.bboxMin[base+d])
			hi := int32(ix.bboxMax[base+d])
			var gap int32
			if qd < lo {
				gap = lo - qd
			} else if qd > hi {
				gap = qd - hi
			}
			lb += int64(gap) * int64(gap)
		}
		out[c] = (lb << CidBits) | int64(c)
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
		var dist int32
		for d := 0; d < NDims; d++ {
			dd := int32(ix.flatVec[base+d]) - int32(q[d])
			dist += dd * dd
		}
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
