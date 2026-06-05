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

	var vq [16]int16
	for vi := start; vi < end; vi++ {
		for p := 0; p < NPairs; p++ {
			vq[2*p] = ix.pairs[p][2*vi]
			vq[2*p+1] = ix.pairs[p][2*vi+1]
		}
		d0 := int32(vq[0]) - int32(q[0])
		if wk != math.MaxInt64 && int64(d0*d0) >= (wk>>IdxBits) {
			continue
		}
		var dist int32
		for d := 0; d < NDims; d++ {
			dd := int32(vq[d]) - int32(q[d])
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

func (ix *IvfIndex) searchCore(q *[16]int16, maxProbes int, topkK *[5]int64, topkL *[5]uint8) {
	const maxI64 = int64(math.MaxInt64)

	var packed [MaxK]int64
	ix.computeClusterPacked(q, packed[:ix.NClusters])

	for i := 0; i < KNeighbors; i++ {
		topkK[i] = maxI64
	}
	worstKey := maxI64

	probe := 0
	repairDone := false
	for {
	probeLoop:
		for probe < maxProbes {
			best := maxI64
			for c := 0; c < ix.NClusters; c++ {
				if packed[c] < best {
					best = packed[c]
				}
			}
			if best == maxI64 {
				break probeLoop
			}
			bestLb := best >> CidBits
			if (bestLb << IdxBits) >= worstKey {
				break probeLoop
			}
			bestC := int(best & CidMask)
			packed[bestC] = maxI64
			ix.scanClusterGather(bestC, q, topkK, topkL, &worstKey)
			probe++
		}

		if repairDone {
			return
		}
		if worstKey != maxI64 {
			cnt := 0
			for _, l := range topkL {
				cnt += int(l)
			}
			if cnt < NProbeRepairMin || cnt > NProbeRepairMax {
				return
			}
		}
		repairDone = true
		maxProbes = ix.NClusters
	}
}

func (ix *IvfIndex) Search(q *[16]int16) uint8 {
	var topkK [5]int64
	var topkL [5]uint8
	ix.searchCore(q, NProbeInitial, &topkK, &topkL)
	var cnt uint8
	for _, l := range topkL {
		cnt += l
	}
	return cnt
}
