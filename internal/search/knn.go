package search

import "math"

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
