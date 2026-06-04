package search

import "math"

const invScale = float32(1.0 / 10000.0) // multiply is cheaper than divide

type knnEntry struct {
	dist  float32
	label uint8
}

type centEntry struct {
	dist float32
	id   int
}

func knnFindMax(entries []knnEntry) (maxDist float32, maxPos int) {
	maxDist = entries[0].dist
	maxPos = 0
	for i := 1; i < len(entries); i++ {
		if entries[i].dist > maxDist {
			maxDist = entries[i].dist
			maxPos = i
		}
	}
	return
}

func centFindMax(entries []centEntry) (maxDist float32, maxPos int) {
	maxDist = entries[0].dist
	maxPos = 0
	for i := 1; i < len(entries); i++ {
		if entries[i].dist > maxDist {
			maxDist = entries[i].dist
			maxPos = i
		}
	}
	return
}

const (
	NCoarseProbeSubseq = 4
	NCoarseProbeFirst  = 3
	nprobeInit         = 8
	nprobeMax          = 20
)

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func (idx *IVFHIndex) KNN(query [16]float32, k int) int {
	q0 := query[0]
	q1 := query[1]
	q2 := query[2]
	q3 := query[3]
	q4 := query[4]
	q5 := query[5]
	q6 := query[6]
	q7 := query[7]
	q8 := query[8]
	q9 := query[9]
	q10 := query[10]
	q11 := query[11]
	q12 := query[12]
	q13 := query[13]
	q14 := query[14]
	q15 := query[15]

	nCoarse := min(idx.NCoarseProbe, idx.K1)
	nMicro := min(nprobeMax, idx.K1*idx.K2)

	var topMacroArr [NCoarseProbeSubseq]centEntry
	topMacro := topMacroArr[:0]
	maxMacroD := float32(0)
	maxMacroP := 0

	cents := idx.MacroCentroids
	for c, base := 0, 0; c < idx.K1; c, base = c+1, base+dims {
		_ = cents[base+15]
		d0 := q0 - cents[base]
		d1 := q1 - cents[base+1]
		d2 := q2 - cents[base+2]
		d3 := q3 - cents[base+3]
		d4 := q4 - cents[base+4]
		d5 := q5 - cents[base+5]
		d6 := q6 - cents[base+6]
		d7 := q7 - cents[base+7]
		d8 := q8 - cents[base+8]
		d9 := q9 - cents[base+9]
		d10 := q10 - cents[base+10]
		d11 := q11 - cents[base+11]
		d12 := q12 - cents[base+12]
		d13 := q13 - cents[base+13]
		d14 := q14 - cents[base+14]
		d15 := q15 - cents[base+15]
		d := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13 +
			d14*d14 + d15*d15

		if len(topMacro) < nCoarse {
			topMacro = append(topMacro, centEntry{d, c})
			if len(topMacro) == nCoarse {
				maxMacroD, maxMacroP = centFindMax(topMacro)
			}
		} else if d < maxMacroD {
			topMacro[maxMacroP] = centEntry{d, c}
			maxMacroD, maxMacroP = centFindMax(topMacro)
		}
	}

	var topMicroArr [nprobeMax]centEntry
	topMicro := topMicroArr[:0]
	maxMicroD := float32(0)
	maxMicroP := 0

	mcents := idx.MicroCentroids
	for _, me := range topMacro {
		macroBase := me.id * idx.K2
		for j := 0; j < idx.K2; j++ {
			base := (macroBase + j) * dims
			_ = mcents[base+15]
			d0 := q0 - mcents[base]
			d1 := q1 - mcents[base+1]
			d2 := q2 - mcents[base+2]
			d3 := q3 - mcents[base+3]
			d4 := q4 - mcents[base+4]
			d5 := q5 - mcents[base+5]
			d6 := q6 - mcents[base+6]
			d7 := q7 - mcents[base+7]
			d8 := q8 - mcents[base+8]
			d9 := q9 - mcents[base+9]
			d10 := q10 - mcents[base+10]
			d11 := q11 - mcents[base+11]
			d12 := q12 - mcents[base+12]
			d13 := q13 - mcents[base+13]
			d14 := q14 - mcents[base+14]
			d15 := q15 - mcents[base+15]
			d := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
				d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13 +
				d14*d14 + d15*d15

			leafID := macroBase + j
			if len(topMicro) < nMicro {
				topMicro = append(topMicro, centEntry{d, leafID})
				if len(topMicro) == nMicro {
					maxMicroD, maxMicroP = centFindMax(topMicro)
				}
			} else if d < maxMicroD {
				topMicro[maxMicroP] = centEntry{d, leafID}
				maxMicroD, maxMicroP = centFindMax(topMicro)
			}
		}
	}

	// sort ascending: required for centroid pruning in repair path
	n := len(topMicro)
	for i := 1; i < n; i++ {
		key := topMicro[i]
		j := i - 1
		for j >= 0 && topMicro[j].dist > key.dist {
			topMicro[j+1] = topMicro[j]
			j--
		}
		topMicro[j+1] = key
	}

	return ivfhScanVectors(idx, topMicro, &query, k, q0)
}

func ivfhScanVectors(idx *IVFHIndex, topMicro []centEntry, query *[16]float32, k int, q0 float32) int {
	var topArr [5]knnEntry
	top := topArr[:0]
	maxDist := float32(0)
	maxPos := 0

	vecs := idx.Vectors
	labs := idx.Labels

	fastEnd := min(nprobeInit, len(topMicro))

	for _, ce := range topMicro[:fastEnd] {
		start := int(idx.Starts[ce.id])
		size := int(idx.Sizes[ce.id])
		base := start * dims

		for vi := start; vi < start+size; vi, base = vi+1, base+dims {
			_ = vecs[base+15]
			d0 := q0 - float32(vecs[base])*invScale
			if len(top) == k && d0*d0 >= maxDist {
				continue
			}
			dist := distL2i16_16(vecs, base, query)
			if len(top) < k {
				top = append(top, knnEntry{dist, labs[vi]})
				if len(top) == k {
					maxDist, maxPos = knnFindMax(top)
				}
			} else if dist < maxDist {
				top[maxPos] = knnEntry{dist, labs[vi]}
				maxDist, maxPos = knnFindMax(top)
			}
		}
	}

	if len(top) == k {
		fraudCount := countFraudH(top)
		if fraudCount == 5 {
			return 5
		}
		if fraudCount == 0 && maxDist < idx.DSafeSq {
			return 0
		}
	}

	// repair path: probe remaining clusters with centroid-distance pruning
	for _, ce := range topMicro[fastEnd:] {
		dCentroid := sqrt32(ce.dist)
		radius := idx.Radii[ce.id]
		lowerBound := dCentroid - radius
		if lowerBound > 0 && lowerBound*lowerBound > maxDist {
			break
		}

		start := int(idx.Starts[ce.id])
		size := int(idx.Sizes[ce.id])
		base := start * dims

		for vi := start; vi < start+size; vi, base = vi+1, base+dims {
			_ = vecs[base+15]
			d0 := q0 - float32(vecs[base])*invScale
			if len(top) == k && d0*d0 >= maxDist {
				continue
			}
			dist := distL2i16_16(vecs, base, query)
			if len(top) < k {
				top = append(top, knnEntry{dist, labs[vi]})
				if len(top) == k {
					maxDist, maxPos = knnFindMax(top)
				}
			} else if dist < maxDist {
				top[maxPos] = knnEntry{dist, labs[vi]}
				maxDist, maxPos = knnFindMax(top)
			}
		}
	}

	return countFraudH(top)
}

func countFraudH(entries []knnEntry) int {
	n := 0
	for _, e := range entries {
		if e.label == 1 {
			n++
		}
	}
	return n
}
