package search

import "math"

const invScale = float32(1.0 / 10000.0) // multiply is cheaper than divide
const invScaleSq = invScale * invScale   // for centroid-pruning in int32 path

type knnEntry struct {
	dist  int32
	label uint8
}

type centEntry struct {
	dist float32
	id   int
}

func knnFindMax(entries []knnEntry) (maxDist int32, maxPos int) {
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
	NCoarseProbeFirstKnown    = 12
	NCoarseProbeFirstUnknown  = 12
	NCoarseProbeSubseqKnown   = 16
	NCoarseProbeSubseqUnknown = 16
	nprobeInit                = 8
	nprobeMax                 = 20
	nprobeRepair       = 128
	nprobeThreshRepair = 128
)

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// boxMinDistSqI computes the squared minimum distance from qi16 to the axis-aligned
// bounding box [boxMin[base..base+dims), boxMax[base..base+dims)) in int16 space.
// Returns 0 when the query falls inside the box on all dimensions.
func boxMinDistSqI(qi16 *[dims]int16, boxMin, boxMax []int16, base int) int32 {
	_ = boxMin[base+15]
	_ = boxMax[base+15]
	var d int32
	for i := 0; i < dims; i++ {
		q := int32(qi16[i])
		lo := int32(boxMin[base+i])
		hi := int32(boxMax[base+i])
		var gap int32
		if q < lo {
			gap = lo - q
		} else if q > hi {
			gap = q - hi
		}
		d += gap * gap
	}
	return d
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
	nMicro := min(nprobeRepair, idx.K1*idx.K2)

	var topMacroArr [NCoarseProbeSubseqKnown]centEntry
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

	var topMicroArr [nprobeRepair]centEntry
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

	return ivfhScanVectors(idx, topMicro, &query, k)
}

func ivfhScanVectors(idx *IVFHIndex, topMicro []centEntry, query *[16]float32, k int) int {
	// Convert query float32→int16 once; all vector distances computed in int32.
	var qi16 [16]int16
	for i, v := range query {
		qi16[i] = int16(math.Round(float64(v) * 10000))
	}
	q0i16 := int32(qi16[0])

	var topArr [5]knnEntry
	top := topArr[:0]
	maxDist := int32(0)
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
			d0 := q0i16 - int32(vecs[base])
			if len(top) == k && d0*d0 >= maxDist {
				continue
			}
			dist := distL2i16q(vecs, base, &qi16)
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
			return idx.knnFullMacroRepair(query, k)
		}
		if fraudCount == 0 && maxDist < idx.DSafeSqI32 {
			return 0
		}
	}

	// phase 2: standard repair — probe next clusters up to nprobeMax with centroid pruning
	standardEnd := min(nprobeMax, len(topMicro))
	for _, ce := range topMicro[fastEnd:standardEnd] {
		dCentroid := sqrt32(ce.dist)
		radius := idx.Radii[ce.id]
		lowerBound := dCentroid - radius
		if lowerBound > 0 && lowerBound*lowerBound > float32(maxDist)*invScaleSq {
			break
		}
		if len(top) == k && idx.BoxMin != nil {
			if boxMinDistSqI(&qi16, idx.BoxMin, idx.BoxMax, ce.id*dims) >= maxDist {
				continue
			}
		}

		start := int(idx.Starts[ce.id])
		size := int(idx.Sizes[ce.id])
		base := start * dims

		for vi := start; vi < start+size; vi, base = vi+1, base+dims {
			_ = vecs[base+15]
			d0 := q0i16 - int32(vecs[base])
			if len(top) == k && d0*d0 >= maxDist {
				continue
			}
			dist := distL2i16q(vecs, base, &qi16)
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

	// mid-check: clear consensus after standard repair → skip deep repair
	if len(top) == k {
		fraudCount := countFraudH(top)
		if fraudCount == 0 {
			return 0
		}
		// ambiguous or fraud-side (1–5): deep repair — probe remaining candidates
		for _, ce := range topMicro[standardEnd:] {
			dCentroid := sqrt32(ce.dist)
			radius := idx.Radii[ce.id]
			lowerBound := dCentroid - radius
			if lowerBound > 0 && lowerBound*lowerBound > float32(maxDist)*invScaleSq {
				break
			}
			if len(top) == k && idx.BoxMin != nil {
				if boxMinDistSqI(&qi16, idx.BoxMin, idx.BoxMax, ce.id*dims) >= maxDist {
					continue
				}
			}

			start := int(idx.Starts[ce.id])
			size := int(idx.Sizes[ce.id])
			base := start * dims

			for vi := start; vi < start+size; vi, base = vi+1, base+dims {
				_ = vecs[base+15]
				d0 := q0i16 - int32(vecs[base])
				if len(top) == k && d0*d0 >= maxDist {
					continue
				}
				dist := distL2i16q(vecs, base, &qi16)
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
	}

	finalCount := countFraudH(top)
	if len(top) == k && finalCount >= 3 {
		return idx.knnFullMacroRepair(query, k)
	}
	return finalCount
}

// knnFullMacroRepair scans the top-nprobeThreshRepair micro clusters to find true top-k neighbors.
// Called when IVF returns an ambiguous or high-fraud count.
func (idx *IVFHIndex) knnFullMacroRepair(query *[16]float32, k int) int {
	q0 := (*query)[0]
	q1 := (*query)[1]
	q2 := (*query)[2]
	q3 := (*query)[3]
	q4 := (*query)[4]
	q5 := (*query)[5]
	q6 := (*query)[6]
	q7 := (*query)[7]
	q8 := (*query)[8]
	q9 := (*query)[9]
	q10 := (*query)[10]
	q11 := (*query)[11]
	q12 := (*query)[12]
	q13 := (*query)[13]
	q14 := (*query)[14]
	q15 := (*query)[15]

	// Convert query for int16-space vector scan.
	var qi16 [16]int16
	for i, v := range query {
		qi16[i] = int16(math.Round(float64(v) * 10000))
	}
	q0i16 := int32(qi16[0])

	nMicro := min(nprobeThreshRepair, idx.K1*idx.K2)
	var allMicro [nprobeThreshRepair]centEntry
	topMicro := allMicro[:0]
	maxMicroD := float32(0)
	maxMicroP := 0

	mcents := idx.MicroCentroids
	total := idx.K1 * idx.K2
	for leafID, base := 0, 0; leafID < total; leafID, base = leafID+1, base+dims {
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

	var topArr [5]knnEntry
	top := topArr[:0]
	maxDist := int32(0)
	maxPos := 0

	vecs := idx.Vectors
	labs := idx.Labels

	for _, ce := range allMicro {
		dCentroid := sqrt32(ce.dist)
		radius := idx.Radii[ce.id]
		lowerBound := dCentroid - radius
		if len(top) == k && lowerBound > 0 && lowerBound*lowerBound > float32(maxDist)*invScaleSq {
			break
		}
		if len(top) == k && idx.BoxMin != nil {
			if boxMinDistSqI(&qi16, idx.BoxMin, idx.BoxMax, ce.id*dims) >= maxDist {
				continue
			}
		}

		start := int(idx.Starts[ce.id])
		size := int(idx.Sizes[ce.id])
		base := start * dims

		for vi := start; vi < start+size; vi, base = vi+1, base+dims {
			_ = vecs[base+15]
			d0 := q0i16 - int32(vecs[base])
			if len(top) == k && d0*d0 >= maxDist {
				continue
			}
			dist := distL2i16q(vecs, base, &qi16)
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
