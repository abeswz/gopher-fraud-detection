package search

const nprobe = 15

type knnEntry struct {
	dist  float32
	label uint8
}

type centEntry struct {
	dist float32
	id   int
}

// KNN finds the k nearest neighbors in the IVF index by searching the nprobe
// nearest clusters, then returns the count of fraud labels among the top-k.
func (idx *IVFIndex) KNN(query [14]float32, k int) int {
	np := nprobe
	if np > idx.C {
		np = idx.C
	}

	topC := make([]centEntry, 0, np)
	maxCD := float32(0)
	maxCP := 0

	for c := 0; c < idx.C; c++ {
		base := c * dims
		var d float32
		for j := 0; j < dims; j++ {
			diff := query[j] - idx.Centroids[base+j]
			d += diff * diff
		}
		if len(topC) < np {
			topC = append(topC, centEntry{d, c})
			if len(topC) == np {
				maxCD, maxCP = centFindMax(topC)
			}
		} else if d < maxCD {
			topC[maxCP] = centEntry{d, c}
			maxCD, maxCP = centFindMax(topC)
		}
	}

	top := make([]knnEntry, 0, k)
	maxDist := float32(0)
	maxPos := 0

	for _, ce := range topC {
		start := int(idx.Starts[ce.id])
		size := int(idx.Sizes[ce.id])
		for i := start; i < start+size; i++ {
			base := i * dims
			var dist float32
			for j := 0; j < dims; j++ {
				ref := float32(idx.Vectors[base+j]) / 10000.0
				diff := query[j] - ref
				dist += diff * diff
			}
			if len(top) < k {
				top = append(top, knnEntry{dist, idx.Labels[i]})
				if len(top) == k {
					maxDist, maxPos = knnFindMax(top)
				}
			} else if dist < maxDist {
				top[maxPos] = knnEntry{dist, idx.Labels[i]}
				maxDist, maxPos = knnFindMax(top)
			}
		}
	}

	fraudCount := 0
	for _, e := range top {
		if e.label == 1 {
			fraudCount++
		}
	}
	return fraudCount
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
