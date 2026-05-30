package search

type knnEntry struct {
	dist  float32
	label uint8
}

func (idx *Index) KNN(query [14]float32, k int) int {
	top := make([]knnEntry, 0, k)
	maxDist := float32(0)
	maxPos := 0

	for i := 0; i < idx.N; i++ {
		base := i * 14
		var dist float32
		for j := 0; j < 14; j++ {
			ref := float32(idx.Vectors[base+j]) / 10000.0
			d := query[j] - ref
			dist += d * d
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
