package search

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// Ref is one corpus vector. V[14], V[15] = 0 (padding for 16-lane SIMD).
type Ref struct {
	V     [16]float32
	Label uint8
}

// TagFromFloat computes the 4-bit partition tag from float32 vector values.
//
//	bit0 = has_last_tx      (V[5] >= 0; sentinel is -1.0)
//	bit1 = unknown_merchant (V[11] > 0.5)
//	bit2 = is_online        (V[9]  > 0.5)
//	bit3 = card_present     (V[10] > 0.5)
func TagFromFloat(v *[16]float32) int {
	tag := 0
	if v[5] >= 0 {
		tag |= 1
	}
	if v[11] > 0.5 {
		tag |= 2
	}
	if v[9] > 0.5 {
		tag |= 4
	}
	if v[10] > 0.5 {
		tag |= 8
	}
	return tag
}

// FilterByTag compacts refs in-place keeping only those matching tag.
func FilterByTag(refs []Ref, tag int) []Ref {
	dst := 0
	for src := range refs {
		if TagFromFloat(&refs[src].V) == tag {
			if src != dst {
				refs[dst] = refs[src]
			}
			dst++
		}
	}
	return refs[:dst]
}

func quantizeRef(v *[16]float32) [16]int16 {
	var q [16]int16
	for d := 0; d < NDims; d++ {
		x := math.RoundToEven(float64(v[d]) * Scale)
		if x > math.MaxInt16 {
			x = math.MaxInt16
		} else if x < math.MinInt16 {
			x = math.MinInt16
		}
		q[d] = int16(x)
	}
	return q
}

func l2sqF32(a, b *[16]float32) float32 {
	var s float32
	for i := 0; i < NDims; i++ {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// KMeans runs Lloyd's algorithm with evenly-spaced seed init.
// Parallel assign phase; serial update phase.
func KMeans(refs []Ref, k, iters int) (cent [][16]float32, assignments []int32) {
	n := len(refs)
	if k > n {
		k = n
	}
	if k < 1 {
		k = 1
	}
	cent = make([][16]float32, k)
	assignments = make([]int32, n)

	step := n / k
	if step < 1 {
		step = 1
	}
	for c := 0; c < k; c++ {
		src := c * step
		if src >= n {
			src = n - 1
		}
		copy(cent[c][:NDims], refs[src].V[:NDims])
	}

	workers := runtime.NumCPU()
	sums := make([][NDims]float64, k)
	counts := make([]uint64, k)

	for it := 0; it < iters; it++ {
		var wg sync.WaitGroup
		chunk := (n + workers - 1) / workers
		for w := 0; w < workers; w++ {
			lo := w * chunk
			hi := lo + chunk
			if hi > n {
				hi = n
			}
			if lo >= hi {
				break
			}
			wg.Add(1)
			go func(lo, hi int) {
				defer wg.Done()
				for i := lo; i < hi; i++ {
					best := l2sqF32(&refs[i].V, &cent[0])
					bestC := int32(0)
					for c := 1; c < k; c++ {
						if d := l2sqF32(&refs[i].V, &cent[c]); d < best {
							best = d
							bestC = int32(c)
						}
					}
					assignments[i] = bestC
				}
			}(lo, hi)
		}
		wg.Wait()

		for c := range sums {
			sums[c] = [NDims]float64{}
			counts[c] = 0
		}
		for i := 0; i < n; i++ {
			c := assignments[i]
			counts[c]++
			for d := 0; d < NDims; d++ {
				sums[c][d] += float64(refs[i].V[d])
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				continue
			}
			inv := 1.0 / float64(counts[c])
			for d := 0; d < NDims; d++ {
				cent[c][d] = float32(sums[c][d] * inv)
			}
		}
	}
	return cent, assignments
}

// CountingSortByCluster returns offsets (len k+1) and order (len n) for
// stable cluster-grouped ordering of refs.
func CountingSortByCluster(assignments []int32, k int) (offsets []uint32, order []uint32) {
	n := len(assignments)
	offsets = make([]uint32, k+1)
	order = make([]uint32, n)
	for _, c := range assignments {
		offsets[c+1]++
	}
	for c := 0; c < k; c++ {
		offsets[c+1] += offsets[c]
	}
	cursor := make([]uint32, k)
	copy(cursor, offsets[:k])
	for i, c := range assignments {
		order[cursor[c]] = uint32(i)
		cursor[c]++
	}
	return offsets, order
}

// SortWithinClusters reorders each cluster's slice of order so vectors
// nearest their centroid come first.
func SortWithinClusters(refs []Ref, cent [][16]float32, assignments []int32, offsets, order []uint32) {
	k := len(cent)
	for c := 0; c < k; c++ {
		lo, hi := offsets[c], offsets[c+1]
		if hi-lo < 2 {
			continue
		}
		seg := order[lo:hi]
		ctr := &cent[c]
		sort.Slice(seg, func(i, j int) bool {
			return l2sqF32(&refs[seg[i]].V, ctr) < l2sqF32(&refs[seg[j]].V, ctr)
		})
	}
}

// BBoxPack computes per-cluster int16 bounding boxes and packs quantized
// dims into 7 SoA pair arrays. Each pair_arr[p][pos] = lo|hi<<16 where
// lo=uint16(dim2p), hi=uint16(dim2p+1) — two int16s per int32.
func BBoxPack(refs []Ref, order, offsets []uint32, k int) (bboxMin, bboxMax []int16, pairArr [NPairs][]int32, labels []uint8) {
	n := len(order)
	bboxMin = make([]int16, k*16)
	bboxMax = make([]int16, k*16)
	for c := 0; c < k; c++ {
		for lane := 0; lane < NDims; lane++ {
			bboxMin[c*16+lane] = math.MaxInt16
			bboxMax[c*16+lane] = math.MinInt16
		}
	}
	for p := 0; p < NPairs; p++ {
		pairArr[p] = make([]int32, n)
	}
	labels = make([]uint8, n)

	cid := 0
	for pos := 0; pos < n; pos++ {
		for offsets[cid+1] <= uint32(pos) {
			cid++
		}
		ref := &refs[order[pos]]
		labels[pos] = ref.Label
		qv := quantizeRef(&ref.V)

		base := cid * 16
		for lane := 0; lane < NDims; lane++ {
			if qv[lane] < bboxMin[base+lane] {
				bboxMin[base+lane] = qv[lane]
			}
			if qv[lane] > bboxMax[base+lane] {
				bboxMax[base+lane] = qv[lane]
			}
		}
		for p := 0; p < NPairs; p++ {
			lo := uint32(uint16(qv[2*p]))
			hi := uint32(uint16(qv[2*p+1]))
			pairArr[p][pos] = int32(lo | hi<<16)
		}
	}
	return bboxMin, bboxMax, pairArr, labels
}
