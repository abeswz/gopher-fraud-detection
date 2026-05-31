package search

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// buildVPIndexForTest recursively builds a VPIndex from float32 vectors and labels.
func buildVPIndexForTest(rawVecs [][]float32, labels []uint8, leafSize int) *VPIndex {
	rng := rand.New(rand.NewSource(42))
	n := len(rawVecs)

	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	type nodeEntry struct {
		tau      float32
		childOff uint32
		count    uint16
		vec      [14]int16
	}

	var nodeArr []nodeEntry
	var vecOrder []int

	var buildTree func(idx []int)
	buildTree = func(idx []int) {
		if len(idx) <= leafSize {
			vecStart := len(vecOrder)
			vecOrder = append(vecOrder, idx...)
			nodeArr = append(nodeArr, nodeEntry{
				count:    uint16(len(idx)),
				childOff: uint32(vecStart),
			})
			return
		}

		pivotPos := rng.Intn(len(idx))
		pivotIdx := idx[pivotPos]
		pivotVec := rawVecs[pivotIdx]

		dists := make([]float32, len(idx))
		for i, vi := range idx {
			dists[i] = euclidDistVP(rawVecs[vi], pivotVec)
		}

		sorted := make([]float32, len(dists))
		copy(sorted, dists)
		sort.Slice(sorted, func(a, b int) bool { return sorted[a] < sorted[b] })
		tau := sorted[len(sorted)/2]

		var inner, outer []int
		for i, vi := range idx {
			if dists[i] <= tau {
				inner = append(inner, vi)
			} else {
				outer = append(outer, vi)
			}
		}
		if len(inner) == 0 || len(outer) == 0 {
			mid := len(idx) / 2
			inner, outer = idx[:mid], idx[mid:]
		}

		ni := len(nodeArr)
		nodeArr = append(nodeArr, nodeEntry{})
		buildTree(inner)
		rightNi := len(nodeArr)

		var vec [14]int16
		for j, f := range pivotVec {
			vec[j] = int16(math.Round(float64(f) * 10000))
		}
		nodeArr[ni] = nodeEntry{
			tau:      tau,
			childOff: uint32(rightNi),
			count:    0,
			vec:      vec,
		}
		buildTree(outer)
	}

	buildTree(indices)

	result := &VPIndex{
		Nodes:   make([]VPNode, len(nodeArr)),
		Vectors: make([]int16, n*dims),
		Labels:  make([]uint8, n),
	}
	for i, e := range nodeArr {
		result.Nodes[i] = VPNode{
			Tau:      e.tau,
			ChildOff: e.childOff,
			Count:    e.count,
			Vec:      e.vec,
		}
	}
	for newIdx, origIdx := range vecOrder {
		for j := 0; j < dims; j++ {
			result.Vectors[newIdx*dims+j] = int16(math.Round(float64(rawVecs[origIdx][j]) * 10000))
		}
		result.Labels[newIdx] = labels[origIdx]
	}
	return result
}

func euclidDistVP(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

func bruteForceVPKNN(idx *VPIndex, query [14]float32, k int) int {
	n := len(idx.Labels)
	var topArr [5]knnEntry
	top := topArr[:0]
	maxDist := float32(0)
	maxPos := 0

	for vi := 0; vi < n; vi++ {
		base := vi * dims
		_ = idx.Vectors[base+13]
		d0 := query[0] - float32(idx.Vectors[base])*invScale
		d1 := query[1] - float32(idx.Vectors[base+1])*invScale
		d2 := query[2] - float32(idx.Vectors[base+2])*invScale
		d3 := query[3] - float32(idx.Vectors[base+3])*invScale
		d4 := query[4] - float32(idx.Vectors[base+4])*invScale
		d5 := query[5] - float32(idx.Vectors[base+5])*invScale
		d6 := query[6] - float32(idx.Vectors[base+6])*invScale
		d7 := query[7] - float32(idx.Vectors[base+7])*invScale
		d8 := query[8] - float32(idx.Vectors[base+8])*invScale
		d9 := query[9] - float32(idx.Vectors[base+9])*invScale
		d10 := query[10] - float32(idx.Vectors[base+10])*invScale
		d11 := query[11] - float32(idx.Vectors[base+11])*invScale
		d12 := query[12] - float32(idx.Vectors[base+12])*invScale
		d13 := query[13] - float32(idx.Vectors[base+13])*invScale
		distSq := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

		if len(top) < k {
			top = append(top, knnEntry{distSq, idx.Labels[vi]})
			if len(top) == k {
				maxDist, maxPos = knnFindMax(top)
			}
		} else if distSq < maxDist {
			top[maxPos] = knnEntry{distSq, idx.Labels[vi]}
			maxDist, maxPos = knnFindMax(top)
		}
	}

	count := 0
	for _, e := range top {
		if e.label == 1 {
			count++
		}
	}
	return count
}

func TestVPKNNLeafOnly(t *testing.T) {
	rawVecs := make([][]float32, 10)
	labels := make([]uint8, 10)
	rng := rand.New(rand.NewSource(1))
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = rng.Float32()
		}
		if i < 3 {
			labels[i] = 1
		}
	}

	idx := buildVPIndexForTest(rawVecs, labels, 16)
	if len(idx.Nodes) != 1 || idx.Nodes[0].Count == 0 {
		t.Fatalf("expected single leaf node, got %d nodes, count=%d", len(idx.Nodes), idx.Nodes[0].Count)
	}

	var query [14]float32
	for j := range query {
		query[j] = rng.Float32()
	}

	got := idx.KNN(query, 5)
	want := bruteForceVPKNN(idx, query, 5)
	if got != want {
		t.Errorf("LeafOnly: got %d, want %d", got, want)
	}
}

func TestVPKNNMatchesBruteForce(t *testing.T) {
	const n = 500
	rawVecs := make([][]float32, n)
	labels := make([]uint8, n)
	rng := rand.New(rand.NewSource(99))

	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = rng.Float32()*2 - 1
		}
		if rng.Float32() < 0.3 {
			labels[i] = 1
		}
	}

	idx := buildVPIndexForTest(rawVecs, labels, 16)

	failures := 0
	for q := 0; q < 200; q++ {
		var query [14]float32
		for j := range query {
			query[j] = rng.Float32()*2 - 1
		}
		got := idx.KNN(query, 5)
		want := bruteForceVPKNN(idx, query, 5)
		if got != want {
			t.Errorf("query %d: got fraudCount=%d, want %d", q, got, want)
			failures++
			if failures >= 5 {
				t.Fatal("too many mismatches, stopping")
			}
		}
	}
}

func TestVPKNNExactPivotHit(t *testing.T) {
	rawVecs := make([][]float32, 30)
	labels := make([]uint8, 30)
	rng := rand.New(rand.NewSource(7))

	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = rng.Float32()
		}
	}
	for j := range rawVecs[0] {
		rawVecs[0][j] = 0.5
	}
	labels[0] = 1

	idx := buildVPIndexForTest(rawVecs, labels, 16)

	var query [14]float32
	for vi := 0; vi < len(labels); vi++ {
		base := vi * dims
		match := true
		for j := 0; j < dims; j++ {
			if idx.Vectors[base+j] != 5000 {
				match = false
				break
			}
		}
		if match {
			for j := 0; j < dims; j++ {
				query[j] = float32(idx.Vectors[base+j]) * invScale
			}
			break
		}
	}

	got := idx.KNN(query, 5)
	want := bruteForceVPKNN(idx, query, 5)
	if got != want {
		t.Errorf("ExactPivotHit: got %d, want %d", got, want)
	}
}

func TestVPKNNDegenerateSplit(t *testing.T) {
	const n = 32
	rawVecs := make([][]float32, n)
	labels := make([]uint8, n)
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		rawVecs[i][0] = float32(i) * (1.0 / float32(n))
		for j := 1; j < dims; j++ {
			rawVecs[i][j] = 0.5
		}
	}
	for i := 0; i < 16; i++ {
		labels[i] = 1
	}

	idx := buildVPIndexForTest(rawVecs, labels, 16)

	var query [14]float32
	query[0] = 0.9
	for j := 1; j < dims; j++ {
		query[j] = 0.5
	}

	got := idx.KNN(query, 5)
	want := bruteForceVPKNN(idx, query, 5)
	if got != want {
		t.Errorf("DegenerateSplit: got %d, want %d", got, want)
	}
}

func TestVPKNNAllFraud(t *testing.T) {
	rawVecs := make([][]float32, 10)
	labels := make([]uint8, 10)
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = 1.0
		}
		labels[i] = 1
	}
	idx := buildVPIndexForTest(rawVecs, labels, 16)

	var query [14]float32
	for j := range query {
		query[j] = 0.9
	}
	got := idx.KNN(query, 5)
	if got != 5 {
		t.Errorf("AllFraud: got %d, want 5", got)
	}
}

func TestVPKNNAllLegit(t *testing.T) {
	rawVecs := make([][]float32, 10)
	labels := make([]uint8, 10)
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
	}
	idx := buildVPIndexForTest(rawVecs, labels, 16)

	var query [14]float32
	for j := range query {
		query[j] = 0.1
	}
	got := idx.KNN(query, 5)
	if got != 0 {
		t.Errorf("AllLegit: got %d, want 0", got)
	}
}

func BenchmarkVPKNN(b *testing.B) {
	const n = 3000
	rawVecs := make([][]float32, n)
	labels := make([]uint8, n)
	rng := rand.New(rand.NewSource(42))
	for i := range rawVecs {
		rawVecs[i] = make([]float32, dims)
		for j := range rawVecs[i] {
			rawVecs[i][j] = rng.Float32()
		}
		if rng.Float32() < 0.3 {
			labels[i] = 1
		}
	}
	idx := buildVPIndexForTest(rawVecs, labels, 16)
	var query [14]float32
	for j := range query {
		query[j] = rng.Float32()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.KNN(query, 5)
	}
}
