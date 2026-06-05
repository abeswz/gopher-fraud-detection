package search

import (
	"math"
	"testing"
)

// buildTinyIndex builds a minimal in-memory IvfIndex for testing.
func buildTinyIndex(t *testing.T) *IvfIndex {
	t.Helper()
	const n = 10
	const k = 2

	refs := make([]Ref, n)
	for i := 0; i < 5; i++ {
		refs[i].Label = 0
		for d := 0; d < NDims; d++ {
			refs[i].V[d] = float32(i) * 0.01
		}
	}
	for i := 5; i < n; i++ {
		refs[i].Label = 1
		for d := 0; d < NDims; d++ {
			refs[i].V[d] = 0.9 + float32(i-5)*0.01
		}
	}

	cent := [][16]float32{
		refs[2].V,
		refs[7].V,
	}
	assignments := make([]int32, n)
	for i := 5; i < n; i++ {
		assignments[i] = 1
	}
	offsets, order := CountingSortByCluster(assignments, k)
	SortWithinClusters(refs, cent, assignments, offsets, order)
	bboxMin, bboxMax, pairArr, labels := BBoxPack(refs, order, offsets, k)

	ix := &IvfIndex{
		NClusters:      k,
		NVectors:       n,
		clusterOffsets: offsets,
		bboxMin:        bboxMin,
		bboxMax:        bboxMax,
		labels:         labels,
	}
	for p := 0; p < NPairs; p++ {
		i16 := make([]int16, 2*n+16)
		for j := 0; j < n; j++ {
			packed := pairArr[p][j]
			i16[2*j] = int16(packed & 0xFFFF)
			i16[2*j+1] = int16(uint32(packed) >> 16)
		}
		ix.pairs[p] = i16
	}
	ix.buildBPSOA()
	return ix
}

func TestSearch_AllLegit(t *testing.T) {
	ix := buildTinyIndex(t)
	var q [16]int16
	cnt := ix.Search(&q)
	if cnt != 0 {
		t.Errorf("expected 0 fraud, got %d", cnt)
	}
}

func TestSearch_AllFraud(t *testing.T) {
	ix := buildTinyIndex(t)
	var q [16]int16
	for i := 0; i < NDims; i++ {
		q[i] = int16(math.Round(0.95 * Scale))
	}
	cnt := ix.Search(&q)
	if cnt != 5 {
		t.Errorf("expected 5 fraud, got %d", cnt)
	}
}
