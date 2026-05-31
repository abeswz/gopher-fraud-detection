package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const ivfMagic = "IVF1"
const dims = 14

// IVFIndex stores vectors grouped by cluster for fast approximate KNN.
// Binary format (little-endian):
//
//	[4]       "IVF1" magic
//	[4]       uint32 C (number of clusters)
//	[4]       uint32 N (total vectors)
//	[C×14×4]  float32 centroids
//	[C×4]     uint32 cluster starts (index into Vectors/Labels)
//	[C×4]     uint32 cluster sizes
//	[N×14×2]  int16 vectors (all vectors contiguous, not interleaved with labels)
//	[N×1]     uint8 labels
type IVFIndex struct {
	C         int
	N         int
	Centroids []float32
	Starts    []uint32
	Sizes     []uint32
	Vectors   []int16
	Labels    []uint8
}

func LoadIVFIndex(path string) (*IVFIndex, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 12 {
		return nil, fmt.Errorf("index too small: %d bytes", len(data))
	}

	if string(data[0:4]) != ivfMagic {
		return nil, fmt.Errorf("bad magic: %q (want %q)", data[0:4], ivfMagic)
	}

	c := int(binary.LittleEndian.Uint32(data[4:8]))
	n := int(binary.LittleEndian.Uint32(data[8:12]))

	centSize := c * dims * 4
	startsSize := c * 4
	sizesSize := c * 4
	vecsSize := n * dims * 2
	labelsSize := n
	expected := 12 + centSize + startsSize + sizesSize + vecsSize + labelsSize
	if len(data) != expected {
		return nil, fmt.Errorf("size mismatch: got %d, want %d", len(data), expected)
	}

	off := 12

	centroids := make([]float32, c*dims)
	for i := range centroids {
		centroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	starts := make([]uint32, c)
	for i := range starts {
		starts[i] = binary.LittleEndian.Uint32(data[off:])
		off += 4
	}

	sizes := make([]uint32, c)
	for i := range sizes {
		sizes[i] = binary.LittleEndian.Uint32(data[off:])
		off += 4
	}

	vectors := make([]int16, n*dims)
	for i := range vectors {
		vectors[i] = int16(binary.LittleEndian.Uint16(data[off:]))
		off += 2
	}

	labels := make([]uint8, n)
	copy(labels, data[off:off+n])

	return &IVFIndex{
		C: c, N: n,
		Centroids: centroids,
		Starts:    starts,
		Sizes:     sizes,
		Vectors:   vectors,
		Labels:    labels,
	}, nil
}

// Index is the common interface for both IVFIndex and VPIndex.
type Index interface {
	KNN(query [14]float32, k int) int
}
