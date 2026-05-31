package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
	"unsafe"
)

const ivfMagic = "IVF1"
const dims = 14

// IVFIndex stores vectors grouped by cluster for fast approximate KNN.
// Vectors and Labels are zero-copy views into mmap'd file data.
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
	mmap      []byte // retains reference to prevent GC of mmap'd region
}

// Close unmaps the index file. Call at process shutdown.
func (idx *IVFIndex) Close() {
	if idx.mmap != nil {
		_ = syscall.Munmap(idx.mmap)
		idx.mmap = nil
	}
}

func LoadIVFIndex(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(fi.Size())

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	if err := parseIVF(data); err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}

	c := int(binary.LittleEndian.Uint32(data[4:8]))
	n := int(binary.LittleEndian.Uint32(data[8:12]))

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

	// Zero-copy: reinterpret mmap bytes as []int16 and []uint8.
	// Safe on little-endian (x86/x86-64): file is LE, host is LE.
	vecBytes := data[off : off+n*dims*2]
	labelsBytes := data[off+n*dims*2 : off+n*dims*2+n]

	vectors := unsafe.Slice((*int16)(unsafe.Pointer(&vecBytes[0])), n*dims)
	labels := labelsBytes

	return &IVFIndex{
		C: c, N: n,
		Centroids: centroids,
		Starts:    starts,
		Sizes:     sizes,
		Vectors:   vectors,
		Labels:    labels,
		mmap:      data,
	}, nil
}

func parseIVF(data []byte) error {
	if len(data) < 12 {
		return fmt.Errorf("index too small: %d bytes", len(data))
	}
	if string(data[0:4]) != ivfMagic {
		return fmt.Errorf("bad magic: %q (want %q)", data[0:4], ivfMagic)
	}
	c := int(binary.LittleEndian.Uint32(data[4:8]))
	n := int(binary.LittleEndian.Uint32(data[8:12]))
	centSize := c * dims * 4
	startsSize := c * 4
	sizesSize := c * 4
	vecsSize := n * dims * 2
	expected := 12 + centSize + startsSize + sizesSize + vecsSize + n
	if len(data) != expected {
		return fmt.Errorf("size mismatch: got %d, want %d", len(data), expected)
	}
	return nil
}

// Index is the common interface for both IVFIndex and VPIndex.
type Index interface {
	KNN(query [14]float32, k int) int
}
