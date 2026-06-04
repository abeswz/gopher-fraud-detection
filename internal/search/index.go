package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
	"unsafe"
)

const ivfMagic = "IVF2"
const dims = 16

// IVFIndex stores vectors grouped by cluster for fast approximate KNN.
// Vectors and Labels are zero-copy views into mmap'd file data.
// Binary format (little-endian):
//
//	[4]       "IVF2" magic
//	[4]       uint32 C (number of clusters)
//	[4]       uint32 N (total vectors)
//	[C×16×4]  float32 centroids
//	[C×4]     uint32 cluster starts (index into Vectors/Labels)
//	[C×4]     uint32 cluster sizes
//	[N×16×2]  int16 vectors (all vectors contiguous, not interleaved with labels)
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
	if n == 0 {
		return fmt.Errorf("index has zero vectors")
	}
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
	KNN(query [16]float32, k int) int
}

const ivfhMagic = "IVFH"

// IVFHIndex is a 2-level hierarchical IVF index (IVF_H2 format).
// MacroCentroids, MicroCentroids, Radii are parsed into Go slices at load time.
// Vectors and Labels are zero-copy views into the mmap'd file.
type IVFHIndex struct {
	K1, K2, N    int
	DSafe        float32 // L2 dist threshold (stored as L2, not L2²)
	DSafeSq      float32 // DSafe*DSafe — pre-computed at load time
	NCoarseProbe int     // top macro clusters to probe
	MacroCentroids []float32 // K1×16
	MicroCentroids []float32 // K1×K2×16, indexed by [macro*K2+micro]*16
	Starts         []uint32  // K1×K2
	Sizes          []uint32  // K1×K2
	Radii          []float32 // K1×K2, max L2 dist from centroid to any vector
	Vectors        []int16   // N×16 (zero-copy mmap view)
	Labels         []uint8   // N    (zero-copy mmap view)
	mmap           []byte
}

// Close unmaps the index file. Call at process shutdown.
func (idx *IVFHIndex) Close() {
	if idx.mmap != nil {
		_ = syscall.Munmap(idx.mmap)
		idx.mmap = nil
	}
}

// LoadIVFHIndex mmaps path and parses the IVF_H2 binary format.
// nCoarseProbe is the number of macro clusters to probe per query.
func LoadIVFHIndex(path string, nCoarseProbe int) (*IVFHIndex, error) {
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

	if err := parseIVFH(data); err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}

	dSafe := math.Float32frombits(binary.LittleEndian.Uint32(data[4:8]))
	k1 := int(binary.LittleEndian.Uint32(data[8:12]))
	k2 := int(binary.LittleEndian.Uint32(data[12:16]))
	n := int(binary.LittleEndian.Uint32(data[16:20]))
	nLeaf := k1 * k2

	off := 20

	macroCentroids := make([]float32, k1*dims)
	for i := range macroCentroids {
		macroCentroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	microCentroids := make([]float32, nLeaf*dims)
	for i := range microCentroids {
		microCentroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	starts := make([]uint32, nLeaf)
	for i := range starts {
		starts[i] = binary.LittleEndian.Uint32(data[off:])
		off += 4
	}

	sizes := make([]uint32, nLeaf)
	for i := range sizes {
		sizes[i] = binary.LittleEndian.Uint32(data[off:])
		off += 4
	}

	radii := make([]float32, nLeaf)
	for i := range radii {
		radii[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	vecBytes := data[off : off+n*dims*2]
	labelsBytes := data[off+n*dims*2 : off+n*dims*2+n]

	vectors := unsafe.Slice((*int16)(unsafe.Pointer(&vecBytes[0])), n*dims)
	labels := labelsBytes

	return &IVFHIndex{
		K1: k1, K2: k2, N: n,
		DSafe:          dSafe,
		DSafeSq:        dSafe * dSafe,
		NCoarseProbe:   nCoarseProbe,
		MacroCentroids: macroCentroids,
		MicroCentroids: microCentroids,
		Starts:         starts,
		Sizes:          sizes,
		Radii:          radii,
		Vectors:        vectors,
		Labels:         labels,
		mmap:           data,
	}, nil
}

func parseIVFH(data []byte) error {
	if len(data) < 20 {
		return fmt.Errorf("ivfh: file too small: %d bytes", len(data))
	}
	if string(data[0:4]) != ivfhMagic {
		return fmt.Errorf("ivfh: bad magic: %q (want %q)", data[0:4], ivfhMagic)
	}
	k1 := int(binary.LittleEndian.Uint32(data[8:12]))
	k2 := int(binary.LittleEndian.Uint32(data[12:16]))
	n := int(binary.LittleEndian.Uint32(data[16:20]))
	if n == 0 {
		return fmt.Errorf("ivfh: index has zero vectors")
	}
	nLeaf := k1 * k2
	expected := 20 +
		k1*dims*4 +    // macro_centroids
		nLeaf*dims*4 + // micro_centroids
		nLeaf*4 +      // starts
		nLeaf*4 +      // sizes
		nLeaf*4 +      // radii
		n*dims*2 +     // vectors (int16)
		n              // labels (uint8)
	if len(data) != expected {
		return fmt.Errorf("ivfh: size mismatch: got %d, want %d", len(data), expected)
	}
	return nil
}
