package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
	"unsafe"
)

const dims = 16

type Index interface {
	KNN(query [16]float32, k int) int
}

const ivfhMagic = "IVFH"

type IVFHIndex struct {
	K1, K2, N    int
	DSafe        float32 // stored as L2, not L2²
	DSafeSq      float32
	NCoarseProbe int
	MacroCentroids []float32
	MicroCentroids []float32
	Starts         []uint32
	Sizes          []uint32
	Radii          []float32
	Vectors        []int16 // zero-copy mmap view
	Labels         []uint8 // zero-copy mmap view
	mmap           []byte
}

func (idx *IVFHIndex) Close() {
	if idx.mmap != nil {
		_ = syscall.Munmap(idx.mmap)
		idx.mmap = nil
	}
}

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
