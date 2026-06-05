package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// IvfIndex is one partition's index memory-mapped into the process.
type IvfIndex struct {
	data      []byte
	NClusters int
	NVectors  int

	clusterOffsets []uint32
	bboxMin        []int16 // K*16
	bboxMax        []int16 // K*16
	pairs          [NPairs][]int16
	labels         []uint8

	// bpsoaMin/bpsoaMax: K rounded up to multiples of 8, pair SoA for SIMD
	bpsoaMin []int16
	bpsoaMax []int16
}

func align64(x int) int { return (x + 63) &^ 63 }

func viewAt[T any](data []byte, off, n int) []T {
	return unsafe.Slice((*T)(unsafe.Pointer(&data[off])), n)
}

// Open mmaps the partition index file, validates the header, and builds bpsoaMin/bpsoaMax.
func Open(path string) (*IvfIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(st.Size())
	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE|unix.MAP_POPULATE)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	unix.Mlock(data)
	unix.Madvise(data, unix.MADV_HUGEPAGE)
	unix.Madvise(data, unix.MADV_WILLNEED)

	if size < 64 || string(data[0:8]) != magic {
		unix.Munmap(data)
		return nil, fmt.Errorf("index: bad magic in %s", path)
	}
	nc := int(binary.LittleEndian.Uint32(data[8:12]))
	if nc < 1 || nc > MaxK {
		unix.Munmap(data)
		return nil, fmt.Errorf("index: n_clusters=%d out of range in %s", nc, path)
	}
	nv := int(binary.LittleEndian.Uint32(data[12:16]))

	ix := &IvfIndex{data: data, NClusters: nc, NVectors: nv}

	off := 64
	ix.clusterOffsets = viewAt[uint32](data, off, nc+1)
	off = align64(off + (nc+1)*4)
	ix.bboxMin = viewAt[int16](data, off, nc*16)
	off = align64(off + nc*32)
	ix.bboxMax = viewAt[int16](data, off, nc*16)
	off = align64(off + nc*32)
	for p := 0; p < NPairs; p++ {
		ix.pairs[p] = viewAt[int16](data, off, 2*nv+16) // +16 for SIMD over-read safety
		off = align64(off + nv*4)
	}
	ix.labels = viewAt[uint8](data, off, nv)

	ix.buildBPSOA()
	return ix, nil
}

// Close unmaps the file.
func (ix *IvfIndex) Close() error {
	if ix.data != nil {
		err := unix.Munmap(ix.data)
		ix.data = nil
		return err
	}
	return nil
}

// buildBPSOA reshapes flat per-cluster bbox arrays into 8-cluster pair groups
// for SIMD processing.
func (ix *IvfIndex) buildBPSOA() {
	K := ix.NClusters
	nGroups := (K + 7) / 8
	ix.bpsoaMin = make([]int16, nGroups*NPairs*16)
	ix.bpsoaMax = make([]int16, nGroups*NPairs*16)
	for g := 0; g < nGroups; g++ {
		for p := 0; p < NPairs; p++ {
			dst := (g*NPairs + p) * 16
			for l := 0; l < 8; l++ {
				c := g*8 + l
				di := dst + l*2
				if c < K {
					ix.bpsoaMin[di] = ix.bboxMin[c*16+2*p]
					ix.bpsoaMin[di+1] = ix.bboxMin[c*16+2*p+1]
					ix.bpsoaMax[di] = ix.bboxMax[c*16+2*p]
					ix.bpsoaMax[di+1] = ix.bboxMax[c*16+2*p+1]
				} else {
					ix.bpsoaMin[di] = math.MaxInt16
					ix.bpsoaMin[di+1] = math.MaxInt16
					ix.bpsoaMax[di] = math.MinInt16
					ix.bpsoaMax[di+1] = math.MinInt16
				}
			}
		}
	}
}
