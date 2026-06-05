package search

import (
	"bufio"
	"encoding/binary"
	"os"
	"unsafe"
)

const (
	NDims      = 14
	NPairs     = 7   // 7 pairs of dims: (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13)
	Scale      = 10000
	MaxK       = 2048 // maximum clusters per partition
	KNeighbors = 5

	// NPartitions is the 4-bit tag space: card_present<<3 | is_online<<2 | unknown<<1 | has_last
	NPartitions = 16

	// Packed-key bit layout for topkK: (dist << IdxBits) | vecIdx
	IdxBits = 22 // supports up to 4M vectors per partition
	// Packed-key bit layout for cluster: (lb << CidBits) | clusterID
	CidBits = 12 // supports up to 4096 clusters
	CidMask = 0xFFF

	// Probe budget
	NProbeInitial   = 12
	NProbeRepairMin = 1
	NProbeRepairMax = 4

	magic = "RNH4-IDX"
)

func bytesOf[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(s[0])))
}

func writePadded(w *bufio.Writer, b []byte) error {
	if _, err := w.Write(b); err != nil {
		return err
	}
	if pad := (64 - len(b)%64) % 64; pad > 0 {
		var z [64]byte
		if _, err := w.Write(z[:pad]); err != nil {
			return err
		}
	}
	return nil
}

// WriteIndexBin serializes one partition's flat IVF index to path.
// On-disk layout (sections zero-padded to 64 bytes, except labels+tail):
//
//	header           64B     magic | n_clusters | n_vectors
//	cluster_offsets  (K+1)*4
//	bbox_min         K*16*2  int16[16] per cluster
//	bbox_max         K*16*2
//	pair_arr[0..6]   n*4     two int16s packed per int32 (SoA)
//	labels           n       uint8 per vector (1=fraud)
//	tail pad         64B     allows safe SIMD over-read on last batch
func WriteIndexBin(
	path string,
	n int,
	clusterOffsets []uint32,
	bboxMin, bboxMax []int16,
	pairArr [NPairs][]int32,
	labels []uint8,
) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	var hdr [64]byte
	copy(hdr[0:8], magic)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(len(clusterOffsets)-1))
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(n))
	if err := writePadded(w, hdr[:]); err != nil {
		return err
	}
	if err := writePadded(w, bytesOf(clusterOffsets)); err != nil {
		return err
	}
	if err := writePadded(w, bytesOf(bboxMin)); err != nil {
		return err
	}
	if err := writePadded(w, bytesOf(bboxMax)); err != nil {
		return err
	}
	for p := 0; p < NPairs; p++ {
		if err := writePadded(w, bytesOf(pairArr[p])); err != nil {
			return err
		}
	}
	if _, err := w.Write(labels); err != nil {
		return err
	}
	var tail [64]byte
	if _, err := w.Write(tail[:]); err != nil {
		return err
	}
	return w.Flush()
}
