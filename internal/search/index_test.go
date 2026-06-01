package search

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func writeIVFBinary(idx *IVFIndex) []byte {
	var buf bytes.Buffer
	buf.Write([]byte(ivfMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(idx.C))
	binary.Write(&buf, binary.LittleEndian, uint32(idx.N))
	for _, v := range idx.Centroids {
		binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	for _, v := range idx.Starts {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	for _, v := range idx.Sizes {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	for _, v := range idx.Vectors {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	buf.Write(idx.Labels)
	return buf.Bytes()
}

func TestLoadIVFIndex_Basic(t *testing.T) {
	idx := &IVFIndex{
		C:         1,
		N:         2,
		Centroids: make([]float32, 16),
		Starts:    []uint32{0},
		Sizes:     []uint32{2},
		Vectors:   make([]int16, 2*16),
		Labels:    []uint8{0, 1},
	}
	for j := 0; j < 16; j++ {
		idx.Vectors[16+j] = 10000
	}

	tmp := t.TempDir() + "/test.bin"
	if err := os.WriteFile(tmp, writeIVFBinary(idx), 0644); err != nil {
		t.Fatal(err)
	}

	got, err := LoadIVFIndex(tmp)
	if err != nil {
		t.Fatalf("LoadIVFIndex error: %v", err)
	}
	if got.N != 2 {
		t.Errorf("N: got %d, want 2", got.N)
	}
	if got.C != 1 {
		t.Errorf("C: got %d, want 1", got.C)
	}
	if got.Labels[0] != 0 || got.Labels[1] != 1 {
		t.Errorf("Labels: got %v, want [0 1]", got.Labels)
	}
	if got.Vectors[0] != 0 {
		t.Errorf("Vectors[0]: got %d, want 0", got.Vectors[0])
	}
	if got.Vectors[16] != 10000 {
		t.Errorf("Vectors[16]: got %d, want 10000", got.Vectors[16])
	}
}

func TestLoadIVFIndex_SentinelMinus1(t *testing.T) {
	vecs := make([]int16, 16)
	vecs[5] = -10000
	vecs[6] = -10000

	idx := &IVFIndex{
		C:         1,
		N:         1,
		Centroids: make([]float32, 16),
		Starts:    []uint32{0},
		Sizes:     []uint32{1},
		Vectors:   vecs,
		Labels:    []uint8{0},
	}

	tmp := t.TempDir() + "/test.bin"
	os.WriteFile(tmp, writeIVFBinary(idx), 0644)

	got, err := LoadIVFIndex(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if got.Vectors[5] != -10000 {
		t.Errorf("Vectors[5]: got %d, want -10000", got.Vectors[5])
	}
	if got.Vectors[6] != -10000 {
		t.Errorf("Vectors[6]: got %d, want -10000", got.Vectors[6])
	}
}
