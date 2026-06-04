package search

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// writeIVFHBinary serializes an IVFHIndex to the IVFH on-disk format.
// Used only in tests — mirrors the format that build_index.py writes.
func writeIVFHBinary(idx *IVFHIndex) []byte {
	K1, K2, N := idx.K1, idx.K2, idx.N
	var buf bytes.Buffer
	buf.Write([]byte("IVFH"))
	binary.Write(&buf, binary.LittleEndian, idx.DSafe)
	binary.Write(&buf, binary.LittleEndian, uint32(K1))
	binary.Write(&buf, binary.LittleEndian, uint32(K2))
	binary.Write(&buf, binary.LittleEndian, uint32(N))
	for _, v := range idx.MacroCentroids {
		binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	for _, v := range idx.MicroCentroids {
		binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	for _, v := range idx.Starts {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	for _, v := range idx.Sizes {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	for _, v := range idx.Radii {
		binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	for _, v := range idx.Vectors {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	buf.Write(idx.Labels)
	return buf.Bytes()
}

func makeMinimalIVFHIndex() *IVFHIndex {
	// K1=2, K2=2 → 4 leaf clusters; N=4 (one vector per leaf)
	K1, K2, N := 2, 2, 4
	idx := &IVFHIndex{
		K1: K1, K2: K2, N: N,
		DSafe:          0.5,
		NCoarseProbe:   2,
		MacroCentroids: make([]float32, K1*16),
		MicroCentroids: make([]float32, K1*K2*16),
		Starts:         []uint32{0, 1, 2, 3},
		Sizes:          []uint32{1, 1, 1, 1},
		Radii:          []float32{0.1, 0.1, 0.1, 0.1},
		Vectors:        make([]int16, N*16),
		Labels:         []uint8{0, 1, 0, 1},
	}
	// macro 0 centroid at (0,0,...), macro 1 at (1,0,...)
	idx.MacroCentroids[16] = 1.0 // macro 1, dim 0
	// set vector 1 (leaf 1) to 1.0 in all dims → int16 10000
	for j := 0; j < 16; j++ {
		idx.Vectors[1*16+j] = 10000
	}
	return idx
}

func TestLoadIVFHIndex_Fields(t *testing.T) {
	idx := makeMinimalIVFHIndex()
	data := writeIVFHBinary(idx)

	tmp := t.TempDir() + "/test.ivfh"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		t.Fatal(err)
	}

	got, err := LoadIVFHIndex(tmp, 2)
	if err != nil {
		t.Fatalf("LoadIVFHIndex: %v", err)
	}
	defer got.Close()

	if got.K1 != 2 {
		t.Errorf("K1: got %d, want 2", got.K1)
	}
	if got.K2 != 2 {
		t.Errorf("K2: got %d, want 2", got.K2)
	}
	if got.N != 4 {
		t.Errorf("N: got %d, want 4", got.N)
	}
	if got.DSafe != 0.5 {
		t.Errorf("DSafe: got %f, want 0.5", got.DSafe)
	}
	if got.DSafeSq != 0.25 {
		t.Errorf("DSafeSq: got %f, want 0.25", got.DSafeSq)
	}
	if got.NCoarseProbe != 2 {
		t.Errorf("NCoarseProbe: got %d, want 2", got.NCoarseProbe)
	}
	if got.MacroCentroids[16] != 1.0 {
		t.Errorf("MacroCentroids[16]: got %f, want 1.0", got.MacroCentroids[16])
	}
	if got.Radii[0] != 0.1 {
		t.Errorf("Radii[0]: got %f, want 0.1", got.Radii[0])
	}
	if got.Vectors[1*16] != 10000 {
		t.Errorf("Vectors[16]: got %d, want 10000", got.Vectors[1*16])
	}
	if got.Labels[1] != 1 {
		t.Errorf("Labels[1]: got %d, want 1", got.Labels[1])
	}
}

func TestLoadIVFHIndex_BadMagic(t *testing.T) {
	data := []byte("IVF2\x00\x00\x00\x00")
	tmp := t.TempDir() + "/bad.ivfh"
	os.WriteFile(tmp, data, 0644)
	_, err := LoadIVFHIndex(tmp, 1)
	if err == nil {
		t.Error("expected error for bad magic, got nil")
	}
}

func TestLoadIVFHIndex_SizeMismatch(t *testing.T) {
	idx := makeMinimalIVFHIndex()
	data := writeIVFHBinary(idx)
	data = data[:len(data)-10] // truncate

	tmp := t.TempDir() + "/trunc.ivfh"
	os.WriteFile(tmp, data, 0644)
	_, err := LoadIVFHIndex(tmp, 1)
	if err == nil {
		t.Error("expected error for truncated file, got nil")
	}
}
