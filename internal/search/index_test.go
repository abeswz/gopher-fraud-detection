package search

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

func TestLoadIndex_Basic(t *testing.T) {
	// Build a 2-vector binary: vector0 all-zeros legit, vector1 all-10000 fraud
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, uint32(2))
	for j := 0; j < 14; j++ {
		binary.Write(buf, binary.LittleEndian, int16(0))
	}
	buf.WriteByte(0) // legit
	for j := 0; j < 14; j++ {
		binary.Write(buf, binary.LittleEndian, int16(10000))
	}
	buf.WriteByte(1) // fraud

	tmp := t.TempDir() + "/test.bin"
	if err := os.WriteFile(tmp, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	idx, err := LoadIndex(tmp)
	if err != nil {
		t.Fatalf("LoadIndex error: %v", err)
	}
	if idx.N != 2 {
		t.Errorf("N: got %d, want 2", idx.N)
	}
	if idx.Labels[0] != 0 {
		t.Errorf("label[0]: got %d, want 0 (legit)", idx.Labels[0])
	}
	if idx.Labels[1] != 1 {
		t.Errorf("label[1]: got %d, want 1 (fraud)", idx.Labels[1])
	}
	// vector 1 (offset 14) dim 0 should be 10000
	if idx.Vectors[14] != 10000 {
		t.Errorf("Vectors[14]: got %d, want 10000", idx.Vectors[14])
	}
	// vector 0 dim 0 should be 0
	if idx.Vectors[0] != 0 {
		t.Errorf("Vectors[0]: got %d, want 0", idx.Vectors[0])
	}
}

func TestLoadIndex_SentinelMinus1(t *testing.T) {
	// vector with -1 sentinel (int16 = -10000)
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, uint32(1))
	for j := 0; j < 14; j++ {
		if j == 5 || j == 6 {
			binary.Write(buf, binary.LittleEndian, int16(-10000))
		} else {
			binary.Write(buf, binary.LittleEndian, int16(0))
		}
	}
	buf.WriteByte(0)

	tmp := t.TempDir() + "/test.bin"
	os.WriteFile(tmp, buf.Bytes(), 0644)

	idx, err := LoadIndex(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if idx.Vectors[5] != -10000 {
		t.Errorf("Vectors[5]: got %d, want -10000", idx.Vectors[5])
	}
	if idx.Vectors[6] != -10000 {
		t.Errorf("Vectors[6]: got %d, want -10000", idx.Vectors[6])
	}
}
