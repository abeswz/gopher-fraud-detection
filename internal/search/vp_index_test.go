package search

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// writeVPBinary serializes a VPIndex to the VPT1 wire format.
func writeVPBinary(nodes []VPNode, vectors []int16, labels []uint8) []byte {
	n := len(labels)
	nodeCount := len(nodes)
	var buf bytes.Buffer
	buf.Write([]byte(vpMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(n))
	binary.Write(&buf, binary.LittleEndian, uint32(nodeCount))
	binary.Write(&buf, binary.LittleEndian, uint32(16)) // leafSize
	for _, nd := range nodes {
		binary.Write(&buf, binary.LittleEndian, nd.Tau)
		binary.Write(&buf, binary.LittleEndian, nd.ChildOff)
		binary.Write(&buf, binary.LittleEndian, nd.Count)
		buf.Write([]byte{0, 0}) // _pad
		for _, v := range nd.Vec {
			binary.Write(&buf, binary.LittleEndian, v)
		}
	}
	for _, v := range vectors {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	buf.Write(labels)
	return buf.Bytes()
}

func TestLoadVPIndex(t *testing.T) {
	var pivotVec [14]int16
	pivotVec[0] = 5000

	nodes := []VPNode{
		{Tau: 0.5, ChildOff: 2, Count: 0, Vec: pivotVec},
		{Tau: 0.0, ChildOff: 0, Count: 5},
		{Tau: 0.0, ChildOff: 5, Count: 5},
	}

	vectors := make([]int16, 10*14)
	for i := 0; i < 5; i++ {
		vectors[i*14] = 1000
	}
	for i := 5; i < 10; i++ {
		vectors[i*14] = 8000
	}

	labels := []uint8{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}

	tmp := t.TempDir() + "/vp.bin"
	if err := os.WriteFile(tmp, writeVPBinary(nodes, vectors, labels), 0644); err != nil {
		t.Fatal(err)
	}

	got, err := LoadVPIndex(tmp)
	if err != nil {
		t.Fatalf("LoadVPIndex error: %v", err)
	}

	if len(got.Nodes) != 3 {
		t.Errorf("nodeCount: got %d, want 3", len(got.Nodes))
	}
	if len(got.Labels) != 10 {
		t.Errorf("N: got %d, want 10", len(got.Labels))
	}

	root := got.Nodes[0]
	if math.Abs(float64(root.Tau-0.5)) > 1e-6 {
		t.Errorf("root.Tau: got %v, want 0.5", root.Tau)
	}
	if root.ChildOff != 2 {
		t.Errorf("root.ChildOff: got %d, want 2", root.ChildOff)
	}
	if root.Count != 0 {
		t.Errorf("root.Count: got %d, want 0 (internal)", root.Count)
	}
	if root.Vec[0] != 5000 {
		t.Errorf("root.Vec[0]: got %d, want 5000", root.Vec[0])
	}

	left := got.Nodes[1]
	if left.Count != 5 {
		t.Errorf("left.Count: got %d, want 5", left.Count)
	}
	if left.ChildOff != 0 {
		t.Errorf("left.ChildOff: got %d, want 0", left.ChildOff)
	}

	if got.Vectors[0] != 1000 {
		t.Errorf("Vectors[0]: got %d, want 1000", got.Vectors[0])
	}
	if got.Vectors[5*14] != 8000 {
		t.Errorf("Vectors[5*14]: got %d, want 8000", got.Vectors[5*14])
	}

	if got.Labels[1] != 1 {
		t.Errorf("Labels[1]: got %d, want 1", got.Labels[1])
	}
}

func TestLoadVPIndexBadMagic(t *testing.T) {
	nodes := []VPNode{{Tau: 0, ChildOff: 0, Count: 1}}
	vectors := make([]int16, 14)
	labels := []uint8{0}
	data := writeVPBinary(nodes, vectors, labels)
	copy(data[0:4], []byte("BAD!"))

	tmp := t.TempDir() + "/bad.bin"
	os.WriteFile(tmp, data, 0644)

	_, err := LoadVPIndex(tmp)
	if err == nil {
		t.Fatal("expected error for bad magic, got nil")
	}
}

func TestLoadVPIndexSizeMismatch(t *testing.T) {
	nodes := []VPNode{{Tau: 0, ChildOff: 0, Count: 5}}
	vectors := make([]int16, 5*14)
	labels := []uint8{0, 0, 0, 0, 0}
	data := writeVPBinary(nodes, vectors, labels)

	tmp := t.TempDir() + "/trunc.bin"
	os.WriteFile(tmp, data[:len(data)-10], 0644)

	_, err := LoadVPIndex(tmp)
	if err == nil {
		t.Fatal("expected error for truncated file, got nil")
	}
}
