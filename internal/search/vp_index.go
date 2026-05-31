package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const vpMagic = "VPT1"
const vpNodeSize = 40 // 4+4+2+2+28

// VPNode is one node in the VP-tree, stored in DFS order.
// Layout matches the VPT1 binary format exactly (40 bytes, little-endian).
type VPNode struct {
	Tau      float32
	ChildOff uint32
	Count    uint16
	pad      [2]byte
	Vec      [14]int16
}

// VPIndex is the in-memory VP-tree exact-search index.
type VPIndex struct {
	Nodes   []VPNode
	Vectors []int16
	Labels  []uint8
}

// LoadVPIndex reads a VPT1-format binary file and returns a ready-to-query VPIndex.
func LoadVPIndex(path string) (*VPIndex, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 16 {
		return nil, fmt.Errorf("vp index too small: %d bytes", len(data))
	}

	if string(data[0:4]) != vpMagic {
		return nil, fmt.Errorf("bad magic: %q (want %q)", data[0:4], vpMagic)
	}

	n := int(binary.LittleEndian.Uint32(data[4:8]))
	nodeCount := int(binary.LittleEndian.Uint32(data[8:12]))
	// data[12:16] is leafSize — stored but not needed at query time

	expected := 16 + nodeCount*vpNodeSize + n*dims*2 + n
	if len(data) != expected {
		return nil, fmt.Errorf("size mismatch: got %d, want %d", len(data), expected)
	}

	off := 16

	nodes := make([]VPNode, nodeCount)
	for i := range nodes {
		nodes[i].Tau = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		nodes[i].ChildOff = binary.LittleEndian.Uint32(data[off+4:])
		nodes[i].Count = binary.LittleEndian.Uint16(data[off+8:])
		// [off+10 : off+12] is pad — skip
		for j := 0; j < dims; j++ {
			nodes[i].Vec[j] = int16(binary.LittleEndian.Uint16(data[off+12+j*2:]))
		}
		off += vpNodeSize
	}

	vectors := make([]int16, n*dims)
	for i := range vectors {
		vectors[i] = int16(binary.LittleEndian.Uint16(data[off:]))
		off += 2
	}

	labels := make([]uint8, n)
	copy(labels, data[off:off+n])

	return &VPIndex{
		Nodes:   nodes,
		Vectors: vectors,
		Labels:  labels,
	}, nil
}
