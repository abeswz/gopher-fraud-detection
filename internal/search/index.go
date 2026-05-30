package search

import (
	"encoding/binary"
	"fmt"
	"os"
)

type Index struct {
	Vectors []int16
	Labels  []uint8
	N       int
}

func LoadIndex(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("index file too small: %d bytes", len(data))
	}

	n := int(binary.LittleEndian.Uint32(data[0:4]))

	const recordSize = 14*2 + 1
	expected := 4 + n*recordSize
	if len(data) != expected {
		return nil, fmt.Errorf("index size mismatch: got %d bytes, want %d", len(data), expected)
	}

	vectors := make([]int16, n*14)
	labels := make([]uint8, n)

	offset := 4
	for i := 0; i < n; i++ {
		for j := 0; j < 14; j++ {
			vectors[i*14+j] = int16(binary.LittleEndian.Uint16(data[offset : offset+2]))
			offset += 2
		}
		labels[i] = data[offset]
		offset++
	}

	return &Index{Vectors: vectors, Labels: labels, N: n}, nil
}
