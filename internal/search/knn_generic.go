//go:build !amd64

package search

func distL2i16_16(vecs []int16, base int, q *[16]float32) float32 {
	_ = vecs[base+15]
	var dist float32
	for i := 0; i < 16; i++ {
		d := float32(vecs[base+i])*invScale - q[i]
		dist += d * d
	}
	return dist
}

func distL2i16q(vecs []int16, base int, q *[16]int16) int32 {
	_ = vecs[base+15]
	var dist int32
	for i := 0; i < 16; i++ {
		d := int32(vecs[base+i]) - int32(q[i])
		dist += d * d
	}
	return dist
}
