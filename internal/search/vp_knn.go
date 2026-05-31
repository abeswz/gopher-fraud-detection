package search

import "math"

type vpStackEntry struct {
	nodeIdx uint32
	minDist float32
}

// KNN finds the k nearest neighbors in the VP-tree using iterative DFS with
// branch-and-bound pruning (triangle inequality). Returns the fraud label count
// among the top-k neighbors. Zero heap allocations per call.
func (idx *VPIndex) KNN(query [14]float32, k int) int {
	var stackArr [40]vpStackEntry
	stack := stackArr[:0]

	var topArr [5]knnEntry
	top := topArr[:0]
	maxDistSq := float32(math.MaxFloat32)
	maxRadius := float32(math.MaxFloat32)
	maxPos := 0

	q0 := query[0]
	q1 := query[1]
	q2 := query[2]
	q3 := query[3]
	q4 := query[4]
	q5 := query[5]
	q6 := query[6]
	q7 := query[7]
	q8 := query[8]
	q9 := query[9]
	q10 := query[10]
	q11 := query[11]
	q12 := query[12]
	q13 := query[13]

	stack = append(stack, vpStackEntry{0, 0})

	nodes := idx.Nodes
	vecs := idx.Vectors
	labs := idx.Labels

	for len(stack) > 0 {
		e := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if len(top) == k && e.minDist >= maxRadius {
			continue
		}

		node := nodes[e.nodeIdx]

		if node.Count > 0 {
			base := int(node.ChildOff) * dims
			end := int(node.ChildOff) + int(node.Count)
			for vi := int(node.ChildOff); vi < end; vi, base = vi+1, base+dims {
				_ = vecs[base+13]

				d0 := q0 - float32(vecs[base])*invScale
				distSq := d0 * d0
				if len(top) == k && distSq >= maxDistSq {
					continue
				}

				d1 := q1 - float32(vecs[base+1])*invScale
				d2 := q2 - float32(vecs[base+2])*invScale
				d3 := q3 - float32(vecs[base+3])*invScale
				d4 := q4 - float32(vecs[base+4])*invScale
				d5 := q5 - float32(vecs[base+5])*invScale
				d6 := q6 - float32(vecs[base+6])*invScale
				d7 := q7 - float32(vecs[base+7])*invScale
				distSq += d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7
				if len(top) == k && distSq >= maxDistSq {
					continue
				}

				d8 := q8 - float32(vecs[base+8])*invScale
				d9 := q9 - float32(vecs[base+9])*invScale
				d10 := q10 - float32(vecs[base+10])*invScale
				d11 := q11 - float32(vecs[base+11])*invScale
				d12 := q12 - float32(vecs[base+12])*invScale
				d13 := q13 - float32(vecs[base+13])*invScale
				distSq += d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

				if len(top) < k {
					top = append(top, knnEntry{distSq, labs[vi]})
					if len(top) == k {
						maxDistSq, maxPos = knnFindMax(top)
						maxRadius = float32(math.Sqrt(float64(maxDistSq)))
					}
				} else if distSq < maxDistSq {
					top[maxPos] = knnEntry{distSq, labs[vi]}
					maxDistSq, maxPos = knnFindMax(top)
					maxRadius = float32(math.Sqrt(float64(maxDistSq)))
				}
			}
			continue
		}

		v := node.Vec
		_ = v[13]
		d0 := q0 - float32(v[0])*invScale
		d1 := q1 - float32(v[1])*invScale
		d2 := q2 - float32(v[2])*invScale
		d3 := q3 - float32(v[3])*invScale
		d4 := q4 - float32(v[4])*invScale
		d5 := q5 - float32(v[5])*invScale
		d6 := q6 - float32(v[6])*invScale
		d7 := q7 - float32(v[7])*invScale
		d8 := q8 - float32(v[8])*invScale
		d9 := q9 - float32(v[9])*invScale
		d10 := q10 - float32(v[10])*invScale
		d11 := q11 - float32(v[11])*invScale
		d12 := q12 - float32(v[12])*invScale
		d13 := q13 - float32(v[13])*invScale
		distSq := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
		dist := float32(math.Sqrt(float64(distSq)))
		tau := node.Tau

		leftChild := e.nodeIdx + 1
		rightChild := node.ChildOff

		if dist <= tau {
			outerMin := tau - dist
			if outerMin < maxRadius {
				stack = append(stack, vpStackEntry{rightChild, outerMin})
			}
			stack = append(stack, vpStackEntry{leftChild, 0})
		} else {
			innerMin := dist - tau
			if innerMin < maxRadius {
				stack = append(stack, vpStackEntry{leftChild, innerMin})
			}
			stack = append(stack, vpStackEntry{rightChild, 0})
		}
	}

	fraudCount := 0
	for _, e := range top {
		if e.label == 1 {
			fraudCount++
		}
	}
	return fraudCount
}
