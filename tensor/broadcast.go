// Package tensor provides a multi-dimensional array (tensor) implementation.
package tensor

import "fmt"

// BroadcastShapes computes the resulting shape of a broadcast operation between two shapes.
func BroadcastShapes(a, b []int) (shape []int, broadcastA, broadcastB bool, err error) {
	lenA := len(a)
	lenB := len(b)
	maxLen := lenA
	if lenB > maxLen {
		maxLen = lenB
	}
	result := make([]int, maxLen)

	for i := 1; i <= maxLen; i++ {
		dimA := 1
		if i <= lenA {
			dimA = a[lenA-i]
		}
		dimB := 1
		if i <= lenB {
			dimB = b[lenB-i]
		}

		if dimA != dimB && dimA != 1 && dimB != 1 {
			return nil, false, false, fmt.Errorf("shapes %v and %v are not broadcast compatible (dimension %d: %d vs %d)", a, b, i, dimA, dimB)
		}

		if dimA > dimB {
			result[maxLen-i] = dimA
		} else {
			result[maxLen-i] = dimB
		}
	}

	return result, !SameShape(a, result), !SameShape(b, result), nil
}

// BroadcastIndex computes the index in the original tensor for a given index in the broadcasted tensor.
func BroadcastIndex(index int, shape, outputShape []int, broadcast bool) int {
	if !broadcast {
		return index
	}

	outputStrides := strides(outputShape)
	originalStrides := strides(shape)
	originalIndex := 0
	for i := 0; i < len(outputShape); i++ {
		coord := (index / outputStrides[i]) % outputShape[i]
		shapeI := len(shape) - 1 - (len(outputShape) - 1 - i)
		if shapeI >= 0 && shape[shapeI] != 1 {
			originalIndex += coord * originalStrides[shapeI]
		}
	}
	return originalIndex
}

// SameShape checks if two shapes are identical.
func SameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// strides computes the strides for a given shape.
func strides(shape []int) []int {
	s := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = stride
		stride *= shape[i]
	}
	return s
}
