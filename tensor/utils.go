package tensor

import "fmt"

// SameShape checks if two tensors have the same shape.
func SameShape[T Numeric](a, b *Tensor[T]) bool {
	if a.Dims() != b.Dims() {
		return false
	}
	for i := 0; i < a.Dims(); i++ {
		if a.shape[i] != b.shape[i] {
			return false
		}
	}
	return true
}

// BroadcastShapes computes the output shape for a broadcasting operation.
func BroadcastShapes(a, b []int) (shape []int, broadcastA, broadcastB bool, err error) {
	lenA, lenB := len(a), len(b)
	maxLen := lenA
	if lenB > maxLen {
		maxLen = lenB
	}
	shape = make([]int, maxLen)
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
			return nil, false, false, fmt.Errorf("tensor shapes %v and %v are not compatible for broadcasting", a, b)
		}
		if dimA > dimB {
			shape[maxLen-i] = dimA
		} else {
			shape[maxLen-i] = dimB
		}
	}
	broadcastA = len(a) < len(shape) || !shapesEqual(a, shape)
	broadcastB = len(b) < len(shape) || !shapesEqual(b, shape)
	return
}

// BroadcastIndex computes the index into a tensor for a broadcasting operation.
func BroadcastIndex(index int, shape, outputShape []int, broadcast bool) int {
	if !broadcast {
		return index
	}
	outputStrides := strides(outputShape)
	inputStrides := strides(shape)
	inputIndex := 0
	for i := range shape {
		dim := shape[len(shape)-1-i]
		outputCoord := (index / outputStrides[len(outputShape)-1-i]) % outputShape[len(outputShape)-1-i]
		inputCoord := outputCoord
		if dim == 1 {
			inputCoord = 0
		}
		inputIndex += inputCoord * inputStrides[len(shape)-1-i]
	}
	return inputIndex
}

func shapesEqual(a, b []int) bool {
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

func strides(shape []int) []int {
	s := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = stride
		stride *= shape[i]
	}
	return s
}
