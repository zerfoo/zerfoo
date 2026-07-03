package attention

import (
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestTryFlashDecodeBailouts(t *testing.T) {
	tests := []struct {
		name          string
		qShape        []int
		kShape        []int
		vShape        []int
		headDim       int
		numQueryHeads int
		numKVHeads    int
	}{
		{
			name:          "CPU tensors",
			qShape:        []int{4, 1, 64},
			kShape:        []int{4, 32, 64},
			vShape:        []int{4, 32, 64},
			headDim:       64,
			numQueryHeads: 4,
			numKVHeads:    4,
		},
		{
			name:          "seqLen > 1 (prefill, not decode)",
			qShape:        []int{4, 8, 64},
			kShape:        []int{4, 8, 64},
			vShape:        []int{4, 8, 64},
			headDim:       64,
			numQueryHeads: 4,
			numKVHeads:    4,
		},
		{
			name:          "headDim > 128",
			qShape:        []int{4, 1, 256},
			kShape:        []int{4, 32, 256},
			vShape:        []int{4, 32, 256},
			headDim:       256,
			numQueryHeads: 4,
			numKVHeads:    4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q, err := tensor.New(tt.qShape, make([]float32, product(tt.qShape)))
			if err != nil {
				t.Fatalf("tensor Q: %v", err)
			}
			k, err := tensor.New(tt.kShape, make([]float32, product(tt.kShape)))
			if err != nil {
				t.Fatalf("tensor K: %v", err)
			}
			v, err := tensor.New(tt.vShape, make([]float32, product(tt.vShape)))
			if err != nil {
				t.Fatalf("tensor V: %v", err)
			}

			result, err := tryFlashDecode(q, k, v, tt.headDim, tt.numQueryHeads, tt.numKVHeads, nil, nil, nil, nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result != nil {
				t.Fatal("expected nil result (bail out)")
			}
		})
	}
}

func product(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}
