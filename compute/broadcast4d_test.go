package compute

import (
	"testing"
)

func TestBroadcastStrides4D(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []int
		wantDims [4]int
		wantSA   [4]int
		wantSB   [4]int
		wantOK   bool
	}{
		{
			name:     "scalar_times_vector",
			a:        []int{1},
			b:        []int{4},
			wantDims: [4]int{1, 1, 1, 4},
			// a=[1] padded to [1,1,1,1], b=[4] padded to [1,1,1,4].
			// a's dim3 is 1 vs out dim3=4: broadcast -> stride 0.
			// Leading dims are 1 in both so strides don't matter (single iteration).
			wantSA: [4]int{1, 1, 1, 0},
			wantSB: [4]int{4, 4, 4, 1},
			wantOK: true,
		},
		{
			name:     "row_broadcast_1N_MN",
			a:        []int{1, 4},
			b:        []int{3, 4},
			wantDims: [4]int{1, 1, 3, 4},
			// a=[1,4] padded to [1,1,1,4]. dim2: a=1 vs out=3 -> stride 0.
			wantSA: [4]int{4, 4, 0, 1},
			wantSB: [4]int{12, 12, 4, 1},
			wantOK: true,
		},
		{
			name:     "col_broadcast_M1_MN",
			a:        []int{3, 1},
			b:        []int{3, 4},
			wantDims: [4]int{1, 1, 3, 4},
			// a=[3,1] padded to [1,1,3,1]. dim3: a=1 vs out=4 -> stride 0.
			wantSA: [4]int{3, 3, 1, 0},
			wantSB: [4]int{12, 12, 4, 1},
			wantOK: true,
		},
		{
			name:     "4D_broadcast",
			a:        []int{2, 1, 3, 1},
			b:        []int{1, 4, 1, 5},
			wantDims: [4]int{2, 4, 3, 5},
			wantSA:   [4]int{3, 0, 1, 0},
			wantSB:   [4]int{0, 5, 0, 1},
			wantOK:   true,
		},
		{
			name:     "3D_broadcast",
			a:        []int{2, 3, 4},
			b:        []int{4},
			wantDims: [4]int{1, 2, 3, 4},
			// a=[2,3,4] padded to [1,2,3,4]. All dims match -> no zeroing.
			wantSA: [4]int{24, 12, 4, 1},
			// b=[4] padded to [1,1,1,4]. dims 1,2: b=1 vs out>1 -> stride 0.
			wantSB: [4]int{4, 0, 0, 1},
			wantOK: true,
		},
		{
			name:   "5D_not_supported",
			a:      []int{2, 3, 4, 5, 6},
			b:      []int{6},
			wantOK: false,
		},
		{
			name:     "same_shape_2D",
			a:        []int{3, 4},
			b:        []int{3, 4},
			wantDims: [4]int{1, 1, 3, 4},
			wantSA:   [4]int{12, 12, 4, 1},
			wantSB:   [4]int{12, 12, 4, 1},
			wantOK:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dims, sa, sb, ok := broadcastStrides4D(tc.a, tc.b)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if !ok {
				return
			}
			if dims != tc.wantDims {
				t.Errorf("dims = %v, want %v", dims, tc.wantDims)
			}
			if sa != tc.wantSA {
				t.Errorf("aStrides = %v, want %v", sa, tc.wantSA)
			}
			if sb != tc.wantSB {
				t.Errorf("bStrides = %v, want %v", sb, tc.wantSB)
			}
		})
	}
}
