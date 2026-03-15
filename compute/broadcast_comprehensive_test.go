package compute

import (
	"reflect"
	"testing"
)

// TestBroadcastShape_Comprehensive covers all ONNX-relevant broadcasting combos
// that arise when fused ops (e.g. RMSNorm) are decomposed into individual ops.
func TestBroadcastShape_Comprehensive(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want []int
	}{
		// 1D vs 3D: weight [2048] broadcast against activation [1,1,2048]
		{"1D_vs_3D_weight_broadcast", []int{2048}, []int{1, 1, 2048}, []int{1, 1, 2048}},
		// 4D attention broadcast: [1,32,1,64] vs [1,1,5,64]
		{"4D_attention_broadcast", []int{1, 32, 1, 64}, []int{1, 1, 5, 64}, []int{1, 32, 5, 64}},
		// scalar [1] vs 3D
		{"scalar_vs_3D", []int{1}, []int{1, 1, 2048}, []int{1, 1, 2048}},
		// 2D cross-broadcast: [5,1] vs [1,13]
		{"2D_cross_broadcast", []int{5, 1}, []int{1, 13}, []int{5, 13}},
		// identity: same 1D
		{"1D_identity", []int{2048}, []int{2048}, []int{2048}},
		// 2D vs 1D: [1,2048] vs [2048]
		{"2D_vs_1D", []int{1, 2048}, []int{2048}, []int{1, 2048}},
		// 3D cross-broadcast: [3,1,2048] vs [1,5,2048]
		{"3D_cross_broadcast", []int{3, 1, 2048}, []int{1, 5, 2048}, []int{3, 5, 2048}},
		// scalar broadcast: [] vs [2048]
		{"empty_vs_1D", []int{}, []int{2048}, []int{2048}},
		// RMSNorm: variance [1,1,1] vs activation [1,1,2048]
		{"variance_broadcast", []int{1, 1, 1}, []int{1, 1, 2048}, []int{1, 1, 2048}},
		// batch broadcast: [1,1,2048] vs [4,1,2048]
		{"batch_broadcast", []int{1, 1, 2048}, []int{4, 1, 2048}, []int{4, 1, 2048}},
		// full 4D: [1,1,1,1] vs [2,32,5,64]
		{"4D_scalar_broadcast", []int{1, 1, 1, 1}, []int{2, 32, 5, 64}, []int{2, 32, 5, 64}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := broadcastShape(tc.a, tc.b)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("broadcastShape(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
			// Broadcasting should be commutative.
			gotReverse := broadcastShape(tc.b, tc.a)
			if !reflect.DeepEqual(gotReverse, tc.want) {
				t.Errorf("broadcastShape(%v, %v) = %v, want %v (reverse)", tc.b, tc.a, gotReverse, tc.want)
			}
		})
	}
}

// TestBroadcastStrides4D_Comprehensive tests broadcastStrides4D with 4D shapes
// used in ONNX attention and normalization patterns.
func TestBroadcastStrides4D_Comprehensive(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []int
		wantDims [4]int
		wantSA   [4]int
		wantSB   [4]int
		wantOK   bool
	}{
		{
			name:     "1D_vs_3D_weight",
			a:        []int{2048},
			b:        []int{1, 1, 2048},
			wantDims: [4]int{1, 1, 1, 2048},
			wantSA:   [4]int{2048, 2048, 2048, 1},
			wantSB:   [4]int{2048, 2048, 2048, 1},
			wantOK:   true,
		},
		{
			name:     "4D_attention_kv_broadcast",
			a:        []int{1, 32, 1, 64},
			b:        []int{1, 1, 5, 64},
			wantDims: [4]int{1, 32, 5, 64},
			// a=[1,32,1,64]: dim2=1 vs out=5 -> stride 0
			wantSA: [4]int{2048, 64, 0, 1},
			// b=[1,1,5,64]: dim1=1 vs out=32 -> stride 0
			wantSB: [4]int{320, 0, 64, 1},
			wantOK: true,
		},
		{
			name:     "2D_cross_broadcast",
			a:        []int{5, 1},
			b:        []int{1, 13},
			wantDims: [4]int{1, 1, 5, 13},
			// a=[5,1] padded [1,1,5,1]: dim3=1 vs out=13 -> stride 0
			wantSA: [4]int{5, 5, 1, 0},
			// b=[1,13] padded [1,1,1,13]: dim2=1 vs out=5 -> stride 0
			wantSB: [4]int{13, 13, 0, 1},
			wantOK: true,
		},
		{
			name:     "3D_batch_sequence",
			a:        []int{3, 1, 2048},
			b:        []int{1, 5, 2048},
			wantDims: [4]int{1, 3, 5, 2048},
			// a=[3,1,2048] padded [1,3,1,2048]: dim2=1 vs out=5 -> stride 0
			wantSA: [4]int{6144, 2048, 0, 1},
			// b=[1,5,2048] padded [1,1,5,2048]: dim1=1 vs out=3 -> stride 0
			wantSB: [4]int{10240, 0, 2048, 1},
			wantOK: true,
		},
		{
			name:     "scalar_vs_3D",
			a:        []int{1},
			b:        []int{1, 1, 2048},
			wantDims: [4]int{1, 1, 1, 2048},
			// a=[1] padded [1,1,1,1]: dim3=1 vs out=2048 -> stride 0
			wantSA: [4]int{1, 1, 1, 0},
			wantSB: [4]int{2048, 2048, 2048, 1},
			wantOK: true,
		},
		{
			name:   "incompatible_dims",
			a:      []int{3, 4},
			b:      []int{5, 4},
			wantOK: false,
		},
		{
			name:   "5D_exceeds_limit",
			a:      []int{1, 2, 3, 4, 5},
			b:      []int{5},
			wantOK: false,
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

// TestFlattenTo2D tests flattenTo2D edge cases used in GPU kernel dispatch.
func TestFlattenTo2D(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		wantM int
		wantD int
	}{
		{"empty_scalar", []int{}, 1, 1},
		{"1D_vector", []int{5}, 1, 5},
		{"2D_matrix", []int{3, 5}, 3, 5},
		{"3D_batch", []int{2, 3, 5}, 6, 5},
		{"3D_unit_batch", []int{1, 1, 2048}, 1, 2048},
		{"4D_attention", []int{1, 32, 5, 64}, 160, 64},
		{"1D_large", []int{2048}, 1, 2048},
		{"4D_unit_dims", []int{1, 1, 1, 2048}, 1, 2048},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, d := flattenTo2D(tc.shape)
			if m != tc.wantM {
				t.Errorf("flattenTo2D(%v) M = %d, want %d", tc.shape, m, tc.wantM)
			}
			if d != tc.wantD {
				t.Errorf("flattenTo2D(%v) D = %d, want %d", tc.shape, d, tc.wantD)
			}
		})
	}
}

// TestTrailingDimsMatch tests trailingDimsMatch used for fast-path broadcasting.
func TestTrailingDimsMatch(t *testing.T) {
	tests := []struct {
		name    string
		longer  []int
		shorter []int
		want    bool
	}{
		{"3D_vs_1D_match", []int{1, 1, 2048}, []int{2048}, true},
		{"3D_vs_2D_match", []int{3, 5, 2048}, []int{5, 2048}, true},
		{"3D_vs_2D_no_match", []int{3, 5, 2048}, []int{4, 2048}, false},
		{"4D_vs_2D_match", []int{1, 32, 5, 64}, []int{5, 64}, true},
		{"4D_vs_3D_match", []int{1, 32, 5, 64}, []int{32, 5, 64}, true},
		{"4D_vs_3D_no_match", []int{1, 32, 5, 64}, []int{16, 5, 64}, false},
		{"equal_length_returns_false", []int{3, 4}, []int{3, 4}, false},
		{"shorter_longer_returns_false", []int{4}, []int{3, 4}, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := trailingDimsMatch(tc.longer, tc.shorter)
			if got != tc.want {
				t.Errorf("trailingDimsMatch(%v, %v) = %v, want %v",
					tc.longer, tc.shorter, got, tc.want)
			}
		})
	}
}
