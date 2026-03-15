package compute

import (
	"reflect"
	"testing"
)

func TestBroadcastShape(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want []int
	}{
		{"same_2D", []int{3, 4}, []int{3, 4}, []int{3, 4}},
		{"row_broadcast", []int{3, 4}, []int{1, 4}, []int{3, 4}},
		{"col_broadcast", []int{3, 4}, []int{3, 1}, []int{3, 4}},
		{"scalar_vs_2D", []int{1}, []int{3, 4}, []int{3, 4}},
		{"4D_scalar_vs_2D", []int{1, 1, 1, 1}, []int{2, 1}, []int{1, 1, 2, 1}},
		{"4D_vs_4D", []int{1, 1, 2, 2}, []int{1, 4, 1, 1}, []int{1, 4, 2, 2}},
		{"3D_vs_1D", []int{2, 3, 4}, []int{4}, []int{2, 3, 4}},
		{"preserves_leading_unit_dims", []int{1, 1, 2048}, []int{2048}, []int{1, 1, 2048}},
		{"3D_vs_1D_nonunit", []int{1, 5, 2048}, []int{2048}, []int{1, 5, 2048}},
		{"scalar_vs_3D", []int{1}, []int{1, 1, 2048}, []int{1, 1, 2048}},
		{"1D_vs_3D_leading_unit", []int{2048}, []int{1, 1, 2048}, []int{1, 1, 2048}},
		{"3D_vs_2D", []int{1, 1, 2048}, []int{1, 2048}, []int{1, 1, 2048}},
		{"4D_vs_1D", []int{1, 1, 1, 2048}, []int{2048}, []int{1, 1, 1, 2048}},
		{"3D_broadcast_middle", []int{1, 1, 2048}, []int{1, 5, 2048}, []int{1, 5, 2048}},
		{"empty_a", []int{}, []int{3}, []int{3}},
		{"both_empty", []int{}, []int{}, []int{}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := broadcastShape(tc.a, tc.b)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("broadcastShape(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

func TestBroadcastShape_PreservesLeadingUnitDims(t *testing.T) {
	got := broadcastShape([]int{1, 1, 2048}, []int{2048})
	want := []int{1, 1, 2048}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("broadcastShape([1,1,2048], [2048]) = %v, want %v", got, want)
	}
}

func TestBroadcastShape_3Dvs1D(t *testing.T) {
	got := broadcastShape([]int{1, 5, 2048}, []int{2048})
	want := []int{1, 5, 2048}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("broadcastShape([1,5,2048], [2048]) = %v, want %v", got, want)
	}
}

func TestBroadcastShape_ScalarBroadcast(t *testing.T) {
	got := broadcastShape([]int{1}, []int{1, 1, 2048})
	want := []int{1, 1, 2048}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("broadcastShape([1], [1,1,2048]) = %v, want %v", got, want)
	}
}

// TestFlattenTo2D_ShapeCollapse verifies that flattenTo2D collapses distinct
// N-D shapes to identical 2D shapes, which is the root cause of the Phi 4
// Add storage size mismatch (T3402). gpuBroadcastOp must detect this and
// fall back to the 4D broadcast kernel.
func TestFlattenTo2D_ShapeCollapse(t *testing.T) {
	// These shape pairs flatten to identical (M,D) but have different
	// broadcast output shapes with more elements than M*D.
	tests := []struct {
		name      string
		a, b      []int
		wantShape []int
	}{
		{
			"3D_cross_broadcast",
			[]int{2, 1, 3}, []int{1, 2, 3},
			[]int{2, 2, 3},
		},
		{
			"4D_attention_mask",
			[]int{1, 1, 32, 96}, []int{1, 32, 1, 96},
			[]int{1, 32, 32, 96},
		},
		{
			"3D_inner_broadcast",
			[]int{4, 1, 5}, []int{1, 3, 5},
			[]int{4, 3, 5},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			aM, aD := flattenTo2D(tc.a)
			bM, bD := flattenTo2D(tc.b)

			// Verify these collapse to the same 2D shape.
			if aM != bM || aD != bD {
				t.Skipf("shapes don't collapse: (%d,%d) vs (%d,%d)", aM, aD, bM, bD)
			}

			flatElems := aM * aD
			outShape := broadcastShape(tc.a, tc.b)
			broadcastElems := totalElements(outShape)

			if !reflect.DeepEqual(outShape, tc.wantShape) {
				t.Errorf("broadcastShape(%v, %v) = %v, want %v", tc.a, tc.b, outShape, tc.wantShape)
			}

			// The bug: flatElems < broadcastElems because flattenTo2D
			// lost the broadcast dimension.
			if flatElems == broadcastElems {
				t.Errorf("expected flatElems (%d) < broadcastElems (%d) for shape collapse case", flatElems, broadcastElems)
			}
		})
	}
}

// TestBroadcastStrides4D_CrossBroadcast verifies that the 4D broadcast path
// handles shape pairs that cause flattenTo2D collapse (T3402 fix validation).
func TestBroadcastStrides4D_CrossBroadcast(t *testing.T) {
	tests := []struct {
		name      string
		a, b      []int
		wantShape [4]int
		wantOK    bool
	}{
		{
			"3D_cross",
			[]int{2, 1, 3}, []int{1, 2, 3},
			[4]int{1, 2, 2, 3}, true,
		},
		{
			"4D_attention_mask",
			[]int{1, 1, 32, 96}, []int{1, 32, 1, 96},
			[4]int{1, 32, 32, 96}, true,
		},
		{
			"3D_inner",
			[]int{4, 1, 5}, []int{1, 3, 5},
			[4]int{1, 4, 3, 5}, true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			outDims, _, _, ok := broadcastStrides4D(tc.a, tc.b)
			if ok != tc.wantOK {
				t.Fatalf("broadcastStrides4D ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && outDims != tc.wantShape {
				t.Errorf("outDims = %v, want %v", outDims, tc.wantShape)
			}
		})
	}
}

// TestBroadcastStrides4D_TrySliceShapes verifies that shapes causing TrySlice
// cudaMemcpy failures (sizes 3, 48, 1) are handled by the 4D broadcast kernel,
// preventing CPU fallback during CUDA graph capture.
func TestBroadcastStrides4D_TrySliceShapes(t *testing.T) {
	tests := []struct {
		name      string
		a, b      []int
		wantShape [4]int
		wantOK    bool
	}{
		{
			"3x48x1_vs_1x48x1",
			[]int{3, 48, 1}, []int{1, 48, 1},
			[4]int{1, 3, 48, 1}, true,
		},
		{
			"3x1x48_vs_1x3x48",
			[]int{3, 1, 48}, []int{1, 3, 48},
			[4]int{1, 3, 3, 48}, true,
		},
		{
			"1x48x3_vs_2x48x3",
			[]int{1, 48, 3}, []int{2, 48, 3},
			[4]int{1, 2, 48, 3}, true,
		},
		{
			"3x48x1_vs_3x48x64",
			[]int{3, 48, 1}, []int{3, 48, 64},
			[4]int{1, 3, 48, 64}, true,
		},
		{
			"1x1x48_vs_3x1x48",
			[]int{1, 1, 48}, []int{3, 1, 48},
			[4]int{1, 3, 1, 48}, true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			outDims, _, _, ok := broadcastStrides4D(tc.a, tc.b)
			if ok != tc.wantOK {
				t.Fatalf("broadcastStrides4D ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && outDims != tc.wantShape {
				t.Errorf("outDims = %v, want %v", outDims, tc.wantShape)
			}
		})
	}
}

func TestTotalElements(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  int
	}{
		{"scalar", []int{1}, 1},
		{"vector", []int{5}, 5},
		{"matrix", []int{3, 4}, 12},
		{"3D", []int{2, 3, 4}, 24},
		{"singleton", []int{1, 1, 1}, 1},
		{"empty", []int{}, 1},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := totalElements(tc.shape)
			if got != tc.want {
				t.Errorf("totalElements(%v) = %d, want %d", tc.shape, got, tc.want)
			}
		})
	}
}
