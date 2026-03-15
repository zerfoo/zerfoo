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
