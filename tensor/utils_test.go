package tensor

import (
	"reflect"
	"testing"
)

func TestSameShape(t *testing.T) {
	a, _ := New[int]([]int{2, 3}, nil)
	b, _ := New[int]([]int{2, 3}, nil)
	c, _ := New[int]([]int{3, 2}, nil)
	d, _ := New[int]([]int{2, 3, 1}, nil)

	if !SameShape(a.Shape(), b.Shape()) {
		t.Error("expected SameShape to return true for same shapes")
	}
	if SameShape(a.Shape(), c.Shape()) {
		t.Error("expected SameShape to return false for different shapes")
	}
	if SameShape(a.Shape(), d.Shape()) {
		t.Error("expected SameShape to return false for different dimensions")
	}
	if SameShape(d.Shape(), a.Shape()) {
		t.Error("expected SameShape to return false for different dimensions")
	}
}

func TestBroadcastShapes(t *testing.T) {
	cases := []struct {
		a, b, expected         []int
		broadcastA, broadcastB bool
		err                    bool
	}{
		{[]int{2, 3}, []int{2, 3}, []int{2, 3}, false, false, false},
		{[]int{2, 1}, []int{2, 3}, []int{2, 3}, true, false, false},
		{[]int{2, 3}, []int{1, 3}, []int{2, 3}, false, true, false},
		{[]int{4, 1}, []int{1, 3}, []int{4, 3}, true, true, false},
		{[]int{4}, []int{2, 4}, []int{2, 4}, true, false, false},
		{[]int{2, 3}, []int{2, 4}, nil, false, false, true},
		{[]int{1, 5}, []int{5, 1}, []int{5, 5}, true, true, false},
	}

	for _, tc := range cases {
		shape, broadcastA, broadcastB, err := BroadcastShapes(tc.a, tc.b)
		if (err != nil) != tc.err {
			t.Errorf("BroadcastShapes(%v, %v): expected error %v, got %v", tc.a, tc.b, tc.err, err)
		}
		if err == nil {
			if !reflect.DeepEqual(shape, tc.expected) {
				t.Errorf("BroadcastShapes(%v, %v): expected shape %v, got %v", tc.a, tc.b, tc.expected, shape)
			}
			if broadcastA != tc.broadcastA {
				t.Errorf("BroadcastShapes(%v, %v): expected broadcastA %v, got %v", tc.a, tc.b, tc.broadcastA, broadcastA)
			}
			if broadcastB != tc.broadcastB {
				t.Errorf("BroadcastShapes(%v, %v): expected broadcastB %v, got %v", tc.a, tc.b, tc.broadcastB, broadcastB)
			}
		}
	}
}

func TestBroadcastIndex(t *testing.T) {
	cases := []struct {
		index       int
		shape       []int
		outputShape []int
		broadcast   bool
		expected    int
	}{
		{5, []int{2, 3}, []int{2, 3}, false, 5},
		{5, []int{2, 1}, []int{2, 3}, true, 1},
		{5, []int{1, 3}, []int{2, 3}, true, 2},
		{1, []int{1, 3}, []int{2, 3}, true, 1},
	}

	for _, tc := range cases {
		index := BroadcastIndex(tc.index, tc.shape, tc.outputShape, tc.broadcast)
		if index != tc.expected {
			t.Errorf("BroadcastIndex(%d, %v, %v, %v): expected %d, got %d", tc.index, tc.shape, tc.outputShape, tc.broadcast, tc.expected, index)
		}
	}
}

func TestShapesEqual(t *testing.T) {
	if SameShape([]int{1, 2}, []int{1, 2, 3}) {
		t.Error("expected SameShape to return false for different lengths")
	}
	if !SameShape([]int{1, 2}, []int{1, 2}) {
		t.Error("expected SameShape to return true for identical shapes")
	}
}
