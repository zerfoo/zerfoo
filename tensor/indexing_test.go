package tensor

import (
	"reflect"
	"testing"
)

func TestAt(t *testing.T) {
	tensor, _ := New([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})

	t.Run("ValidIndex", func(t *testing.T) {
		val, err := tensor.At(1, 1)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if val != 5 {
			t.Errorf("expected value 5, got %d", val)
		}
	})

	t.Run("InvalidIndexCount", func(t *testing.T) {
		_, err := tensor.At(1)
		if err == nil {
			t.Fatal("expected an error for wrong number of indices, got nil")
		}
	})

	t.Run("IndexOutOfBounds", func(t *testing.T) {
		_, err := tensor.At(2, 0)
		if err == nil {
			t.Fatal("expected an error for out-of-bounds index, got nil")
		}
	})

	t.Run("NegativeIndex", func(t *testing.T) {
		_, err := tensor.At(-1, 0)
		if err == nil {
			t.Fatal("expected an error for negative index, got nil")
		}
	})
}

func TestSet(t *testing.T) {
	tensor, _ := New([]int{2, 2}, []int{1, 2, 3, 4})

	err := tensor.Set(99, 1, 0)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	val, _ := tensor.At(1, 0)
	if val != 99 {
		t.Errorf("expected value 99 after Set, got %d", val)
	}

	t.Run("SetIndexOutOfBounds", func(t *testing.T) {
		err := tensor.Set(100, 3, 0)
		if err == nil {
			t.Fatal("expected an error for out-of-bounds index, got nil")
		}
	})

	t.Run("SetOnView", func(t *testing.T) {
		slice, _ := tensor.Slice([2]int{0, 1}, [2]int{0, 1})
		err := slice.Set(100, 0, 0)
		if err == nil {
			t.Fatal("expected an error for setting on a view, got nil")
		}
	})

	t.Run("SetInvalidIndexCount", func(t *testing.T) {
		err := tensor.Set(100, 0)
		if err == nil {
			t.Fatal("expected an error for wrong number of indices, got nil")
		}
	})

	t.Run("SetNegativeIndex", func(t *testing.T) {
		err := tensor.Set(100, -1, 0)
		if err == nil {
			t.Fatal("expected an error for negative index, got nil")
		}
	})
}

func TestSlice(t *testing.T) {
	tensor, _ := New([]int{3, 4}, []int{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
	})

	t.Run("ValidSlice", func(t *testing.T) {
		slice, err := tensor.Slice([2]int{1, 3}, [2]int{0, 2})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		expectedShape := []int{2, 2}
		if !reflect.DeepEqual(slice.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, slice.Shape())
		}

		// Check if the view accesses the correct data
		val1, _ := slice.At(0, 0) // Should be tensor.At(1, 0) -> 4
		val2, _ := slice.At(1, 1) // Should be tensor.At(2, 1) -> 9
		if val1 != 4 || val2 != 9 {
			t.Errorf("slice data is incorrect. Expected (4, 9), got (%d, %d)", val1, val2)
		}

		// Test that modifying the slice affects the original tensor
		err = slice.Set(99, 0, 0)
		if err == nil {
			t.Fatalf("setting value on slice should have failed as it is a view")
		}
		originalVal, _ := tensor.At(1, 0)
		if originalVal == 99 {
			t.Errorf("modifying slice affected original tensor. Expected not 99, got %d", originalVal)
		}
	})

	t.Run("InvalidSliceRange", func(t *testing.T) {
		_, err := tensor.Slice([2]int{0, 4}, [2]int{0, 2})
		if err == nil {
			t.Fatal("expected an error for invalid slice range, got nil")
		}
	})

	t.Run("SliceFullTensor", func(t *testing.T) {
		slice, err := tensor.Slice([2]int{0, 3}, [2]int{0, 4})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !reflect.DeepEqual(slice.Shape(), tensor.Shape()) {
			t.Errorf("expected shape %v, got %v", tensor.Shape(), slice.Shape())
		}
	})

	t.Run("InvalidSliceArgs", func(t *testing.T) {
		_, err := tensor.Slice([2]int{1, 0})
		if err == nil {
			t.Fatal("expected an error for invalid slice arguments, got nil")
		}
	})

	t.Run("InvalidSliceDimensions", func(t *testing.T) {
		_, err := tensor.Slice([2]int{0, 1}, [2]int{0, 1}, [2]int{0, 1})
		if err == nil {
			t.Fatal("expected an error for invalid slice dimensions, got nil")
		}
	})
}

func TestSlice_Errors(t *testing.T) {
	tensor, _ := New([]int{3, 4}, []int{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
	})

	t.Run("TooManySliceRanges", func(t *testing.T) {
		_, err := tensor.Slice([2]int{0, 1}, [2]int{0, 1}, [2]int{0, 1})
		if err == nil {
			t.Fatal("expected an error for too many slice ranges, got nil")
		}
	})

	t.Run("InvalidSliceRange_StartLessThanZero", func(t *testing.T) {
		_, err := tensor.Slice([2]int{-1, 2})
		if err == nil {
			t.Fatal("expected an error for invalid slice range, got nil")
		}
	})

	t.Run("InvalidSliceRange_EndGreaterThanSize", func(t *testing.T) {
		_, err := tensor.Slice([2]int{0, 5})
		if err == nil {
			t.Fatal("expected an error for invalid slice range, got nil")
		}
	})

	t.Run("InvalidSliceRange_StartGreaterThanEnd", func(t *testing.T) {
		_, err := tensor.Slice([2]int{2, 1})
		if err == nil {
			t.Fatal("expected an error for invalid slice range, got nil")
		}
	})
}
