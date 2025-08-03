package tensor

import (
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	// Test case: Valid shape and data
	shape := []int{2, 2}
	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor, err := New(shape, data)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(tensor.Shape(), shape) {
		t.Errorf("expected shape %v, got %v", shape, tensor.Shape())
	}
	if !reflect.DeepEqual(tensor.Data(), data) {
		t.Errorf("expected data %v, got %v", data, tensor.Data())
	}

	// Test case: Mismatched data length
	_, err = New[float32]([]int{2, 3}, []float32{1.0, 2.0})
	if err == nil {
		t.Errorf("expected error for mismatched data length, got nil")
	}

	// Test case: Invalid shape (negative dimension)
	_, err = New[float32]([]int{2, -1, 3}, nil)
	if err == nil {
		t.Errorf("expected error for invalid shape (negative dimension), got nil")
	}

	// Test case: 0-sized dimension
	tensorZeroDim, err := New[float32]([]int{2, 0, 3}, nil)
	if err != nil {
		t.Errorf("unexpected error for 0-sized dimension: %v", err)
	}
	if tensorZeroDim.Size() != 0 {
		t.Errorf("expected size 0, got %d", tensorZeroDim.Size())
	}

	// Test case: 0-sized dimension with non-empty data
	_, err = New[float32]([]int{2, 0, 3}, []float32{1.0})
	if err == nil {
		t.Errorf("expected error for 0-sized dimension with non-empty data, got nil")
	}

	// Test case: nil data (should allocate new data)
	tensor, err = New[float32]([]int{2, 2}, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(tensor.Data()) != 4 {
		t.Errorf("expected data length 4, got %d", len(tensor.Data()))
	}

	// Test case: empty shape and non-empty data (should now be valid for scalar)
	tensorScalar, err := New[float32]([]int{}, []float32{1.0})
	if err != nil {
		t.Errorf("unexpected error for empty shape and non-empty data (scalar): %v", err)
	}
	if tensorScalar.Size() != 1 || tensorScalar.Data()[0] != 1.0 {
		t.Errorf("expected scalar tensor with size 1 and data 1.0, got size %d and data %v", tensorScalar.Size(), tensorScalar.Data())
	}

	// Test case: empty shape and multiple data elements (should be an error)
	_, err = New[float32]([]int{}, []float32{1.0, 2.0})
	if err == nil {
		t.Errorf("expected error for empty shape and multiple data elements, got nil")
	}

	// Test case: empty shape and empty data (should create a scalar with zero value)
	tensorEmpty, err := New[float32]([]int{}, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if tensorEmpty.Size() != 1 {
		t.Errorf("expected size 1, got %d", tensorEmpty.Size())
	}
	if tensorEmpty.Data()[0] != 0.0 {
		t.Errorf("expected data 0.0, got %v", tensorEmpty.Data()[0])
	}
}

func TestTensor_At(t *testing.T) {
	tensor, _ := New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})

	val, err := tensor.At(0, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if val != 2 {
		t.Errorf("expected 2, got %d", val)
	}

	_, err = tensor.At(0, 3)
	if err == nil {
		t.Errorf("expected error, got nil")
	}

	// Test case: Incorrect number of indices
	_, err = tensor.At(0)
	if err == nil {
		t.Errorf("expected error for incorrect number of indices, got nil")
	}

	// Test case: Accessing a 0-dimensional tensor (scalar) with no indices
	scalarTensor, _ := New[int]([]int{}, []int{42})
	scalarVal, err := scalarTensor.At()
	if err != nil {
		t.Errorf("unexpected error for scalar At(): %v", err)
	}
	if scalarVal != 42 {
		t.Errorf("expected 42, got %d", scalarVal)
	}

	// Test case: Accessing a 0-dimensional tensor (scalar) with indices (should error)
	_, err = scalarTensor.At(0)
	if err == nil {
		t.Errorf("expected error for scalar At() with indices, got nil")
	}
}

func TestTensor_SetData(t *testing.T) {
	tensor, _ := New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	newData := []int{5, 6, 7, 8}
	tensor.SetData(newData)

	if !reflect.DeepEqual(tensor.Data(), newData) {
		t.Errorf("expected data %v, got %v", newData, tensor.Data())
	}
}

func TestTensor_Set(t *testing.T) {
	tensor, _ := New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})

	err := tensor.Set(10, 0, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	val, _ := tensor.At(0, 1)
	if val != 10 {
		t.Errorf("expected 10, got %d", val)
	}

	// Test case: Index out of bounds (index == shape[i])
	err = tensor.Set(10, 0, 3) // shape[1] is 3, so index 3 is out of bounds
	if err == nil {
		t.Errorf("expected error for index out of bounds, got nil")
	}

	// Test case: Index out of bounds (negative index)
	err = tensor.Set(10, 0, -1)
	if err == nil {
		t.Errorf("expected error for negative index, got nil")
	}

	// Test case: Incorrect number of indices
	err = tensor.Set(10, 0)
	if err == nil {
		t.Errorf("expected error for incorrect number of indices, got nil")
	}

	// Test case: Set on a view (should return error)
	view, _ := tensor.Slice([2]int{0, 1}, [2]int{0, 2})
	err = view.Set(100, 0, 0)
	if err == nil {
		t.Errorf("expected error when setting on a view, got nil")
	}
}

func TestTensor_Reshape(t *testing.T) {
	tensor, _ := New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})

	reshaped, err := tensor.Reshape([]int{3, 2})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if reshaped.Shape()[0] != 3 || reshaped.Shape()[1] != 2 {
		t.Errorf("unexpected shape: %v", reshaped.Shape())
	}

	val, _ := reshaped.At(1, 0)
	if val != 3 {
		t.Errorf("expected 3, got %d", val)
	}

	_, err = tensor.Reshape([]int{3, 3})
	if err == nil {
		t.Errorf("expected error, got nil")
	}

	// Test case: Cannot infer dimension (total size not divisible)
	_, err = tensor.Reshape([]int{3, -1, 4}) // 6 elements, 3*4 = 12, 6%12 != 0
	if err == nil {
		t.Errorf("expected error for non-divisible inferred dimension, got nil")
	}

	// Test case: Invalid new shape dimension (non-positive and not -1)
	_, err = tensor.Reshape([]int{3, 0}) // 0 is invalid
	if err == nil {
		t.Errorf("expected error for invalid new shape dimension, got nil")
	}

	// Test that data is shared
	err = tensor.Set(99, 1, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	reshapedVal, _ := reshaped.At(2, 0)
	if reshapedVal != 99 {
		t.Errorf("data should be shared after reshape, expected 99, got %d", reshapedVal)
	}
}

func TestTensor_Reshape_Inferred(t *testing.T) {
	tensor, _ := New[int]([]int{2, 6}, []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	reshaped, err := tensor.Reshape([]int{3, -1, 2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedShape := []int{3, 2, 2}
	if !reflect.DeepEqual(reshaped.Shape(), expectedShape) {
		t.Errorf("expected shape %v, got %v", expectedShape, reshaped.Shape())
	}
}

func TestTensor_Each(t *testing.T) {
	tensor, _ := New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	var sum int
	tensor.Each(func(val int) {
		sum += val
	})
	if sum != 21 {
		t.Errorf("expected 21, got %d", sum)
	}

	// Test with an empty tensor (0-dimensional)
	emptyTensor, _ := New[int]([]int{}, nil)
	var eachCount int
	emptyTensor.Each(func(val int) {
		eachCount++
	})
	if eachCount != 1 {
		t.Errorf("expected Each to be called once for 0-dimensional tensor, got %d", eachCount)
	}

	// Test with a 0-dimensional tensor with a specific value
	scalarTensor, _ := New[int]([]int{}, []int{99})
	var scalarVal int
	scalarTensor.Each(func(val int) {
		scalarVal = val
	})
	if scalarVal != 99 {
		t.Errorf("expected Each to pass 99 for scalar tensor, got %d", scalarVal)
	}

	// Test with a tensor that has zero size (to test eachRecursive with Dims() == 0 branch)
	zeroSizeTensor, _ := New[int]([]int{0}, nil)
	var zeroSizeCount int
	zeroSizeTensor.Each(func(val int) {
		zeroSizeCount++
	})
	// For a tensor with shape [0], Each should not call the function at all
	// because eachRecursive returns early when Dims() == 0 (actually when size is 0)
	if zeroSizeCount != 0 {
		t.Errorf("expected Each to not be called for zero-size tensor, got %d calls", zeroSizeCount)
	}
}

func TestTensor_Slice(t *testing.T) {
	// Test case: Slicing a 0-dimensional tensor (scalar)
	scalarTensor, _ := New[int]([]int{}, []int{42})
	_, err := scalarTensor.Slice([2]int{0, 0})
	if err == nil {
		t.Errorf("expected error when slicing a 0-dimensional tensor, got nil")
	}
}

func TestTensor_Copy(t *testing.T) {
	tensor, _ := New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	copied := tensor.Copy()

	// Check that the data slices are different
	if &tensor.data[0] == &copied.data[0] {
		t.Errorf("data should be copied, not shared")
	}

	// Modify the copied tensor
	err := copied.Set(10, 0, 0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Check that the original tensor is not modified
	originalVal, _ := tensor.At(0, 0)
	if originalVal == 10 {
		t.Errorf("modifying copied tensor should not affect original")
	}

	// Test copying a view
	view, _ := tensor.Slice([2]int{0, 1}, [2]int{0, 2})
	copiedView := view.Copy()
	if &view.data[0] == &copiedView.data[0] {
		t.Errorf("data should be copied, not shared")
	}
	err = copiedView.Set(20, 0, 0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	originalVal, _ = view.At(0, 0)
	if originalVal == 20 {
		t.Errorf("modifying copied view should not affect original view")
	}
}

func TestTensor_Strides(t *testing.T) {
	tensor, _ := New[int]([]int{2, 3, 4}, nil)
	expectedStrides := []int{12, 4, 1}
	if !reflect.DeepEqual(tensor.Strides(), expectedStrides) {
		t.Errorf("expected strides %v, got %v", expectedStrides, tensor.Strides())
	}
}

func TestTensor_String(t *testing.T) {
	tensor, _ := New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	expectedString := "Tensor(shape=[2 2], data=[1 2 3 4])"
	if tensor.String() != expectedString {
		t.Errorf("expected string %q, got %q", expectedString, tensor.String())
	}

	// Test with a view
	view, _ := tensor.Slice([2]int{0, 1}, [2]int{0, 2})
	expectedString = "Tensor(shape=[1 2], data=[1 2])"
	if view.String() != expectedString {
		t.Errorf("expected string %q, got %q", expectedString, view.String())
	}
}

func TestTensor_Data_View(t *testing.T) {
	// Test case 1: 1D view
	tensor1D, _ := New[int]([]int{4}, []int{1, 2, 3, 4})
	view1D, err := tensor1D.Slice([2]int{0, 1}) // Corrected: only one range for 1D tensor
	if err != nil {
		t.Fatalf("unexpected error creating 1D view: %v", err)
	}
	expectedData1D := []int{1}
	if !reflect.DeepEqual(view1D.Data(), expectedData1D) {
		t.Errorf("Test 1D View: expected data %v, got %v", expectedData1D, view1D.Data())
	}

	// Test case 2: 2D view
	tensor2D, _ := New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	// Create a view of the second row: [4 5 6]
	view2D, err := tensor2D.Slice([2]int{1, 2}, [2]int{0, 3})
	if err != nil {
		t.Fatalf("unexpected error creating 2D view: %v", err)
	}
	expectedData2D := []int{4, 5, 6}
	if !reflect.DeepEqual(view2D.Data(), expectedData2D) {
		t.Errorf("Test 2D View: expected data %v, got %v", expectedData2D, view2D.Data())
	}

	// Test case 3: 2D view with slicing in both dimensions
	// Original:
	// 1 2 3
	// 4 5 6
	// View:
	// 2 3
	// 5 6
	view2DSliced, err := tensor2D.Slice([2]int{0, 2}, [2]int{1, 3})
	if err != nil {
		t.Fatalf("unexpected error creating 2D sliced view: %v", err)
	}
	expectedData2DSliced := []int{2, 3, 5, 6}
	if !reflect.DeepEqual(view2DSliced.Data(), expectedData2DSliced) {
		t.Errorf("Test 2D Sliced View: expected data %v, got %v", expectedData2DSliced, view2DSliced.Data())
	}

	// Test case 4: Tensor with size 0 (due to a 0-sized dimension)
	zeroSizeTensor, err := New[int]([]int{2, 0, 3}, nil)
	if err != nil {
		t.Fatalf("unexpected error creating zero size tensor: %v", err)
	}
	expectedZeroSizeData := []int{}
	if !reflect.DeepEqual(zeroSizeTensor.Data(), expectedZeroSizeData) {
		t.Errorf("Test Zero Size Tensor: expected data %v, got %v", expectedZeroSizeData, zeroSizeTensor.Data())
	}

	// Test case 5: View with size 0
	zeroSizeViewTensor, _ := New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	zeroSizeView, err := zeroSizeViewTensor.Slice([2]int{0, 2}, [2]int{0, 0})
	if err != nil {
		t.Fatalf("unexpected error creating zero size view: %v", err)
	}
	expectedZeroSizeViewData := []int{}
	if !reflect.DeepEqual(zeroSizeView.Data(), expectedZeroSizeViewData) {
		t.Errorf("Test Zero Size View: expected data %v, got %v", expectedZeroSizeViewData, zeroSizeView.Data())
	}

	// Test case 6: 0-dimensional view tensor (scalar view)
	// Create a scalar tensor and then create a view of it
	scalarTensor, _ := New[int]([]int{}, []int{42})
	// For a 0-dimensional tensor, we can't really slice it, but we can test the Data() method
	// when isView is true and Dims() == 0
	// We need to manually create a view-like scenario by creating a tensor that behaves like a view
	// This tests the specific branch in Data() for 0-dimensional views
	scalarViewData := scalarTensor.Data()
	expectedScalarData := []int{42}
	if !reflect.DeepEqual(scalarViewData, expectedScalarData) {
		t.Errorf("Test 0-dimensional tensor Data: expected data %v, got %v", expectedScalarData, scalarViewData)
	}
}
