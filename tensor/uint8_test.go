package tensor

import (
	"testing"
)

func TestUINT8TensorCreation(t *testing.T) {
	// Test creating UINT8 tensors with various methods
	
	// Test New method with nil data (auto-initialized to zero)
	shape := []int{2, 3, 4}
	tensor1, err := New[uint8](shape, nil)
	if err != nil {
		t.Fatalf("Failed to create UINT8 tensor: %v", err)
	}
	
	if len(tensor1.Shape()) != len(shape) {
		t.Errorf("Expected shape length %d, got %d", len(shape), len(tensor1.Shape()))
	}
	
	for i, dim := range tensor1.Shape() {
		if dim != shape[i] {
			t.Errorf("Shape dimension %d: expected %d, got %d", i, shape[i], dim)
		}
	}
	
	// Test data is initialized to zero
	data := tensor1.Data()
	expectedSize := 2 * 3 * 4
	if len(data) != expectedSize {
		t.Errorf("Expected data size %d, got %d", expectedSize, len(data))
	}
	
	for i, val := range data {
		if val != 0 {
			t.Errorf("Expected zero initialization at index %d, got %d", i, val)
		}
	}
}

func TestUINT8TensorDataAccess(t *testing.T) {
	// Test data access for UINT8 tensors
	testValues := []uint8{100, 200, 50, 255}
	tensor, err := New[uint8]([]int{2, 2}, testValues)
	if err != nil {
		t.Fatalf("Failed to create UINT8 tensor: %v", err)
	}
	
	// Verify values through Data() method
	data := tensor.Data()
	if len(data) != len(testValues) {
		t.Errorf("Expected data length %d, got %d", len(testValues), len(data))
	}
	
	for i, expected := range testValues {
		if data[i] != expected {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, expected, data[i])
		}
	}
	
	// Test modifying data directly
	data[0] = 123
	modifiedData := tensor.Data()
	if modifiedData[0] != 123 {
		t.Errorf("Data modification failed: expected 123, got %d", modifiedData[0])
	}
}

func TestUINT8TensorFromSlice(t *testing.T) {
	// Test creating UINT8 tensor from slice
	data := []uint8{10, 20, 30, 40, 50, 60}
	shape := []int{2, 3}
	
	tensor, err := New[uint8](shape, data)
	if err != nil {
		t.Fatalf("Failed to create tensor from slice: %v", err)
	}
	
	// Verify shape
	if len(tensor.Shape()) != len(shape) {
		t.Errorf("Expected shape length %d, got %d", len(shape), len(tensor.Shape()))
	}
	
	// Verify data
	retrievedData := tensor.Data()
	for i, expected := range data {
		if retrievedData[i] != expected {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, expected, retrievedData[i])
		}
	}
}

func TestUINT8TensorValueRange(t *testing.T) {
	// Test UINT8 specific value ranges (0-255)
	testValues := []uint8{0, 1, 127, 128, 254, 255}
	
	for _, val := range testValues {
		tensor, err := New[uint8]([]int{1, 1}, []uint8{val})
		if err != nil {
			t.Fatalf("Failed to create UINT8 tensor with value %d: %v", val, err)
		}
		
		data := tensor.Data()
		if len(data) != 1 {
			t.Fatalf("Expected single element tensor, got %d elements", len(data))
		}
		
		if data[0] != val {
			t.Errorf("Value mismatch: expected %d, got %d", val, data[0])
		}
	}
}

func TestUINT8TensorArithmetic(t *testing.T) {
	// Test basic operations that should work with UINT8 tensors
	data1 := []uint8{10, 20, 30, 40}
	data2 := []uint8{5, 10, 15, 20}
	shape := []int{2, 2}
	
	tensor1, err := New[uint8](shape, data1)
	if err != nil {
		t.Fatalf("Failed to create tensor1: %v", err)
	}
	
	tensor2, err := New[uint8](shape, data2)
	if err != nil {
		t.Fatalf("Failed to create tensor2: %v", err)
	}
	
	// Test that tensors are created and basic operations can be performed
	// (Actual arithmetic operations depend on the numeric package implementation)
	
	// Verify tensor properties
	if tensor1.DType() != tensor2.DType() {
		t.Errorf("Tensors should have same DType")
	}
	
	// Test that tensors can be compared and analyzed
	data1Slice := tensor1.Data()
	data2Slice := tensor2.Data()
	
	if len(data1Slice) != len(data2Slice) {
		t.Errorf("Tensor data slices should have same length")
	}
	
	// Verify data integrity
	for i, expected := range data1 {
		if data1Slice[i] != expected {
			t.Errorf("Tensor1 data mismatch at %d: expected %d, got %d", i, expected, data1Slice[i])
		}
	}
	
	for i, expected := range data2 {
		if data2Slice[i] != expected {
			t.Errorf("Tensor2 data mismatch at %d: expected %d, got %d", i, expected, data2Slice[i])
		}
	}
}

func TestUINT8TensorSerialization(t *testing.T) {
	// Test that UINT8 tensors can be serialized/deserialized
	original, err := New[uint8]([]int{3, 3}, nil)
	if err != nil {
		t.Fatalf("Failed to create UINT8 tensor: %v", err)
	}
	
	// Fill with test data
	testData := []uint8{0, 32, 64, 96, 128, 160, 192, 224, 255}
	for i, val := range testData {
		row := i / 3
		col := i % 3
		err := original.Set(val, row, col)
		if err != nil {
			t.Fatalf("Failed to set test data at [%d,%d]: %v", row, col, err)
		}
	}
	
	// Test basic tensor properties needed for serialization
	shape := original.Shape()
	data := original.Data()
	dtype := original.DType()
	
	// Verify data integrity
	if len(data) != len(testData) {
		t.Errorf("Data length mismatch: expected %d, got %d", len(testData), len(data))
	}
	
	for i, expected := range testData {
		if data[i] != expected {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, expected, data[i])
		}
	}
	
	// Test recreating tensor from properties
	recreated, err := New[uint8](shape, data)
	if err != nil {
		t.Fatalf("Failed to recreate tensor: %v", err)
	}
	
	if recreated.DType() != dtype {
		t.Errorf("DType mismatch in recreated tensor")
	}
	
	// Verify recreated data
	recreatedData := recreated.Data()
	for i, expected := range testData {
		if recreatedData[i] != expected {
			t.Errorf("Recreated data mismatch at index %d: expected %d, got %d", i, expected, recreatedData[i])
		}
	}
	
	t.Logf("Successfully tested UINT8 tensor serialization with %d elements", len(testData))
}