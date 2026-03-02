package compute

import (
	"bytes"
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/log"
	metrics "github.com/zerfoo/zerfoo/metrics/runtime"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestCPUEngine_UnaryOp(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	op := func(v int) int { return v * 2 }
	result, _ := engine.UnaryOp(context.Background(), a, op, nil)

	expected := []int{2, 4, 6, 8}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Add(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	b, _ := tensor.New[int]([]int{2, 2}, []int{5, 6, 7, 8})
	result, _ := engine.Add(context.Background(), a, b, nil)

	expected := []int{6, 8, 10, 12}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Sub(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 2}, []int{5, 6, 7, 8})
	b, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	result, _ := engine.Sub(context.Background(), a, b, nil)

	expected := []int{4, 4, 4, 4}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Mul(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	b, _ := tensor.New[int]([]int{2, 2}, []int{5, 6, 7, 8})
	result, _ := engine.Mul(context.Background(), a, b, nil)

	expected := []int{5, 12, 21, 32}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Div(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 2}, []int{10, 12, 14, 16})
	b, _ := tensor.New[int]([]int{2, 2}, []int{2, 3, 2, 4})
	result, _ := engine.Div(context.Background(), a, b, nil)

	expected := []int{5, 4, 7, 4}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_MatMul(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[int]([]int{3, 2}, []int{7, 8, 9, 10, 11, 12})
	result, _ := engine.MatMul(context.Background(), a, b, nil)

	expected := []int{58, 64, 139, 154}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Transpose(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	result, _ := engine.Transpose(context.Background(), a, []int{1, 0}, nil)
	expectedData := []int{1, 4, 2, 5, 3, 6}
	expectedShape := []int{3, 2}

	if !reflect.DeepEqual(result.Data(), expectedData) {
		t.Errorf("expected data %v, got %v", expectedData, result.Data())
	}

	if !reflect.DeepEqual(result.Shape(), expectedShape) {
		t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
	}

	// Test 3D transpose
	b, _ := tensor.New[int]([]int{2, 2, 3}, []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	result, _ = engine.Transpose(context.Background(), b, []int{0, 2, 1}, nil)
	expectedData3D := []int{1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12}
	expectedShape3D := []int{2, 3, 2}

	if !reflect.DeepEqual(result.Data(), expectedData3D) {
		t.Errorf("expected data %v, got %v", expectedData3D, result.Data())
	}

	if !reflect.DeepEqual(result.Shape(), expectedShape3D) {
		t.Errorf("expected shape %v, got %v", expectedShape3D, result.Shape())
	}
}

func TestCPUEngine_Sum(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	result, _ := engine.Sum(context.Background(), a, 0, false, nil)

	expected := []int{5, 7, 9}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test sum over axis 1
	result, _ = engine.Sum(context.Background(), a, 1, false, nil)

	expected = []int{6, 15}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test sum with 3D tensor
	b, _ := tensor.New[int]([]int{2, 2, 2}, []int{1, 2, 3, 4, 5, 6, 7, 8})
	result, _ = engine.Sum(context.Background(), b, 0, false, nil)

	expected = []int{6, 8, 10, 12}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test sum with 1D tensor to scalar
	c, _ := tensor.New[int]([]int{4}, []int{1, 2, 3, 4})
	result, _ = engine.Sum(context.Background(), c, -1, false, nil)

	expected = []int{10}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Exp(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	result, _ := engine.Exp(context.Background(), a, nil)

	expected := []float32{2.7182817, 7.389056, 20.085537, 54.59815}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Log(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	result, _ := engine.Log(context.Background(), a, nil)

	expected := []float32{0, 0.6931472, 1.0986123, 1.3862944}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Pow(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	b, _ := tensor.New[float32]([]int{2, 2}, []float32{2, 3, 2, 4})
	result, _ := engine.Pow(context.Background(), a, b, nil)

	expected := []float32{1, 8, 9, 256}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_Dst(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	b, _ := tensor.New[int]([]int{2, 2}, []int{5, 6, 7, 8})
	dst, _ := tensor.New[int]([]int{2, 2}, nil)

	result, _ := engine.Add(context.Background(), a, b, dst)
	if result != dst {
		t.Error("expected result to be dst")
	}
}

func TestCPUEngine_Errors(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	b, _ := tensor.New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	ctx := context.Background()

	// UnaryOp
	_, err := engine.UnaryOp(ctx, nil, func(v int) int { return v }, nil)
	if err == nil {
		t.Error("expected error for nil input to UnaryOp")
	}

	// Add
	_, err = engine.Add(ctx, nil, a, nil)
	if err == nil {
		t.Error("expected error for nil input to Add")
	}

	_, err = engine.Add(ctx, a, nil, nil)
	if err == nil {
		t.Error("expected error for nil input to Add")
	}

	_, err = engine.Add(ctx, a, b, nil)
	if err == nil {
		t.Error("expected error for mismatched shapes in Add")
	}

	// Sub
	_, err = engine.Sub(ctx, nil, a, nil)
	if err == nil {
		t.Error("expected error for nil input to Sub")
	}

	_, err = engine.Sub(ctx, a, nil, nil)
	if err == nil {
		t.Error("expected error for nil input to Sub")
	}

	_, err = engine.Sub(ctx, a, b, nil)
	if err == nil {
		t.Error("expected error for mismatched shapes in Sub")
	}

	// Mul
	_, err = engine.Mul(ctx, nil, a, nil)
	if err == nil {
		t.Error("expected error for nil input to Mul")
	}

	_, err = engine.Mul(ctx, a, nil, nil)
	if err == nil {
		t.Error("expected error for nil input to Mul")
	}

	_, err = engine.Mul(ctx, a, b, nil)
	if err == nil {
		t.Error("expected error for mismatched shapes in Mul")
	}

	// Div
	_, err = engine.Div(ctx, nil, a, nil)
	if err == nil {
		t.Error("expected error for nil input to Div")
	}

	_, err = engine.Div(ctx, a, nil, nil)
	if err == nil {
		t.Error("expected error for nil input to Div")
	}

	_, err = engine.Div(ctx, a, b, nil)
	if err == nil {
		t.Error("expected error for mismatched shapes in Div")
	}

	c, _ := tensor.New[int]([]int{2, 2}, []int{1, 0, 3, 4})

	_, err = engine.Div(ctx, a, c, nil)
	if err == nil {
		t.Error("expected error for division by zero in Div")
	}

	// MatMul
	_, err = engine.MatMul(ctx, nil, a, nil)
	if err == nil {
		t.Error("expected error for nil input to MatMul")
	}

	_, err = engine.MatMul(ctx, a, nil, nil)
	if err == nil {
		t.Error("expected error for nil input to MatMul")
	}

	e, _ := tensor.New[int]([]int{3, 2}, nil)

	_, err = engine.MatMul(ctx, a, e, nil)
	if err == nil {
		t.Error("expected error for mismatched shapes in MatMul")
	}

	// Transpose
	_, err = engine.Transpose(ctx, nil, nil, nil)
	if err == nil {
		t.Error("expected error for nil input to Transpose")
	}

	d, _ := tensor.New[int]([]int{2, 2, 2}, nil)

	_, err = engine.Transpose(ctx, d, []int{0, 1}, nil)
	if err == nil {
		t.Error("expected error for incorrect number of axes in Transpose")
	}

	// Sum
	_, err = engine.Sum(ctx, nil, 0, false, nil)
	if err == nil {
		t.Error("expected error for nil input to Sum")
	}

	_, err = engine.Sum(ctx, a, 2, false, nil)
	if err == nil {
		t.Error("expected error for invalid axis in Sum")
	}

	// Dst shape error
	dst, _ := tensor.New[int]([]int{1, 1}, nil)

	_, err = engine.Add(ctx, a, a, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}
}

func TestCPUEngine_Zero(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})

	a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	if err := engine.Zero(context.Background(), a); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expected := []int{0, 0, 0, 0}
	if !reflect.DeepEqual(a.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, a.Data())
	}
}

func TestCPUEngine_Copy(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	src, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})

	dst, _ := tensor.New[int]([]int{2, 2}, nil)
	if err := engine.Copy(context.Background(), dst, src); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(dst.Data(), src.Data()) {
		t.Errorf("expected %v, got %v", src.Data(), dst.Data())
	}
}

func TestCPUEngine_DstErrors(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	dst, _ := tensor.New[int]([]int{1, 1}, nil)
	ctx := context.Background()

	_, err := engine.UnaryOp(ctx, a, func(v int) int { return v }, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.Sub(ctx, a, a, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.Mul(ctx, a, a, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.Div(ctx, a, a, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.MatMul(ctx, a, a, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.Transpose(ctx, a, []int{1, 0}, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.Sum(ctx, a, 0, false, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.Exp(ctx, a, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.Log(ctx, a, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	_, err = engine.Pow(ctx, a, a, dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}
}

func TestCPUEngine_Add_Int8(t *testing.T) {
	engine := NewCPUEngine[int8](numeric.Int8Ops{})
	a, _ := tensor.New[int8]([]int{2, 2}, []int8{1, 2, 3, 4})
	b, _ := tensor.New[int8]([]int{2, 2}, []int8{5, 6, 7, 8})
	result, _ := engine.Add(context.Background(), a, b, nil)

	expected := []int8{6, 8, 10, 12}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_MatMul_Int8(t *testing.T) {
	engine := NewCPUEngine[int8](numeric.Int8Ops{})
	a, _ := tensor.New[int8]([]int{2, 3}, []int8{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[int8]([]int{3, 2}, []int8{7, 8, 9, 10, 11, 12})
	result, _ := engine.MatMul(context.Background(), a, b, nil)

	expected := []int8{58, 64, -117, -102}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}
}

func TestCPUEngine_SetLogger(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})

	var buf bytes.Buffer
	l := log.New(&buf, log.LevelDebug, log.FormatText)
	engine.SetLogger(l)

	// Verify logger is active by using it.
	engine.logger.Info("test message", "key", "value")
	out := buf.String()
	if !strings.Contains(out, "test message") {
		t.Errorf("expected logger output, got %q", out)
	}
}

func TestCPUEngine_SetLogger_Nil(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	engine.SetLogger(nil) // Should not panic; defaults to Nop.

	// Verify engine still works after nil logger.
	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	b, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	_, err := engine.Add(context.Background(), a, b)
	if err != nil {
		t.Fatalf("Add after nil logger: %v", err)
	}
}

func TestCPUEngine_SetCollector(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	collector := metrics.NewInMemory()
	engine.SetCollector(collector)

	ctx := context.Background()
	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})

	// Run several operations
	_, _ = engine.Add(ctx, a, b)
	_, _ = engine.MatMul(ctx, a, b)
	_, _ = engine.Softmax(ctx, a, -1)

	snap := collector.Snapshot()

	tests := []struct {
		name string
		want int64
	}{
		{"op_count_Add", 1},
		{"op_count_MatMul", 1},
		{"op_count_Softmax", 1},
	}

	for _, tt := range tests {
		got := snap.Counters[tt.name]
		if got != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, got, tt.want)
		}
	}

	// Verify histogram has observations
	h, ok := snap.Histograms["op_duration_seconds"]
	if !ok {
		t.Fatal("expected op_duration_seconds histogram")
	}
	if h.Count != 3 {
		t.Errorf("histogram count = %d, want 3", h.Count)
	}
}

func TestCPUEngine_SetCollector_Nil(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	engine.SetCollector(nil) // Should not panic; defaults to Nop.

	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	b, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	_, err := engine.Add(context.Background(), a, b)
	if err != nil {
		t.Fatalf("Add after nil collector: %v", err)
	}
}

func TestCPUEngine_Close(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	err := engine.Close(context.Background())
	if err != nil {
		t.Fatalf("Close: %v", err)
	}
	// Verify engine is still usable after Close (no-op).
	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	b, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	_, err = engine.Add(context.Background(), a, b)
	if err != nil {
		t.Fatalf("Add after Close: %v", err)
	}
}

func TestCPUEngine_MetricsPerOp(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	collector := metrics.NewInMemory()
	engine.SetCollector(collector)

	ctx := context.Background()
	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})

	ops := []struct {
		name string
		fn   func()
	}{
		{"Add", func() { _, _ = engine.Add(ctx, a, b) }},
		{"Sub", func() { _, _ = engine.Sub(ctx, a, b) }},
		{"Mul", func() { _, _ = engine.Mul(ctx, a, b) }},
		{"Div", func() { _, _ = engine.Div(ctx, a, b) }},
		{"MatMul", func() { _, _ = engine.MatMul(ctx, a, b) }},
		{"Tanh", func() { _, _ = engine.Tanh(ctx, a) }},
		{"Exp", func() { _, _ = engine.Exp(ctx, a) }},
		{"Log", func() { _, _ = engine.Log(ctx, a) }},
		{"Pow", func() { _, _ = engine.Pow(ctx, a, b) }},
		{"Sum", func() { _, _ = engine.Sum(ctx, a, 0, false) }},
		{"ReduceSum", func() { _, _ = engine.ReduceSum(ctx, a, 0, false) }},
		{"ReduceMean", func() { _, _ = engine.ReduceMean(ctx, a, 0, false) }},
		{"Softmax", func() { _, _ = engine.Softmax(ctx, a, -1) }},
		{"Transpose", func() { _, _ = engine.Transpose(ctx, a, nil) }},
	}

	for _, op := range ops {
		op.fn()
	}

	snap := collector.Snapshot()
	for _, op := range ops {
		got := snap.Counters["op_count_"+op.name]
		if got < 1 {
			t.Errorf("op_count_%s = %d, want >= 1", op.name, got)
		}
	}
}

func TestCPUEngine_SetMemoryLimit(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	engine.SetMemoryLimit(1024)

	mt := engine.MemoryTracker()
	if mt.Limit() != 1024 {
		t.Errorf("limit = %d, want 1024", mt.Limit())
	}
}

func TestCPUEngine_MemoryLimit_AllocationSucceeds(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	// 10 float32 elements = 40 bytes. Set limit to 1000 bytes.
	engine.SetMemoryLimit(1000)

	ctx := context.Background()
	a, err := tensor.New[float32]([]int{5}, []float32{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.New[float32]([]int{5}, []float32{6, 7, 8, 9, 10})
	if err != nil {
		t.Fatal(err)
	}

	_, err = engine.Add(ctx, a, b)
	if err != nil {
		t.Fatalf("Add within limit: %v", err)
	}
}

func TestCPUEngine_MemoryLimit_AllocationFails(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	// 4 bytes per float32 * 100 = 400 bytes, set limit to 100 bytes.
	engine.SetMemoryLimit(100)

	ctx := context.Background()
	a, err := tensor.New[float32]([]int{100}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.New[float32]([]int{100}, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = engine.Add(ctx, a, b)
	if err == nil {
		t.Fatal("expected memory limit error, got nil")
	}
	if !strings.Contains(err.Error(), "memory limit exceeded") {
		t.Errorf("expected memory limit error, got: %v", err)
	}
}

func TestCPUEngine_TimeoutCanceled(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	a, err := tensor.New[float32]([]int{5}, []float32{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.New[float32]([]int{5}, []float32{6, 7, 8, 9, 10})
	if err != nil {
		t.Fatal(err)
	}

	// Create an already-canceled context.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = engine.Add(ctx, a, b)
	if err == nil {
		t.Fatal("expected context error, got nil")
	}
	if !strings.Contains(err.Error(), "canceled") {
		t.Errorf("expected canceled error, got: %v", err)
	}
}

func TestCPUEngine_MatMulTimeout(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	a, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})
	if err != nil {
		t.Fatal(err)
	}

	// Already canceled context.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = engine.MatMul(ctx, a, b)
	if err == nil {
		t.Fatal("expected context error, got nil")
	}
	if !strings.Contains(err.Error(), "canceled") {
		t.Errorf("expected canceled error, got: %v", err)
	}
}

func TestCPUEngine_UnaryOpTimeout(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	a, err := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = engine.UnaryOp(ctx, a, func(v float32) float32 { return v * 2 })
	if err == nil {
		t.Fatal("expected context error, got nil")
	}
}

func TestCPUEngine_MemoryLimit_Unlimited(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	// Default is unlimited (0).
	mt := engine.MemoryTracker()
	if mt.Limit() != 0 {
		t.Errorf("default limit = %d, want 0 (unlimited)", mt.Limit())
	}

	ctx := context.Background()
	a, err := tensor.New[float32]([]int{1000}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.New[float32]([]int{1000}, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = engine.Add(ctx, a, b)
	if err != nil {
		t.Fatalf("Add with unlimited: %v", err)
	}
}
