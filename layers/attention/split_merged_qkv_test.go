package attention

import (
	"bytes"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestSplitMergedQKV_CPUPath(t *testing.T) {
	// Verify the CPU fallback produces correct Q, K, V splits.
	qDim, kDim, vDim := 4, 2, 2
	totalDim := qDim + kDim + vDim

	// Create a [1, 8] tensor with known values.
	data := make([]float32, totalDim)
	for i := range data {
		data[i] = float32(i)
	}
	merged, err := tensor.New([]int{1, totalDim}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	q, k, v, err := splitMergedQKV[float32](merged, qDim, kDim, vDim)
	if err != nil {
		t.Fatalf("splitMergedQKV: %v", err)
	}

	// Verify shapes.
	if got, want := q.Shape(), []int{1, qDim}; !shapeEqual(got, want) {
		t.Errorf("q shape = %v, want %v", got, want)
	}
	if got, want := k.Shape(), []int{1, kDim}; !shapeEqual(got, want) {
		t.Errorf("k shape = %v, want %v", got, want)
	}
	if got, want := v.Shape(), []int{1, vDim}; !shapeEqual(got, want) {
		t.Errorf("v shape = %v, want %v", got, want)
	}

	// Verify data: Q=[0,1,2,3], K=[4,5], V=[6,7].
	for i, want := range []float32{0, 1, 2, 3} {
		if got := q.Data()[i]; got != want {
			t.Errorf("q[%d] = %v, want %v", i, got, want)
		}
	}
	for i, want := range []float32{4, 5} {
		if got := k.Data()[i]; got != want {
			t.Errorf("k[%d] = %v, want %v", i, got, want)
		}
	}
	for i, want := range []float32{6, 7} {
		if got := v.Data()[i]; got != want {
			t.Errorf("v[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestSplitMergedQKV_CPUPath_LogsWarning(t *testing.T) {
	// Verify the CPU fallback logs a WARNING about D2H copy.
	qDim, kDim, vDim := 4, 2, 2
	totalDim := qDim + kDim + vDim

	data := make([]float32, totalDim)
	merged, err := tensor.New([]int{1, totalDim}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)

	_, _, _, err = splitMergedQKV[float32](merged, qDim, kDim, vDim)
	if err != nil {
		t.Fatalf("splitMergedQKV: %v", err)
	}

	output := buf.String()
	if !bytes.Contains(buf.Bytes(), []byte("WARN")) && !bytes.Contains(buf.Bytes(), []byte("WARNING")) && !strings.Contains(output, "CPU fallback") {
		t.Errorf("expected WARN log about CPU fallback, got: %q", output)
	}
}

func TestSplitMergedQKV_Float16Path(t *testing.T) {
	// Verify the Float16Storage path produces correct Q, K, V splits
	// without triggering the CPU fallback WARNING.
	qDim, kDim, vDim := 4, 2, 2
	totalDim := qDim + kDim + vDim

	srcData := make([]float32, totalDim)
	for i := range srcData {
		srcData[i] = float32(i)
	}

	fp16s := tensor.NewFloat16StorageFromF32(srcData)
	merged, err := tensor.NewWithStorage[float32]([]int{1, totalDim}, fp16s)
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}

	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)

	q, k, v, err := splitMergedQKV[float32](merged, qDim, kDim, vDim)
	if err != nil {
		t.Fatalf("splitMergedQKV: %v", err)
	}

	// No WARNING should have been logged — FP16 path is a GPU fast path.
	if bytes.Contains(buf.Bytes(), []byte("WARNING")) {
		t.Errorf("Float16Storage should not trigger CPU fallback, got: %q", buf.String())
	}

	// Verify shapes.
	if got, want := q.Shape(), []int{1, qDim}; !shapeEqual(got, want) {
		t.Errorf("q shape = %v, want %v", got, want)
	}
	if got, want := k.Shape(), []int{1, kDim}; !shapeEqual(got, want) {
		t.Errorf("k shape = %v, want %v", got, want)
	}
	if got, want := v.Shape(), []int{1, vDim}; !shapeEqual(got, want) {
		t.Errorf("v shape = %v, want %v", got, want)
	}

	// Verify storage type: all results should be Float16Storage (zero-copy views).
	if _, ok := any(q.GetStorage()).(*tensor.Float16Storage); !ok {
		t.Errorf("q storage type = %T, want *tensor.Float16Storage", q.GetStorage())
	}
	if _, ok := any(k.GetStorage()).(*tensor.Float16Storage); !ok {
		t.Errorf("k storage type = %T, want *tensor.Float16Storage", k.GetStorage())
	}
	if _, ok := any(v.GetStorage()).(*tensor.Float16Storage); !ok {
		t.Errorf("v storage type = %T, want *tensor.Float16Storage", v.GetStorage())
	}

	// Verify data values via FP16 decode.
	qData := q.Data()
	for i, want := range []float32{0, 1, 2, 3} {
		if got := qData[i]; got != want {
			t.Errorf("q[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestSplitMergedQKV_Batched(t *testing.T) {
	// Verify correct splitting with batch dimension [2, 8].
	qDim, kDim, vDim := 4, 2, 2
	totalDim := qDim + kDim + vDim
	batchSize := 2

	data := make([]float32, batchSize*totalDim)
	for i := range data {
		data[i] = float32(i)
	}
	merged, err := tensor.New([]int{batchSize, totalDim}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	q, k, v, err := splitMergedQKV[float32](merged, qDim, kDim, vDim)
	if err != nil {
		t.Fatalf("splitMergedQKV: %v", err)
	}

	if got, want := q.Shape(), []int{batchSize, qDim}; !shapeEqual(got, want) {
		t.Errorf("q shape = %v, want %v", got, want)
	}
	if got, want := k.Shape(), []int{batchSize, kDim}; !shapeEqual(got, want) {
		t.Errorf("k shape = %v, want %v", got, want)
	}
	if got, want := v.Shape(), []int{batchSize, vDim}; !shapeEqual(got, want) {
		t.Errorf("v shape = %v, want %v", got, want)
	}

	// Batch 0: Q=[0,1,2,3] K=[4,5] V=[6,7]
	// Batch 1: Q=[8,9,10,11] K=[12,13] V=[14,15]
	qData := q.Data()
	wantQ := []float32{0, 1, 2, 3, 8, 9, 10, 11}
	for i, want := range wantQ {
		if got := qData[i]; got != want {
			t.Errorf("q[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestSplitMergedQKV_ValidationErrors(t *testing.T) {
	tests := []struct {
		name             string
		shape            []int
		qDim, kDim, vDim int
		wantErr          string
	}{
		{
			name:    "1D tensor",
			shape:   []int{8},
			qDim:    4,
			kDim:    2,
			vDim:    2,
			wantErr: "expected at least 2D",
		},
		{
			name:    "dimension mismatch",
			shape:   []int{1, 10},
			qDim:    4,
			kDim:    2,
			vDim:    2,
			wantErr: "last dim 10 != qDim(4)+kDim(2)+vDim(2)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			size := 1
			for _, d := range tt.shape {
				size *= d
			}
			merged, err := tensor.New[float32](tt.shape, make([]float32, size))
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			_, _, _, err = splitMergedQKV[float32](merged, tt.qDim, tt.kDim, tt.vDim)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if got := err.Error(); !containsStr(got, tt.wantErr) {
				t.Errorf("error = %q, want substring %q", got, tt.wantErr)
			}
		})
	}
}

func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func containsStr(s, sub string) bool {
	return len(s) >= len(sub) && bytes.Contains([]byte(s), []byte(sub))
}
