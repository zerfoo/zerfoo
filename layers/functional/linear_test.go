package functional

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestLinear(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// x: [2, 3], weight: [4, 3], bias: [4]
	// y = x @ weight^T + bias  =>  [2, 4]
	x, err := tensor.New[float32]([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	if err != nil {
		t.Fatal(err)
	}

	weight, err := tensor.New[float32]([]int{4, 3}, []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1,
	})
	if err != nil {
		t.Fatal(err)
	}

	bias, err := tensor.New[float32]([]int{4}, []float32{10, 20, 30, 40})
	if err != nil {
		t.Fatal(err)
	}

	y, err := Linear(ctx, engine, x, weight, bias)
	if err != nil {
		t.Fatal(err)
	}

	shape := y.Shape()
	if shape[0] != 2 || shape[1] != 4 {
		t.Fatalf("expected shape [2, 4], got %v", shape)
	}

	// x @ weight^T:
	//   [1,2,3] @ [[1,0,0,1],[0,1,0,1],[0,0,1,1]]^T = [1, 2, 3, 6]
	//   [4,5,6] @ ... = [4, 5, 6, 15]
	// + bias [10,20,30,40]:
	//   [11, 22, 33, 46]
	//   [14, 25, 36, 55]
	expected := []float32{11, 22, 33, 46, 14, 25, 36, 55}
	data := y.Data()
	for i, v := range expected {
		if math.Abs(float64(data[i]-v)) > 1e-5 {
			t.Errorf("y[%d] = %f, want %f", i, data[i], v)
		}
	}
}

func TestLinearNoBias(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	x, err := tensor.New[float32]([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	if err != nil {
		t.Fatal(err)
	}

	weight, err := tensor.New[float32]([]int{4, 3}, []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1,
	})
	if err != nil {
		t.Fatal(err)
	}

	y, err := Linear(ctx, engine, x, weight, nil)
	if err != nil {
		t.Fatal(err)
	}

	shape := y.Shape()
	if shape[0] != 2 || shape[1] != 4 {
		t.Fatalf("expected shape [2, 4], got %v", shape)
	}

	expected := []float32{1, 2, 3, 6, 4, 5, 6, 15}
	data := y.Data()
	for i, v := range expected {
		if math.Abs(float64(data[i]-v)) > 1e-5 {
			t.Errorf("y[%d] = %f, want %f", i, data[i], v)
		}
	}
}

func TestMultiHeadAttention(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	seqLen, dModel, nHeads := 4, 8, 2

	// Create q, k, v with known data
	qData := make([]float32, seqLen*dModel)
	kData := make([]float32, seqLen*dModel)
	vData := make([]float32, seqLen*dModel)
	for i := range qData {
		qData[i] = float32(i) * 0.01
		kData[i] = float32(i) * 0.01
		vData[i] = float32(i) * 0.1
	}

	q, err := tensor.New[float32]([]int{seqLen, dModel}, qData)
	if err != nil {
		t.Fatal(err)
	}
	k, err := tensor.New[float32]([]int{seqLen, dModel}, kData)
	if err != nil {
		t.Fatal(err)
	}
	v, err := tensor.New[float32]([]int{seqLen, dModel}, vData)
	if err != nil {
		t.Fatal(err)
	}

	out, err := MultiHeadAttention(ctx, engine, q, k, v, nHeads)
	if err != nil {
		t.Fatal(err)
	}

	outShape := out.Shape()
	if outShape[0] != seqLen || outShape[1] != dModel {
		t.Fatalf("expected shape [%d, %d], got %v", seqLen, dModel, outShape)
	}

	// Verify output is finite and non-zero
	data := out.Data()
	allZero := true
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("output contains NaN or Inf")
		}
		if v != 0 {
			allZero = false
		}
	}
	if allZero {
		t.Fatal("output is all zeros")
	}
}

func TestMultiHeadAttention_InvalidInputs(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	t.Run("non_2D_query", func(t *testing.T) {
		q, _ := tensor.New[float32]([]int{2, 3, 4}, make([]float32, 24))
		k, _ := tensor.New[float32]([]int{2, 3, 4}, make([]float32, 24))
		v, _ := tensor.New[float32]([]int{2, 3, 4}, make([]float32, 24))
		_, err := MultiHeadAttention(ctx, engine, q, k, v, 2)
		if err == nil {
			t.Fatal("expected error for non-2D query")
		}
	})

	t.Run("d_model_not_divisible_by_nHeads", func(t *testing.T) {
		q, _ := tensor.New[float32]([]int{3, 5}, make([]float32, 15))
		k, _ := tensor.New[float32]([]int{3, 5}, make([]float32, 15))
		v, _ := tensor.New[float32]([]int{3, 5}, make([]float32, 15))
		_, err := MultiHeadAttention(ctx, engine, q, k, v, 3)
		if err == nil {
			t.Fatal("expected error when d_model not divisible by nHeads")
		}
	})
}

func TestMultiHeadAttention_SingleHead(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	seqLen, dModel := 3, 4

	// Use identity-like values so we can verify the standard attention formula
	qData := make([]float32, seqLen*dModel)
	kData := make([]float32, seqLen*dModel)
	vData := make([]float32, seqLen*dModel)
	for i := range qData {
		qData[i] = float32(i) * 0.1
		kData[i] = float32(i) * 0.1
		vData[i] = float32(i+1) * 0.5
	}

	q, _ := tensor.New[float32]([]int{seqLen, dModel}, qData)
	k, _ := tensor.New[float32]([]int{seqLen, dModel}, kData)
	v, _ := tensor.New[float32]([]int{seqLen, dModel}, vData)

	out, err := MultiHeadAttention(ctx, engine, q, k, v, 1)
	if err != nil {
		t.Fatal(err)
	}

	outShape := out.Shape()
	if outShape[0] != seqLen || outShape[1] != dModel {
		t.Fatalf("expected shape [%d, %d], got %v", seqLen, dModel, outShape)
	}

	// With 1 head, this is standard scaled dot-product attention.
	// Manually compute: scores = q @ k^T / sqrt(d), attn = softmax(scores), out = attn @ v
	scale := float32(1.0 / math.Sqrt(float64(dModel)))

	// Compute scores [3,3]
	scores := make([]float32, seqLen*seqLen)
	for i := range seqLen {
		for j := range seqLen {
			var dot float32
			for d := range dModel {
				dot += qData[i*dModel+d] * kData[j*dModel+d]
			}
			scores[i*seqLen+j] = dot * scale
		}
	}

	// Softmax each row
	attn := make([]float32, seqLen*seqLen)
	for i := range seqLen {
		var maxVal float32 = -1e30
		for j := range seqLen {
			if scores[i*seqLen+j] > maxVal {
				maxVal = scores[i*seqLen+j]
			}
		}
		var sumExp float32
		for j := range seqLen {
			attn[i*seqLen+j] = float32(math.Exp(float64(scores[i*seqLen+j] - maxVal)))
			sumExp += attn[i*seqLen+j]
		}
		for j := range seqLen {
			attn[i*seqLen+j] /= sumExp
		}
	}

	// Compute expected output = attn @ v [3,4]
	expected := make([]float32, seqLen*dModel)
	for i := range seqLen {
		for d := range dModel {
			var sum float32
			for j := range seqLen {
				sum += attn[i*seqLen+j] * vData[j*dModel+d]
			}
			expected[i*dModel+d] = sum
		}
	}

	data := out.Data()
	for i, want := range expected {
		if math.Abs(float64(data[i]-want)) > 1e-4 {
			t.Errorf("out[%d] = %f, want %f", i, data[i], want)
		}
	}
}
