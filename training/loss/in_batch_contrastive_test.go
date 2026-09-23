package loss

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func contrastiveTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	v, err := tensor.New(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestInBatchContrastiveForwardBackward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	loss, err := NewInBatchContrastive[float32](engine, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	qData := []float32{1, 0, 0, 1}
	dData := []float32{1, 0, 0, 1}
	target := contrastiveTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
	forward := func(qValues, dValues []float32) float64 {
		q := contrastiveTensor(t, []int{2, 2}, qValues)
		d := contrastiveTensor(t, []int{2, 2}, dValues)
		value, err := loss.Forward(ctx, q, d, target)
		if err != nil {
			t.Fatal(err)
		}
		return float64(value.Data()[0])
	}
	got := forward(qData, dData)
	want := math.Log1p(math.Exp(-2))
	if math.Abs(got-want) > 1e-5 {
		t.Fatalf("loss %.8f, want %.8f", got, want)
	}
	grad, err := loss.Backward(ctx, types.BackwardMode(0), contrastiveTensor(t, []int{1}, []float32{1}))
	if err != nil {
		t.Fatal(err)
	}
	eps := float32(1e-3)
	for i, analytic := range grad[0].Data() {
		plus := append([]float32(nil), qData...)
		minus := append([]float32(nil), qData...)
		plus[i] += eps
		minus[i] -= eps
		numericGrad := (forward(plus, dData) - forward(minus, dData)) / (2 * float64(eps))
		if math.Abs(float64(analytic)-numericGrad) > 2e-4 {
			t.Fatalf("query gradient %d: got %.6f want %.6f", i, analytic, numericGrad)
		}
	}
	forward(qData, dData)
	grad, err = loss.Backward(ctx, types.BackwardMode(0), contrastiveTensor(t, []int{1}, []float32{1}))
	if err != nil {
		t.Fatal(err)
	}
	for i, analytic := range grad[1].Data() {
		plus := append([]float32(nil), dData...)
		minus := append([]float32(nil), dData...)
		plus[i] += eps
		minus[i] -= eps
		numericGrad := (forward(qData, plus) - forward(qData, minus)) / (2 * float64(eps))
		if math.Abs(float64(analytic)-numericGrad) > 2e-4 {
			t.Fatalf("document gradient %d: got %.6f want %.6f", i, analytic, numericGrad)
		}
	}
}

func TestInBatchContrastiveRejectsInvalidShapes(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	if _, err := NewInBatchContrastive[float32](engine, 0); err == nil {
		t.Fatal("accepted zero temperature")
	}
	loss, err := NewInBatchContrastive[float32](engine, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	q := contrastiveTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
	d := contrastiveTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
	bad := contrastiveTensor(t, []int{2, 1}, []float32{1, 1})
	if _, err := loss.Forward(ctx, q, d, bad); err == nil {
		t.Fatal("accepted invalid targets")
	}
	if _, err := loss.Backward(ctx, types.BackwardMode(0), contrastiveTensor(t, []int{1}, []float32{1})); err == nil {
		t.Fatal("backward accepted missing forward")
	}
}

func TestInBatchContrastiveWeightedRectangularGradients(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	loss, err := NewInBatchContrastive[float32](engine, 0.7)
	if err != nil {
		t.Fatal(err)
	}
	qData := []float32{0.2, 0.7, -0.5, 0.4}
	dData := []float32{0.3, 0.9, -0.4, 0.6, 0.8, -0.2}
	target := contrastiveTensor(t, []int{2, 3}, []float32{1, 1, 0, 0, 0, 0})
	forward := func(qValues, dValues []float32) float64 {
		t.Helper()
		value, err := loss.Forward(ctx,
			contrastiveTensor(t, []int{2, 2}, qValues),
			contrastiveTensor(t, []int{3, 2}, dValues), target)
		if err != nil {
			t.Fatal(err)
		}
		return float64(value.Data()[0])
	}
	forward(qData, dData)
	grads, err := loss.Backward(ctx, types.BackwardMode(0), contrastiveTensor(t, []int{1}, []float32{1}))
	if err != nil {
		t.Fatal(err)
	}
	if grads[0].Data()[2] != 0 || grads[0].Data()[3] != 0 {
		t.Fatalf("all-zero target row changed query: %v", grads[0].Data())
	}
	eps := float32(1e-3)
	for i, analytic := range grads[0].Data() {
		plus := append([]float32(nil), qData...)
		minus := append([]float32(nil), qData...)
		plus[i] += eps
		minus[i] -= eps
		want := (forward(plus, dData) - forward(minus, dData)) / (2 * float64(eps))
		if math.Abs(float64(analytic)-want) > 2e-4 {
			t.Fatalf("query gradient %d: got %.6f want %.6f", i, analytic, want)
		}
	}
	forward(qData, dData)
	grads, err = loss.Backward(ctx, types.BackwardMode(0), contrastiveTensor(t, []int{1}, []float32{1}))
	if err != nil {
		t.Fatal(err)
	}
	for i, analytic := range grads[1].Data() {
		plus := append([]float32(nil), dData...)
		minus := append([]float32(nil), dData...)
		plus[i] += eps
		minus[i] -= eps
		want := (forward(qData, plus) - forward(qData, minus)) / (2 * float64(eps))
		if math.Abs(float64(analytic)-want) > 2e-4 {
			t.Fatalf("document gradient %d: got %.6f want %.6f", i, analytic, want)
		}
	}
}
