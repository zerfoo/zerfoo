package transpose

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func makeEngine() compute.Engine[float32] {
	return compute.NewCPUEngine[float32](numeric.Float32Ops{})
}

func makeTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New(shape, data)
	if err != nil {
		t.Fatalf("makeTensor: %v", err)
	}
	return tn
}

func TestTranspose_OpType(t *testing.T) {
	tr := New[float32](makeEngine(), []int{1, 0})
	if tr.OpType() != "Transpose" {
		t.Errorf("OpType = %q, want %q", tr.OpType(), "Transpose")
	}
}

func TestTranspose_Attributes(t *testing.T) {
	tr := New[float32](makeEngine(), []int{1, 0})
	attrs := tr.Attributes()
	perm, ok := attrs["perm"].([]int)
	if !ok {
		t.Fatal("perm attribute not found or wrong type")
	}
	if len(perm) != 2 || perm[0] != 1 || perm[1] != 0 {
		t.Errorf("perm = %v, want [1, 0]", perm)
	}
}

func TestTranspose_OutputShape(t *testing.T) {
	tr := New[float32](makeEngine(), []int{1, 0})

	// Before forward, outputShape is nil
	if tr.OutputShape() != nil {
		t.Error("expected nil before forward")
	}

	// After forward, outputShape should be set
	input := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	_, err := tr.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	os := tr.OutputShape()
	if len(os) != 2 || os[0] != 3 || os[1] != 2 {
		t.Errorf("OutputShape = %v, want [3, 2]", os)
	}
}

func TestTranspose_Parameters(t *testing.T) {
	tr := New[float32](makeEngine(), []int{1, 0})
	if tr.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestTranspose_Backward(t *testing.T) {
	ctx := context.Background()
	eng := makeEngine()
	tr := New[float32](eng, []int{1, 0})

	input := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	_, err := tr.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	grad := makeTensor(t, []int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	grads, err := tr.Backward(ctx, types.FullBackprop, grad)
	if err != nil {
		t.Fatal(err)
	}

	if len(grads) != 1 {
		t.Fatalf("grads len = %d, want 1", len(grads))
	}

	// Inverse transpose of [3,2] with perm [1,0] -> inv perm [1,0] -> [2,3]
	if grads[0].Shape()[0] != 2 || grads[0].Shape()[1] != 3 {
		t.Errorf("grad shape = %v, want [2, 3]", grads[0].Shape())
	}
}

func TestTranspose_Backward_3D(t *testing.T) {
	ctx := context.Background()
	eng := makeEngine()
	tr := New[float32](eng, []int{0, 2, 1})

	input := makeTensor(t, []int{2, 3, 4}, make([]float32, 24))
	_, err := tr.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	grad := makeTensor(t, []int{2, 4, 3}, make([]float32, 24))
	grads, err := tr.Backward(ctx, types.FullBackprop, grad)
	if err != nil {
		t.Fatal(err)
	}

	// Inverse of [0,2,1] is [0,2,1]
	if grads[0].Shape()[0] != 2 || grads[0].Shape()[1] != 3 || grads[0].Shape()[2] != 4 {
		t.Errorf("grad shape = %v, want [2, 3, 4]", grads[0].Shape())
	}
}

func TestBuildTranspose_Errors(t *testing.T) {
	eng := makeEngine()
	ops := numeric.Float32Ops{}

	t.Run("missing_perm", func(t *testing.T) {
		_, err := BuildTranspose(eng, ops, "", nil, map[string]interface{}{})
		if err == nil {
			t.Error("expected error for missing perm")
		}
	})

	t.Run("invalid_perm_type", func(t *testing.T) {
		_, err := BuildTranspose(eng, ops, "", nil, map[string]interface{}{
			"perm": "bad",
		})
		if err == nil {
			t.Error("expected error for invalid perm type")
		}
	})

	t.Run("int64_perm", func(t *testing.T) {
		node, err := BuildTranspose(eng, ops, "", nil, map[string]interface{}{
			"perm": []int64{1, 0},
		})
		if err != nil {
			t.Fatal(err)
		}
		if node.OpType() != "Transpose" {
			t.Errorf("OpType = %q", node.OpType())
		}
	})
}
