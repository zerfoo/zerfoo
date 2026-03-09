package features

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// ---------- OutputShape ----------

func TestSpectralFingerprint_OutputShape_Comprehensive(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	o := numeric.Float32Ops{}
	tests := []struct {
		topK int
	}{
		{1},
		{4},
		{16},
	}
	for _, tt := range tests {
		layer := NewSpectralFingerprint[float32](eng, o, tt.topK)
		shape := layer.OutputShape()
		if len(shape) != 1 || shape[0] != tt.topK {
			t.Errorf("OutputShape() = %v, want [%d]", shape, tt.topK)
		}
	}
}

// ---------- Forward input count errors ----------

func TestSpectralFingerprint_Forward_WrongInputCount(t *testing.T) {
	layer := NewSpectralFingerprint[float32](compute.NewCPUEngine[float32](numeric.Float32Ops{}), numeric.Float32Ops{}, 4)
	input, _ := tensor.New[float32]([]int{1, 8}, make([]float32, 8))

	tests := []struct {
		name   string
		inputs []*tensor.TensorNumeric[float32]
	}{
		{"zero_inputs", nil},
		{"two_inputs", []*tensor.TensorNumeric[float32]{input, input}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := layer.Forward(context.Background(), tt.inputs...)
			if err == nil {
				t.Error("expected error for wrong input count")
			}
		})
	}
}

// ---------- Backward input count errors ----------

func TestSpectralFingerprint_Backward_WrongInputCount(t *testing.T) {
	layer := NewSpectralFingerprint[float32](compute.NewCPUEngine[float32](numeric.Float32Ops{}), numeric.Float32Ops{}, 4)
	input, _ := tensor.New[float32]([]int{1, 8}, make([]float32, 8))
	outputGrad, _ := tensor.New[float32]([]int{1, 4}, make([]float32, 4))

	tests := []struct {
		name   string
		inputs []*tensor.TensorNumeric[float32]
	}{
		{"zero_inputs", nil},
		{"two_inputs", []*tensor.TensorNumeric[float32]{input, input}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := layer.Backward(context.Background(), types.FullBackprop, outputGrad, tt.inputs...)
			if err == nil {
				t.Error("expected error for wrong input count")
			}
		})
	}
}

// ---------- Layer registration via model.GetLayerBuilder ----------

func TestSpectralFingerprint_LayerRegistration(t *testing.T) {
	builder, err := model.GetLayerBuilder[float32]("SpectralFingerprint")
	if err != nil {
		t.Fatalf("GetLayerBuilder failed: %v", err)
	}

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	t.Run("valid_attributes", func(t *testing.T) {
		node, err := builder(engine, ops, "test", nil, map[string]interface{}{"top_k": 8})
		if err != nil {
			t.Fatalf("builder error: %v", err)
		}
		sf, ok := node.(*SpectralFingerprint[float32])
		if !ok {
			t.Fatal("expected *SpectralFingerprint[float32]")
		}
		if sf.TopK != 8 {
			t.Errorf("TopK = %d, want 8", sf.TopK)
		}
	})

	t.Run("missing_top_k", func(t *testing.T) {
		_, err := builder(engine, ops, "test", nil, map[string]interface{}{})
		if err == nil {
			t.Error("expected error for missing top_k")
		}
	})

	t.Run("wrong_type_top_k", func(t *testing.T) {
		_, err := builder(engine, ops, "test", nil, map[string]interface{}{"top_k": "not_an_int"})
		if err == nil {
			t.Error("expected error for non-int top_k")
		}
	})
}

// ---------- Parameters returns nil ----------

func TestSpectralFingerprint_Parameters_IsNil(t *testing.T) {
	layer := NewSpectralFingerprint[float32](compute.NewCPUEngine[float32](numeric.Float32Ops{}), numeric.Float32Ops{}, 4)
	params := layer.Parameters()
	if params != nil {
		t.Errorf("Parameters() = %v, want nil", params)
	}
}

func TestSpectralFingerprint_Forward_NilInput(t *testing.T) {
	layer := NewSpectralFingerprint[float32](compute.NewCPUEngine[float32](numeric.Float32Ops{}), numeric.Float32Ops{}, 4)
	ctx := context.Background()

	_, err := layer.Forward(ctx, nil)
	if err == nil {
		t.Error("expected error for nil input")
	}
}

func TestSpectralFingerprint_Forward_EmptyInput(t *testing.T) {
	layer := NewSpectralFingerprint[float32](compute.NewCPUEngine[float32](numeric.Float32Ops{}), numeric.Float32Ops{}, 4)
	ctx := context.Background()

	empty, _ := tensor.New[float32]([]int{0}, nil)
	out, err := layer.Forward(ctx, empty)
	if err != nil {
		t.Fatalf("Forward with empty input: %v", err)
	}
	if out.Size() != 4 {
		t.Errorf("output size = %d, want 4", out.Size())
	}
	// All magnitudes should be zero for empty input.
	for i, v := range out.Data() {
		if v != 0 {
			t.Errorf("data[%d] = %v, want 0", i, v)
			break
		}
	}
}

// Statically assert graph.Node implementation
var _ graph.Node[float32] = (*SpectralFingerprint[float32])(nil)
