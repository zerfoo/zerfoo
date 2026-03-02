package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestBuildMul(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	node, err := BuildMul(engine, ops, "", nil, nil)
	if err != nil {
		t.Fatalf("BuildMul: %v", err)
	}
	if node.OpType() != "Mul" {
		t.Errorf("BuildMul OpType = %q, want %q", node.OpType(), "Mul")
	}
}

func TestBuildSub(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	node, err := BuildSub(engine, ops, "", nil, nil)
	if err != nil {
		t.Fatalf("BuildSub: %v", err)
	}
	if node.OpType() != "Sub" {
		t.Errorf("BuildSub OpType = %q, want %q", node.OpType(), "Sub")
	}
}

func TestBuildCast(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	node, err := BuildCast(engine, ops, "", nil, nil)
	if err != nil {
		t.Fatalf("BuildCast: %v", err)
	}
	if node.OpType() != "Cast" {
		t.Errorf("BuildCast OpType = %q, want %q", node.OpType(), "Cast")
	}
}

func TestBuildMatMul(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	node, err := BuildMatMul(engine, ops, "", nil, nil)
	if err != nil {
		t.Fatalf("BuildMatMul: %v", err)
	}
	if node.OpType() != "MatMul" {
		t.Errorf("BuildMatMul OpType = %q, want %q", node.OpType(), "MatMul")
	}
}

func TestBuildShape(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	node, err := BuildShape(engine, ops, "", nil, nil)
	if err != nil {
		t.Fatalf("BuildShape: %v", err)
	}
	if node.OpType() != "Shape" {
		t.Errorf("BuildShape OpType = %q, want %q", node.OpType(), "Shape")
	}
}

func TestBuildReshape(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	tests := []struct {
		name  string
		attrs map[string]any
		want  string // expected error substring, or empty for success
	}{
		{
			name:  "shape_from_int64_slice",
			attrs: map[string]any{"shape": []int64{3, 2}},
		},
		{
			name:  "shape_from_any_slice",
			attrs: map[string]any{"shape": []any{int64(3), int64(2)}},
		},
		{
			name:  "shape_unsupported_type",
			attrs: map[string]any{"shape": "invalid"},
			want:  "unsupported type",
		},
		{
			name:  "no_shape_attr_default",
			attrs: map[string]any{},
		},
		{
			name:  "non_shape_key_int64_slice",
			attrs: map[string]any{"target": []int64{4, 3}},
		},
		{
			name:  "non_shape_key_any_slice",
			attrs: map[string]any{"target": []any{int64(4), int64(3)}},
		},
		{
			name:  "non_shape_key_any_slice_int",
			attrs: map[string]any{"target": []any{int(4), int(3)}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := BuildReshape(engine, ops, "", nil, tt.attrs)
			if tt.want != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.want)
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildReshape: %v", err)
			}
			if node.OpType() != "Reshape" {
				t.Errorf("OpType = %q, want %q", node.OpType(), "Reshape")
			}
		})
	}
}

func TestBuildUnsqueeze(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	tests := []struct {
		name  string
		attrs map[string]any
		want  string
	}{
		{
			name:  "no_axes_default",
			attrs: map[string]any{},
		},
		{
			name:  "axes_int64_slice",
			attrs: map[string]any{"axes": []int64{0, 2}},
		},
		{
			name:  "axes_any_slice",
			attrs: map[string]any{"axes": []any{int64(0), int64(2)}},
		},
		{
			name:  "axes_unsupported_type",
			attrs: map[string]any{"axes": "invalid"},
			want:  "unsupported type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := BuildUnsqueeze(engine, ops, "", nil, tt.attrs)
			if tt.want != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.want)
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildUnsqueeze: %v", err)
			}
			if node.OpType() != "Unsqueeze" {
				t.Errorf("OpType = %q, want %q", node.OpType(), "Unsqueeze")
			}
		})
	}
}

func TestBuildConcat(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	tests := []struct {
		name  string
		attrs map[string]any
		want  string
	}{
		{
			name:  "no_axis_default",
			attrs: map[string]any{},
		},
		{
			name:  "axis_int64",
			attrs: map[string]any{"axis": int64(1)},
		},
		{
			name:  "axis_int",
			attrs: map[string]any{"axis": int(1)},
		},
		{
			name:  "axis_unsupported_type",
			attrs: map[string]any{"axis": "invalid"},
			want:  "unsupported type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := BuildConcat(engine, ops, "", nil, tt.attrs)
			if tt.want != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.want)
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildConcat: %v", err)
			}
			if node.OpType() != "Concat" {
				t.Errorf("OpType = %q, want %q", node.OpType(), "Concat")
			}
		})
	}
}

func TestBuildRotaryEmbedding(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	tests := []struct {
		name  string
		lName string
		attrs map[string]any
	}{
		{
			name:  "no_attrs",
			lName: "",
			attrs: nil,
		},
		{
			name:  "with_name",
			lName: "test_rope",
			attrs: nil,
		},
		{
			name:  "rope_base_float64",
			lName: "r",
			attrs: map[string]any{"rope_base": float64(50000)},
		},
		{
			name:  "rope_base_float32",
			lName: "r",
			attrs: map[string]any{"rope_base": float32(50000)},
		},
		{
			name:  "rope_base_int",
			lName: "r",
			attrs: map[string]any{"rope_base": int(50000)},
		},
		{
			name:  "rope_base_int64",
			lName: "r",
			attrs: map[string]any{"rope_base": int64(50000)},
		},
		{
			name:  "max_seq_len_int",
			lName: "r",
			attrs: map[string]any{"max_seq_len": int(512)},
		},
		{
			name:  "max_seq_len_int64",
			lName: "r",
			attrs: map[string]any{"max_seq_len": int64(512)},
		},
		{
			name:  "max_seq_len_float64",
			lName: "r",
			attrs: map[string]any{"max_seq_len": float64(512)},
		},
		{
			name:  "max_seq_len_float32",
			lName: "r",
			attrs: map[string]any{"max_seq_len": float32(512)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := BuildRotaryEmbedding(engine, ops, tt.lName, nil, tt.attrs)
			if err != nil {
				t.Fatalf("BuildRotaryEmbedding: %v", err)
			}
			if node == nil {
				t.Fatal("BuildRotaryEmbedding returned nil")
			}
		})
	}
}

func TestBuildFFN(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	// Create required parameters
	makeParam := func(name string, shape []int) *graph.Parameter[float32] {
		data := make([]float32, shape[0]*shape[1])
		for i := range data {
			data[i] = float32(i) * 0.01
		}
		ten, err := tensor.New[float32](shape, data)
		if err != nil {
			t.Fatalf("makeParam tensor: %v", err)
		}
		p, err := graph.NewParameter(name, ten, tensor.New[float32])
		if err != nil {
			t.Fatalf("makeParam param: %v", err)
		}
		return p
	}

	// biases
	makeBiasParam := func(name string, size int) *graph.Parameter[float32] {
		data := make([]float32, size)
		ten, err := tensor.New[float32]([]int{size}, data)
		if err != nil {
			t.Fatalf("makeBiasParam tensor: %v", err)
		}
		p, err := graph.NewParameter(name, ten, tensor.New[float32])
		if err != nil {
			t.Fatalf("makeBiasParam param: %v", err)
		}
		return p
	}

	tests := []struct {
		name   string
		attrs  map[string]any
		params map[string]*graph.Parameter[float32]
		want   string
	}{
		{
			name:  "missing_input_dim",
			attrs: map[string]any{},
			want:  "missing or invalid attribute: input_dim",
		},
		{
			name:  "missing_hidden_dim",
			attrs: map[string]any{"input_dim": 4},
			want:  "missing or invalid attribute: hidden_dim",
		},
		{
			name:  "missing_output_dim",
			attrs: map[string]any{"input_dim": 4, "hidden_dim": 8},
			want:  "missing or invalid attribute: output_dim",
		},
		{
			name:   "missing_w1_weights",
			attrs:  map[string]any{"input_dim": 4, "hidden_dim": 8, "output_dim": 4},
			params: map[string]*graph.Parameter[float32]{},
			want:   "missing required parameter",
		},
		{
			name:  "missing_w2_weights",
			attrs: map[string]any{"input_dim": 4, "hidden_dim": 8, "output_dim": 4},
			params: map[string]*graph.Parameter[float32]{
				"test_w1_weights": makeParam("test_w1_weights", []int{4, 8}),
			},
			want: "missing required parameter",
		},
		{
			name:  "missing_w3_weights",
			attrs: map[string]any{"input_dim": 4, "hidden_dim": 8, "output_dim": 4},
			params: map[string]*graph.Parameter[float32]{
				"test_w1_weights": makeParam("test_w1_weights", []int{4, 8}),
				"test_w2_weights": makeParam("test_w2_weights", []int{8, 4}),
			},
			want: "missing required parameter",
		},
		{
			name:  "success_with_bias",
			attrs: map[string]any{"input_dim": 4, "hidden_dim": 8, "output_dim": 4, "with_bias": true},
			params: map[string]*graph.Parameter[float32]{
				"test_w1_weights": makeParam("test_w1_weights", []int{4, 8}),
				"test_w2_weights": makeParam("test_w2_weights", []int{8, 4}),
				"test_w3_weights": makeParam("test_w3_weights", []int{4, 8}),
				"test_w1_biases":  makeBiasParam("test_w1_biases", 8),
				"test_w2_biases":  makeBiasParam("test_w2_biases", 4),
				"test_w3_biases":  makeBiasParam("test_w3_biases", 8),
			},
		},
		{
			name:  "success_no_bias_attr",
			attrs: map[string]any{"input_dim": 4, "hidden_dim": 8, "output_dim": 4},
			params: map[string]*graph.Parameter[float32]{
				"test_w1_weights": makeParam("test_w1_weights", []int{4, 8}),
				"test_w2_weights": makeParam("test_w2_weights", []int{8, 4}),
				"test_w3_weights": makeParam("test_w3_weights", []int{4, 8}),
				"test_w1_biases":  makeBiasParam("test_w1_biases", 8),
				"test_w2_biases":  makeBiasParam("test_w2_biases", 4),
				"test_w3_biases":  makeBiasParam("test_w3_biases", 8),
			},
		},
		{
			name:  "missing_w1_biases",
			attrs: map[string]any{"input_dim": 4, "hidden_dim": 8, "output_dim": 4, "with_bias": true},
			params: map[string]*graph.Parameter[float32]{
				"test_w1_weights": makeParam("test_w1_weights", []int{4, 8}),
				"test_w2_weights": makeParam("test_w2_weights", []int{8, 4}),
				"test_w3_weights": makeParam("test_w3_weights", []int{4, 8}),
			},
			want: "missing required parameter: test_w1_biases",
		},
		{
			name:  "missing_w2_biases",
			attrs: map[string]any{"input_dim": 4, "hidden_dim": 8, "output_dim": 4, "with_bias": true},
			params: map[string]*graph.Parameter[float32]{
				"test_w1_weights": makeParam("test_w1_weights", []int{4, 8}),
				"test_w2_weights": makeParam("test_w2_weights", []int{8, 4}),
				"test_w3_weights": makeParam("test_w3_weights", []int{4, 8}),
				"test_w1_biases":  makeBiasParam("test_w1_biases", 8),
			},
			want: "missing required parameter: test_w2_biases",
		},
		{
			name:  "missing_w3_biases",
			attrs: map[string]any{"input_dim": 4, "hidden_dim": 8, "output_dim": 4, "with_bias": true},
			params: map[string]*graph.Parameter[float32]{
				"test_w1_weights": makeParam("test_w1_weights", []int{4, 8}),
				"test_w2_weights": makeParam("test_w2_weights", []int{8, 4}),
				"test_w3_weights": makeParam("test_w3_weights", []int{4, 8}),
				"test_w1_biases":  makeBiasParam("test_w1_biases", 8),
				"test_w2_biases":  makeBiasParam("test_w2_biases", 4),
			},
			want: "missing required parameter: test_w3_biases",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := buildFFN(engine, ops, "test", tt.params, tt.attrs)
			if tt.want != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.want)
				}
				return
			}
			if err != nil {
				t.Fatalf("buildFFN: %v", err)
			}
			if node == nil {
				t.Fatal("buildFFN returned nil")
			}
		})
	}
}

func TestRotaryEmbedding_Extended(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	re := NewRotaryEmbedding[float32](engine)

	// Name / SetName
	if re.Name() != "RotaryEmbedding" {
		t.Errorf("Name = %q, want %q", re.Name(), "RotaryEmbedding")
	}
	re.SetName("test_rope")
	if re.Name() != "test_rope" {
		t.Errorf("after SetName, Name = %q, want %q", re.Name(), "test_rope")
	}

	// Parameters
	if p := re.Parameters(); p != nil {
		t.Errorf("Parameters = %v, want nil", p)
	}

	// OutputShape (before forward)
	if os := re.OutputShape(); os != nil {
		t.Errorf("OutputShape before Forward = %v, want nil", os)
	}

	// OpType
	if op := re.OpType(); op != "test_rope" {
		t.Errorf("OpType = %q, want %q", op, "test_rope")
	}

	// Attributes
	attr := re.Attributes()
	if attr == nil {
		t.Fatal("Attributes should not be nil")
	}
	if _, ok := attr["rope_base"]; !ok {
		t.Error("Attributes missing 'rope_base'")
	}
	if _, ok := attr["max_seq_len"]; !ok {
		t.Error("Attributes missing 'max_seq_len'")
	}

	// Forward with 2D input (non-3D pass-through)
	input2D := makeTensor(t, []int{2, 4}, make([]float32, 8))
	out, err := re.Forward(ctx, input2D)
	if err != nil {
		t.Fatalf("Forward 2D: %v", err)
	}
	if out != input2D {
		t.Error("Forward 2D should pass through")
	}

	// Backward with nil inner (pass-through)
	re2 := NewRotaryEmbedding[float32](engine)
	grad := makeTensor(t, []int{2, 4}, make([]float32, 8))
	grads, err := re2.Backward(ctx, types.FullBackprop, grad, input2D)
	if err != nil {
		t.Fatalf("Backward nil inner: %v", err)
	}
	if grads[0] != grad {
		t.Error("Backward nil inner should pass through")
	}

	// Forward with 3D input (triggers inner RoPE initialization)
	input3D := makeTensor(t, []int{1, 2, 4}, make([]float32, 8))
	out3D, err := re.Forward(ctx, input3D)
	if err != nil {
		t.Fatalf("Forward 3D: %v", err)
	}
	if out3D == nil {
		t.Fatal("Forward 3D output should not be nil")
	}

	// Backward through inner RoPE
	grad3D := makeTensor(t, []int{1, 2, 4}, make([]float32, 8))
	grads3D, err := re.Backward(ctx, types.FullBackprop, grad3D, input3D)
	if err != nil {
		t.Fatalf("Backward 3D: %v", err)
	}
	if len(grads3D) == 0 {
		t.Fatal("Backward 3D should return gradients")
	}
}

func TestBuildSpectralFingerprint_Extended(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	node, err := BuildSpectralFingerprint(engine, ops, "", nil, map[string]any{
		"window": 4,
		"top_k":  2,
	})
	if err != nil {
		t.Fatalf("BuildSpectralFingerprint: %v", err)
	}
	if node == nil {
		t.Fatal("BuildSpectralFingerprint returned nil")
	}

	// Test with a Forward pass to cover OutputShape, Parameters, Attributes
	sp := node.(*SpectralFingerprint[float32])

	// OutputShape before Forward
	if os := sp.OutputShape(); os != nil {
		t.Errorf("OutputShape before Forward = %v, want nil", os)
	}

	// Parameters
	if p := sp.Parameters(); p != nil {
		t.Errorf("Parameters = %v, want nil", p)
	}

	// Attributes
	attr := sp.Attributes()
	if attr == nil {
		t.Fatal("Attributes should not be nil")
	}
	if attr["window"] != 4 {
		t.Errorf("Attributes window = %v, want 4", attr["window"])
	}

	// Forward with valid input
	input := makeTensor(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	out, err := sp.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("SpectralFingerprint Forward: %v", err)
	}
	if s := out.Shape(); s[0] != 2 || s[1] != 2 {
		t.Errorf("SpectralFingerprint output shape = %v, want [2 2]", s)
	}

	// OutputShape after Forward
	if os := sp.OutputShape(); os[0] != 2 || os[1] != 2 {
		t.Errorf("OutputShape after Forward = %v, want [2 2]", os)
	}

	// Forward with topK >= window (covers zero magnitude path)
	sp2, err := NewSpectralFingerprint[float32](engine, ops, 3, 5)
	if err != nil {
		t.Fatalf("NewSpectralFingerprint: %v", err)
	}
	input2 := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
	out2, err := sp2.Forward(context.Background(), input2)
	if err != nil {
		t.Fatalf("SpectralFingerprint Forward topK>=window: %v", err)
	}
	if out2 == nil {
		t.Fatal("output should not be nil")
	}
}
