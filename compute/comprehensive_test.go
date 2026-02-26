package compute

import (
	"context"
	"testing"

	float16 "github.com/zerfoo/float16"
	float8 "github.com/zerfoo/float8"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// ---------- Split validation ----------

func TestSplit_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()
	a, _ := tensor.New[float32]([]int{4, 6}, nil)

	tests := []struct {
		name      string
		input     *tensor.TensorNumeric[float32]
		numSplits int
		axis      int
	}{
		{"nil_input", nil, 2, 0},
		{"numSplits_zero", a, 0, 0},
		{"numSplits_negative", a, -1, 0},
		{"axis_out_of_bounds", a, 2, 5},
		{"axis_negative_oob", a, 2, -3},
		{"not_divisible", a, 5, 1}, // 6 not divisible by 5
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := eng.Split(ctx, tt.input, tt.numSplits, tt.axis)
			if err == nil {
				t.Errorf("expected error for %s", tt.name)
			}
		})
	}
}

func TestSplit_NegativeAxis(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()
	a, _ := tensor.New[float32]([]int{4, 6}, nil)

	// axis=-1 should be normalized to axis=1
	result, err := eng.Split(ctx, a, 2, -1)
	if err != nil {
		t.Fatalf("Split with axis=-1 failed: %v", err)
	}
	if len(result) != 2 {
		t.Errorf("expected 2 splits, got %d", len(result))
	}
	if result[0].Shape()[1] != 3 {
		t.Errorf("expected split dim=3, got %d", result[0].Shape()[1])
	}
}

// ---------- Gather validation ----------

func TestGather_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	params, _ := tensor.New[float32]([]int{4, 3}, nil)
	indices1D, _ := tensor.New[int]([]int{2}, []int{0, 1})
	output1D, _ := tensor.New[float32]([]int{2, 3}, nil)
	wrongOutput, _ := tensor.New[float32]([]int{3, 3}, nil)
	oobIndices, _ := tensor.New[int]([]int{2}, []int{0, 10})
	indices3D, _ := tensor.New[int]([]int{1, 1, 2}, []int{0, 1})
	params1D, _ := tensor.New[float32]([]int{4}, nil)

	t.Run("nil_params", func(t *testing.T) {
		err := eng.Gather(ctx, nil, indices1D, output1D)
		if err == nil {
			t.Error("expected error for nil params")
		}
	})

	t.Run("nil_indices", func(t *testing.T) {
		err := eng.Gather(ctx, params, nil, output1D)
		if err == nil {
			t.Error("expected error for nil indices")
		}
	})

	t.Run("nil_output", func(t *testing.T) {
		err := eng.Gather(ctx, params, indices1D, nil)
		if err == nil {
			t.Error("expected error for nil output")
		}
	})

	t.Run("non_2D_params", func(t *testing.T) {
		err := eng.Gather(ctx, params1D, indices1D, output1D)
		if err == nil {
			t.Error("expected error for 1D params")
		}
	})

	t.Run("1D_wrong_output_shape", func(t *testing.T) {
		err := eng.Gather(ctx, params, indices1D, wrongOutput)
		if err == nil {
			t.Error("expected error for wrong output shape")
		}
	})

	t.Run("1D_oob_index", func(t *testing.T) {
		err := eng.Gather(ctx, params, oobIndices, output1D)
		if err == nil {
			t.Error("expected error for out-of-bounds index")
		}
	})

	t.Run("3D_indices", func(t *testing.T) {
		err := eng.Gather(ctx, params, indices3D, output1D)
		if err == nil {
			t.Error("expected error for 3D indices")
		}
	})

	t.Run("2D_wrong_output_shape", func(t *testing.T) {
		indices2D, _ := tensor.New[int]([]int{1, 2}, []int{0, 1})
		wrongOut2D, _ := tensor.New[float32]([]int{1, 2, 5}, nil) // dim should be 3

		err := eng.Gather(ctx, params, indices2D, wrongOut2D)
		if err == nil {
			t.Error("expected error for 2D wrong output shape")
		}
	})

	t.Run("2D_oob_index", func(t *testing.T) {
		indices2D, _ := tensor.New[int]([]int{1, 2}, []int{0, 99})
		output2D, _ := tensor.New[float32]([]int{1, 2, 3}, nil)

		err := eng.Gather(ctx, params, indices2D, output2D)
		if err == nil {
			t.Error("expected error for 2D out-of-bounds index")
		}
	})
}

// ---------- ScatterAdd validation ----------

func TestScatterAdd_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	table, _ := tensor.New[float32]([]int{4, 3}, nil)
	indices, _ := tensor.New[int]([]int{2}, []int{0, 1})
	dOut, _ := tensor.New[float32]([]int{2, 3}, nil)

	t.Run("nil_inputs", func(t *testing.T) {
		err := eng.ScatterAdd(ctx, nil, indices, dOut)
		if err == nil {
			t.Error("expected error for nil table")
		}
	})

	t.Run("non_2D_table", func(t *testing.T) {
		table1D, _ := tensor.New[float32]([]int{12}, nil)
		err := eng.ScatterAdd(ctx, table1D, indices, dOut)
		if err == nil {
			t.Error("expected error for 1D table")
		}
	})

	t.Run("wrong_dOut_shape", func(t *testing.T) {
		wrongDOut, _ := tensor.New[float32]([]int{3, 3}, nil) // N should be 2, not 3
		err := eng.ScatterAdd(ctx, table, indices, wrongDOut)
		if err == nil {
			t.Error("expected error for wrong dOut shape")
		}
	})

	t.Run("oob_index", func(t *testing.T) {
		oobIdx, _ := tensor.New[int]([]int{2}, []int{0, 99})
		err := eng.ScatterAdd(ctx, table, oobIdx, dOut)
		if err == nil {
			t.Error("expected error for out-of-bounds index")
		}
	})
}

// ---------- AddScalar / MulScalar nil and dest mismatch ----------

func TestAddScalar_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("nil_input", func(t *testing.T) {
		_, err := eng.AddScalar(ctx, nil, 1.0)
		if err == nil {
			t.Error("expected error for nil input")
		}
	})

	t.Run("dest_shape_mismatch", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, nil)
		dst, _ := tensor.New[float32]([]int{3, 3}, nil)
		_, err := eng.AddScalar(ctx, a, 1.0, dst)
		if err == nil {
			t.Error("expected error for destination shape mismatch")
		}
	})
}

func TestMulScalar_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("nil_input", func(t *testing.T) {
		_, err := eng.MulScalar(ctx, nil, 1.0)
		if err == nil {
			t.Error("expected error for nil input")
		}
	})

	t.Run("dest_shape_mismatch", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, nil)
		dst, _ := tensor.New[float32]([]int{3, 3}, nil)
		_, err := eng.MulScalar(ctx, a, 1.0, dst)
		if err == nil {
			t.Error("expected error for destination shape mismatch")
		}
	})
}

// ---------- DivScalar nil and integer zero division ----------

func TestDivScalar_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("nil_input", func(t *testing.T) {
		_, err := eng.DivScalar(ctx, nil, 1.0)
		if err == nil {
			t.Error("expected error for nil input")
		}
	})

	t.Run("dest_shape_mismatch", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, nil)
		dst, _ := tensor.New[float32]([]int{3, 3}, nil)
		_, err := eng.DivScalar(ctx, a, 1.0, dst)
		if err == nil {
			t.Error("expected error for destination shape mismatch")
		}
	})
}

func TestDivScalar_IntegerZeroDivision(t *testing.T) {
	intOps := numeric.IntOps{}
	eng := NewCPUEngine[int](intOps)
	ctx := context.Background()

	a, _ := tensor.New[int]([]int{3}, []int{1, 2, 3})
	_, err := eng.DivScalar(ctx, a, 0)
	if err == nil {
		t.Error("expected division by zero error for integer engine")
	}
}

// ---------- Zeros nil input ----------

func TestZeros_NilInput(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	err := eng.Zeros(ctx, nil, nil)
	if err == nil {
		t.Error("expected error for nil input")
	}
}

// ---------- Fill nil input ----------

func TestFill_NilInput(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	err := eng.Fill(ctx, nil, 1.0)
	if err == nil {
		t.Error("expected error for nil input")
	}
}

// ---------- RandomUniform nil and min > max ----------

func TestRandomUniform_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("nil_input", func(t *testing.T) {
		err := eng.RandomUniform(ctx, nil, 0, 1)
		if err == nil {
			t.Error("expected error for nil input")
		}
	})

	t.Run("min_greater_than_max", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{10}, nil)
		err := eng.RandomUniform(ctx, a, 5.0, 1.0) // min > max triggers swap
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Verify all values are in [1.0, 5.0]
		for _, v := range a.Data() {
			if v < 1.0 || v > 5.0 {
				t.Errorf("value %f outside expected range [1.0, 5.0]", v)
			}
		}
	})
}

// ---------- Reshape validation ----------

func TestReshape_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()
	a, _ := tensor.New[float32]([]int{2, 3}, nil) // size=6

	tests := []struct {
		name  string
		input *tensor.TensorNumeric[float32]
		shape []int
	}{
		{"nil_input", nil, []int{6}},
		{"multiple_minus_one", a, []int{-1, -1}},
		{"zero_dimension", a, []int{0, 6}},
		{"negative_dimension", a, []int{-2, 3}},
		{"indivisible_infer", a, []int{-1, 4}}, // 6/4 not integer
		{"incompatible_size", a, []int{3, 3}},   // 9 != 6
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := eng.Reshape(ctx, tt.input, tt.shape)
			if err == nil {
				t.Errorf("expected error for %s", tt.name)
			}
		})
	}
}

// ---------- OneHot validation ----------

func TestOneHot_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("nil_input", func(t *testing.T) {
		_, err := eng.OneHot(ctx, nil, 10)
		if err == nil {
			t.Error("expected error for nil input")
		}
	})

	t.Run("depth_zero", func(t *testing.T) {
		input, _ := tensor.New[int]([]int{3}, []int{0, 1, 2})
		_, err := eng.OneHot(ctx, input, 0)
		if err == nil {
			t.Error("expected error for depth=0")
		}
	})

	t.Run("depth_negative", func(t *testing.T) {
		input, _ := tensor.New[int]([]int{3}, []int{0, 1, 2})
		_, err := eng.OneHot(ctx, input, -1)
		if err == nil {
			t.Error("expected error for negative depth")
		}
	})

	t.Run("index_oob", func(t *testing.T) {
		input, _ := tensor.New[int]([]int{3}, []int{0, 1, 10})
		_, err := eng.OneHot(ctx, input, 5) // index 10 >= depth 5
		if err == nil {
			t.Error("expected error for out-of-bounds index")
		}
	})

	t.Run("index_negative", func(t *testing.T) {
		input, _ := tensor.New[int]([]int{2}, []int{-1, 0})
		_, err := eng.OneHot(ctx, input, 5)
		if err == nil {
			t.Error("expected error for negative index")
		}
	})
}

// ---------- Concat validation ----------

func TestConcat_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, nil)
	b, _ := tensor.New[float32]([]int{2, 3}, nil)

	t.Run("empty_list", func(t *testing.T) {
		_, err := eng.Concat(ctx, []*tensor.TensorNumeric[float32]{}, 0)
		if err == nil {
			t.Error("expected error for empty tensor list")
		}
	})

	t.Run("axis_oob", func(t *testing.T) {
		_, err := eng.Concat(ctx, []*tensor.TensorNumeric[float32]{a, b}, 5)
		if err == nil {
			t.Error("expected error for axis out of bounds")
		}
	})

	t.Run("negative_axis", func(t *testing.T) {
		// axis=-1 should work (normalized to last dim)
		result, err := eng.Concat(ctx, []*tensor.TensorNumeric[float32]{a, b}, -1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Concat along dim 1: [2,3] + [2,3] = [2,6]
		if result.Shape()[1] != 6 {
			t.Errorf("expected dim 1 = 6, got %d", result.Shape()[1])
		}
	})

	t.Run("rank_mismatch", func(t *testing.T) {
		c, _ := tensor.New[float32]([]int{6}, nil)
		_, err := eng.Concat(ctx, []*tensor.TensorNumeric[float32]{a, c}, 0)
		if err == nil {
			t.Error("expected error for rank mismatch")
		}
	})

	t.Run("dim_mismatch", func(t *testing.T) {
		c, _ := tensor.New[float32]([]int{3, 3}, nil) // dim 0 is 3, not 2
		_, err := eng.Concat(ctx, []*tensor.TensorNumeric[float32]{a, c}, 1) // concat on axis 1, but axis 0 differs
		if err == nil {
			t.Error("expected error for non-axis dimension mismatch")
		}
	})
}

// ---------- Repeat validation ----------

func TestRepeat_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()
	a, _ := tensor.New[float32]([]int{2, 3}, nil)

	tests := []struct {
		name        string
		input       *tensor.TensorNumeric[float32]
		axis        int
		repetitions int
	}{
		{"nil_input", nil, 0, 2},
		{"axis_oob", a, 5, 2},
		{"repetitions_zero", a, 0, 0},
		{"repetitions_negative", a, 0, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := eng.Repeat(ctx, tt.input, tt.axis, tt.repetitions)
			if err == nil {
				t.Errorf("expected error for %s", tt.name)
			}
		})
	}
}

// ---------- Transpose validation ----------

func TestTranspose_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("nil_axes_3D_tensor", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3, 4}, nil)
		_, err := eng.Transpose(ctx, a, nil)
		if err == nil {
			t.Error("expected error for nil axes on 3D tensor")
		}
	})

	t.Run("axis_out_of_bounds", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, nil)
		_, err := eng.Transpose(ctx, a, []int{0, 5})
		if err == nil {
			t.Error("expected error for axis out of bounds")
		}
	})
}

// ---------- Softmax validation ----------

func TestSoftmax_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("nil_input", func(t *testing.T) {
		_, err := eng.Softmax(ctx, nil, -1)
		if err == nil {
			t.Error("expected error for nil input")
		}
	})

	t.Run("axis_oob", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, nil)
		_, err := eng.Softmax(ctx, a, 5)
		if err == nil {
			t.Error("expected error for axis out of bounds")
		}
	})
}

// ---------- Sum axis out of bounds ----------

func TestSum_AxisOOB(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, nil)
	_, err := eng.Sum(ctx, a, 5, false)
	if err == nil {
		t.Error("expected error for axis out of bounds")
	}
}

// ---------- ReduceMean global (axis < 0) ----------

func TestReduceMean_GlobalMean(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result, err := eng.ReduceMean(ctx, a, -1, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Mean of [1,2,3,4,5,6] = 3.5
	got := result.Data()[0]
	if got < 3.49 || got > 3.51 {
		t.Errorf("expected mean ~3.5, got %f", got)
	}
}

// ---------- MatMul validation ----------

func TestMatMul_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("1D_input", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{4}, nil)
		b, _ := tensor.New[float32]([]int{2, 3}, nil)
		_, err := eng.MatMul(ctx, a, b)
		if err == nil {
			t.Error("expected error for 1D input")
		}
	})

	t.Run("batch_dim_mismatch", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3, 4}, nil)
		b, _ := tensor.New[float32]([]int{3, 4, 5}, nil) // batch 3 != 2
		_, err := eng.MatMul(ctx, a, b)
		if err == nil {
			t.Error("expected error for batch dimension mismatch")
		}
	})

	t.Run("broadcasting_3D_x_2D", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3, 4}, nil)
		b, _ := tensor.New[float32]([]int{4, 5}, nil)
		result, err := eng.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Broadcasting: [2,3,4] x [4,5] = [2,3,5]
		shape := result.Shape()
		if len(shape) != 3 || shape[0] != 2 || shape[1] != 3 || shape[2] != 5 {
			t.Errorf("expected shape [2,3,5], got %v", shape)
		}
	})
}

// ---------- getOrCreateDest shape mismatch ----------

func TestGetOrCreateDest_ShapeMismatch(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, nil)
	wrongDst, _ := tensor.New[float32]([]int{3, 3}, nil)
	// Sum with global axis=-1 uses getOrCreateDest
	_, err := eng.Sum(ctx, a, -1, false, wrongDst)
	if err == nil {
		t.Error("expected error for destination shape mismatch")
	}
}

// ---------- OneHot dest mismatch ----------

func TestOneHot_DestMismatch(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	input, _ := tensor.New[int]([]int{3}, []int{0, 1, 2})
	wrongDst, _ := tensor.New[float32]([]int{5, 5}, nil) // should be [3, 5]
	_, err := eng.OneHot(ctx, input, 5, wrongDst)
	if err == nil {
		t.Error("expected error for OneHot destination shape mismatch")
	}
}

// ---------- Softmax scalar tensor ----------

func TestSoftmax_ScalarTensor(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{}, []float32{5.0})
	result, err := eng.Softmax(ctx, a, -1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Softmax of scalar is always 1.0
	got := result.Data()[0]
	if got < 0.99 || got > 1.01 {
		t.Errorf("expected softmax of scalar ~1.0, got %f", got)
	}
}

// ---------- Div integer zero detection ----------

func TestDiv_IntegerZeroDetection(t *testing.T) {
	intOps := numeric.IntOps{}
	eng := NewCPUEngine[int](intOps)
	ctx := context.Background()

	a, _ := tensor.New[int]([]int{3}, []int{1, 2, 3})
	b, _ := tensor.New[int]([]int{3}, []int{1, 0, 3}) // zero at index 1
	_, err := eng.Div(ctx, a, b)
	if err == nil {
		t.Error("expected error for integer division by zero")
	}
}

// ---------- MatMul type-specific branches ----------

func TestMatMul_Float16(t *testing.T) {
	ops := numeric.Float16Ops{}
	eng := NewCPUEngine[float16.Float16](ops)
	ctx := context.Background()

	aData := make([]float16.Float16, 6)
	bData := make([]float16.Float16, 6)
	for i := range aData {
		aData[i] = float16.FromFloat32(float32(i + 1))
		bData[i] = float16.FromFloat32(float32(i + 1))
	}

	a, _ := tensor.New[float16.Float16]([]int{2, 3}, aData)
	b, _ := tensor.New[float16.Float16]([]int{3, 2}, bData)
	result, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("float16 MatMul failed: %v", err)
	}
	shape := result.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Errorf("expected shape [2,2], got %v", shape)
	}
	// Verify non-zero result values to distinguish from zero-initialized output
	for i, v := range result.Data() {
		if v == float16.FromFloat32(0) {
			t.Errorf("result[%d] is zero; expected non-zero from matmul", i)
		}
	}
}

func TestMatMul_Float8(t *testing.T) {
	ops := numeric.Float8Ops{}
	eng := NewCPUEngine[float8.Float8](ops)
	ctx := context.Background()

	// Use 2x3 @ 3x2 with distinct data to differentiate from float16 test
	aData := make([]float8.Float8, 6)
	bData := make([]float8.Float8, 6)
	for i := range aData {
		aData[i] = float8.FromFloat64(float64(i + 1))
		bData[i] = float8.FromFloat64(float64(i + 1))
	}

	a, _ := tensor.New[float8.Float8]([]int{2, 3}, aData)
	b, _ := tensor.New[float8.Float8]([]int{3, 2}, bData)
	result, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("float8 MatMul failed: %v", err)
	}
	shape := result.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Errorf("expected shape [2,2], got %v", shape)
	}
}

func TestMatMul_Int8(t *testing.T) {
	ops := numeric.Int8Ops{}
	eng := NewCPUEngine[int8](ops)
	ctx := context.Background()

	a, _ := tensor.New[int8]([]int{2, 3}, []int8{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[int8]([]int{3, 2}, []int8{1, 2, 3, 4, 5, 6})
	result, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("int8 MatMul failed: %v", err)
	}
	shape := result.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Errorf("expected shape [2,2], got %v", shape)
	}
}

// ---------- ReduceMean Sum error propagation ----------

func TestReduceMean_SumError(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, nil)
	// axis 5 is out of bounds, causing Sum to fail
	_, err := eng.ReduceMean(ctx, a, 5, false)
	if err == nil {
		t.Error("expected error for axis out of bounds in ReduceMean")
	}
}
