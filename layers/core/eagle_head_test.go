package core

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
)

func TestEAGLEHead_Forward(t *testing.T) {
	tests := []struct {
		name      string
		batch     int
		seq       int
		hidden    int
		wantShape []int
	}{
		{
			name:      "single batch single token",
			batch:     1,
			seq:       1,
			hidden:    64,
			wantShape: []int{1, 1, 64},
		},
		{
			name:      "multi batch single token",
			batch:     2,
			seq:       1,
			hidden:    64,
			wantShape: []int{2, 1, 64},
		},
		{
			name:      "single batch multi token",
			batch:     1,
			seq:       4,
			hidden:    32,
			wantShape: []int{1, 4, 32},
		},
		{
			name:      "multi batch multi token",
			batch:     3,
			seq:       5,
			hidden:    16,
			wantShape: []int{3, 5, 16},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
			ops := &numeric.Float32Ops{}

			head, err := NewEAGLEHead[float32](engine, ops, tt.hidden)
			testutils.AssertNoError(t, err, "NewEAGLEHead")
			testutils.AssertNotNil(t, head, "head should not be nil")

			// Create input tensor with deterministic data
			size := tt.batch * tt.seq * tt.hidden
			data := make([]float32, size)
			for i := range data {
				data[i] = float32(i%7-3) * 0.1 // range roughly [-0.3, 0.3]
			}
			input, err := tensor.New[float32]([]int{tt.batch, tt.seq, tt.hidden}, data)
			testutils.AssertNoError(t, err, "create input tensor")

			output, err := head.Forward(ctx, input)
			testutils.AssertNoError(t, err, "Forward")
			testutils.AssertNotNil(t, output, "output should not be nil")
			testutils.AssertTrue(t, reflect.DeepEqual(output.Shape(), tt.wantShape),
				"output shape mismatch")
		})
	}
}

func TestEAGLEHead_ForwardInputValidation(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	head, err := NewEAGLEHead[float32](engine, ops, 16)
	testutils.AssertNoError(t, err, "NewEAGLEHead")

	t.Run("no inputs", func(t *testing.T) {
		_, err := head.Forward(ctx)
		testutils.AssertNotNil(t, err, "should error with no inputs")
	})

	t.Run("2D input rejected", func(t *testing.T) {
		input, _ := tensor.New[float32]([]int{2, 16}, make([]float32, 32))
		_, err := head.Forward(ctx, input)
		testutils.AssertNotNil(t, err, "should error with 2D input")
	})
}

func TestEAGLEHead_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	head, err := NewEAGLEHead[float32](engine, ops, 32)
	testutils.AssertNoError(t, err, "NewEAGLEHead")

	params := head.Parameters()
	// LayerNorm: gamma + beta = 2
	// fc1: weights = 1
	// fc2: weights = 1
	// Total = 4
	testutils.AssertEqual(t, len(params), 4, "expected 4 parameters (gamma, beta, fc1_weights, fc2_weights)")
}

func TestEAGLEHead_OpType(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	head, err := NewEAGLEHead[float32](engine, ops, 16)
	testutils.AssertNoError(t, err, "NewEAGLEHead")

	testutils.AssertEqual(t, head.OpType(), "EAGLEHead", "OpType mismatch")
}
