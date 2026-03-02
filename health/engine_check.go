package health

import (
	"context"
	"errors"
	"time"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// EngineCheck returns a CheckFunc that verifies a float32 CPUEngine is operational
// by performing a small tensor addition. The check fails if the operation takes
// longer than the given timeout or returns an error.
func EngineCheck(engine *compute.CPUEngine[float32], timeout time.Duration) CheckFunc {
	return func() error {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		a, err := tensor.New[float32]([]int{2}, []float32{1, 2})
		if err != nil {
			return errors.New("engine check: failed to create tensor")
		}

		b, err := tensor.New[float32]([]int{2}, []float32{3, 4})
		if err != nil {
			return errors.New("engine check: failed to create tensor")
		}

		result, err := engine.Add(ctx, a, b)
		if err != nil {
			return errors.New("engine check: Add operation failed")
		}

		data := result.Data()
		if len(data) != 2 || data[0] != 4 || data[1] != 6 {
			return errors.New("engine check: unexpected result")
		}

		return nil
	}
}

// EngineCheckGeneric returns a CheckFunc that verifies a generic CPUEngine is
// operational by performing a small tensor fill and zero.
func EngineCheckGeneric[T tensor.Numeric](engine *compute.CPUEngine[T], ops numeric.Arithmetic[T], timeout time.Duration) CheckFunc {
	return func() error {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		t, err := tensor.New[T]([]int{4}, nil)
		if err != nil {
			return errors.New("engine check: failed to create tensor")
		}

		if err := engine.Fill(ctx, t, ops.FromFloat64(1)); err != nil {
			return errors.New("engine check: Fill operation failed")
		}

		if err := engine.Zero(ctx, t); err != nil {
			return errors.New("engine check: Zero operation failed")
		}

		return nil
	}
}
