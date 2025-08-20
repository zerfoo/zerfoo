package attention

import (
	"context"

	"github.com/zerfoo/zerfoo/tensor"
)

// RopeScaler is an interface for layers that support scaling of RoPE.
type RopeScaler[T tensor.Numeric] interface {
	ScaleRope(ctx context.Context, factor float64) error
}
