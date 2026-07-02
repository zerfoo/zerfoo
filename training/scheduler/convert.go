package scheduler

import (
	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
	"github.com/zerfoo/ztensor/tensor"
)

// float64FromNumeric converts a tensor.Numeric value to float64.
func float64FromNumeric[T tensor.Numeric](v T) float64 {
	switch val := any(v).(type) {
	case float32:
		return float64(val)
	case float64:
		return val
	case float16.Float16:
		return float64(val.ToFloat32())
	case float16.BFloat16:
		return float64(val.ToFloat32())
	case float8.Float8:
		return float64(val.ToFloat32())
	case int:
		return float64(val)
	case int8:
		return float64(val)
	case int16:
		return float64(val)
	case int32:
		return float64(val)
	case int64:
		return float64(val)
	case uint:
		return float64(val)
	case uint8:
		return float64(val)
	case uint32:
		return float64(val)
	case uint64:
		return float64(val)
	default:
		return 0
	}
}

// converterFor returns a function that converts float64 to T.
func converterFor[T tensor.Numeric]() func(float64) T {
	var zero T
	switch any(zero).(type) {
	case float32:
		return func(v float64) T { return any(float32(v)).(T) }
	case float64:
		return func(v float64) T { return any(v).(T) }
	case float16.Float16:
		return func(v float64) T { return any(float16.FromFloat32(float32(v))).(T) }
	case float16.BFloat16:
		return func(v float64) T { return any(float16.BFloat16FromFloat32(float32(v))).(T) }
	case float8.Float8:
		return func(v float64) T { return any(float8.ToFloat8(float32(v))).(T) }
	default:
		return func(v float64) T { return any(int(v)).(T) }
	}
}
