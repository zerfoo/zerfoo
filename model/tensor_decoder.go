// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
)

// DecodeTensor converts a ZMF Tensor protobuf message into a Zerfoo Tensor.
func DecodeTensor[T tensor.Numeric](tensorProto *zmf.Tensor) (*tensor.TensorNumeric[T], error) {
	shape := tensor.ConvertInt64ToInt(tensorProto.Shape)
	size := tensor.Product(shape)

	var zero T

	switch tensorProto.Dtype {
	case zmf.Tensor_FLOAT32:
		// Decode raw bytes into []float32
		if len(tensorProto.Data)%4 != 0 {
			return nil, fmt.Errorf("invalid float32 data length: must be a multiple of 4, got %d", len(tensorProto.Data))
		}

		f32 := make([]float32, size)
		for i := 0; i < size; i++ {
			bits := binary.LittleEndian.Uint32(tensorProto.Data[i*4 : i*4+4])
			f32[i] = math.Float32frombits(bits)
		}

		switch any(zero).(type) {
		case float32:
			data := any(f32).([]T)
			return tensor.New[T](shape, data)
		case float16.Float16:
			f16 := make([]float16.Float16, size)
			for i, v := range f32 {
				f16[i] = float16.FromFloat32(v)
			}
			data := any(f16).([]T)
			return tensor.New[T](shape, data)
		case float16.BFloat16:
			bf := make([]float16.BFloat16, size)
			for i, v := range f32 {
				bf[i] = float16.BFloat16FromFloat32(v)
			}
			data := any(bf).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for FLOAT32 source", zero)
		}

	case zmf.Tensor_FLOAT16:
		if len(tensorProto.Data)%2 != 0 {
			return nil, fmt.Errorf("invalid float16 data length: must be a multiple of 2, got %d", len(tensorProto.Data))
		}

		f16 := make([]float16.Float16, size)
		for i := 0; i < size; i++ {
			bits := binary.LittleEndian.Uint16(tensorProto.Data[i*2 : i*2+2])
			f16[i] = float16.FromBits(bits)
		}

		switch any(zero).(type) {
		case float16.Float16:
			data := any(f16).([]T)
			return tensor.New[T](shape, data)
		case float32:
			f32 := make([]float32, size)
			for i, v := range f16 {
				f32[i] = v.ToFloat32()
			}
			data := any(f32).([]T)
			return tensor.New[T](shape, data)
		case float16.BFloat16:
			bf := make([]float16.BFloat16, size)
			for i, v := range f16 {
				bf[i] = float16.BFloat16FromFloat32(v.ToFloat32())
			}
			data := any(bf).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for FLOAT16 source", zero)
		}

	case zmf.Tensor_INT8:
		if size != len(tensorProto.Data) {
			return nil, fmt.Errorf("invalid int8 data length: expected %d, got %d", size, len(tensorProto.Data))
		}

		switch any(zero).(type) {
		case int8:
			vals := make([]int8, size)
			for i := 0; i < size; i++ {
				vals[i] = int8(tensorProto.Data[i])
			}
			data := any(vals).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for INT8 source", zero)
		}

	default:
		return nil, fmt.Errorf("unsupported tensor dtype: %s", tensorProto.Dtype)
	}
}
