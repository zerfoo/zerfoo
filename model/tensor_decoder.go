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
func DecodeTensor[T tensor.Numeric](tensorProto *zmf.Tensor) (*tensor.Tensor[T], error) {
	shape := tensor.ConvertInt64ToInt(tensorProto.Shape)
	data := make([]T, tensor.Product(shape))

	switch tensorProto.Dtype {
	case zmf.Tensor_FLOAT32:
		if err := decodeFloat32(tensorProto.Data, data); err != nil {
			return nil, err
		}
	case zmf.Tensor_FLOAT16:
		if err := decodeFloat16(tensorProto.Data, data); err != nil {
			return nil, err
		}
	// Add cases for other data types (Float8, Int32, etc.) here.
	default:
		return nil, fmt.Errorf("unsupported tensor dtype: %s", tensorProto.Dtype)
	}

	return tensor.New[T](shape, data)
}

func decodeFloat32[T tensor.Numeric](rawData []byte, dest []T) error {
	if len(rawData)%4 != 0 {
		return fmt.Errorf("invalid float32 data length: must be a multiple of 4, got %d", len(rawData))
	}
	for i := 0; i < len(rawData); i += 4 {
		bits := binary.LittleEndian.Uint32(rawData[i : i+4])
		floatVal := math.Float32frombits(bits)
		dest[i/4] = T(floatVal)
	}
	return nil
}

func decodeFloat16[T tensor.Numeric](rawData []byte, dest []T) error {
	if len(rawData)%2 != 0 {
		return fmt.Errorf("invalid float16 data length: must be a multiple of 2, got %d", len(rawData))
	}
	for i := 0; i < len(rawData); i += 2 {
		bits := binary.LittleEndian.Uint16(rawData[i : i+2])
		f16 := float16.FromBits(bits)
		dest[i/2] = T(f16.ToFloat32())
	}
	return nil
}
