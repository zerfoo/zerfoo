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

// EncodeTensor converts a Zerfoo Tensor into a ZMF Tensor protobuf message.
func EncodeTensor[T tensor.Numeric](t *tensor.TensorNumeric[T]) (*zmf.Tensor, error) {
	tensorProto := &zmf.Tensor{
		Shape: tensor.ConvertIntToInt64(t.Shape()),
	}

	var rawData []byte
	var err error

	// This is a bit of a hack. A better way would be to use type assertions
	// on a generic interface or pass the type information explicitly.
	var zero T
	switch any(zero).(type) {
	case float32:
		tensorProto.Dtype = zmf.Tensor_FLOAT32
		rawData, err = encodeFloat32(t.Data())
	case float16.Float16:
		tensorProto.Dtype = zmf.Tensor_FLOAT16
		rawData, err = encodeFloat16(t.Data())
	default:
		return nil, fmt.Errorf("unsupported tensor type for encoding: %T", zero)
	}

	if err != nil {
		return nil, err
	}

	tensorProto.Data = rawData
	return tensorProto, nil
}

func encodeFloat32[T tensor.Numeric](data []T) ([]byte, error) {
	rawData := make([]byte, len(data)*4)
	for i, val := range data {
		bits := math.Float32bits(float32(val))
		binary.LittleEndian.PutUint32(rawData[i*4:], bits)
	}
	return rawData, nil
}

func encodeFloat16[T tensor.Numeric](data []T) ([]byte, error) {
	rawData := make([]byte, len(data)*2)
	for i, val := range data {
		f16 := float16.FromFloat32(float32(val))
		bits := f16.Bits()
		binary.LittleEndian.PutUint16(rawData[i*2:], bits)
	}
	return rawData, nil
}
