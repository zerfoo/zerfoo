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

	var (
		rawData []byte
		err     error
	)

	switch data := any(t.Data()).(type) {
	case []float32:
		tensorProto.Dtype = zmf.Tensor_FLOAT32
		rawData, err = encodeFloat32(data)
	case []float16.Float16:
		tensorProto.Dtype = zmf.Tensor_FLOAT16
		rawData, err = encodeFloat16(data)
	case []int8:
		tensorProto.Dtype = zmf.Tensor_INT8
		rawData, err = encodeInt8(data)
	default:
		return nil, fmt.Errorf("unsupported tensor type for encoding: %T", t.DType())
	}

	if err != nil {
		return nil, err
	}

	tensorProto.Data = rawData

	return tensorProto, nil
}

func encodeFloat32(data []float32) ([]byte, error) {
	rawData := make([]byte, len(data)*4)
	for i, val := range data {
		bits := math.Float32bits(val)
		binary.LittleEndian.PutUint32(rawData[i*4:], bits)
	}

	return rawData, nil
}

func encodeFloat16(data []float16.Float16) ([]byte, error) {
	rawData := make([]byte, len(data)*2)
	for i, val := range data {
		bits := val.Bits()
		binary.LittleEndian.PutUint16(rawData[i*2:], bits)
	}

	return rawData, nil
}

func encodeInt8(data []int8) ([]byte, error) {
	rawData := make([]byte, len(data))
	for i, v := range data {
		rawData[i] = byte(v)
	}

	return rawData, nil
}
