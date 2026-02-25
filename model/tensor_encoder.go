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
	case []float64:
		tensorProto.Dtype = zmf.Tensor_FLOAT64
		rawData, err = encodeFloat64(data)
	case []float16.Float16:
		tensorProto.Dtype = zmf.Tensor_FLOAT16
		rawData, err = encodeFloat16(data)
	case []int8:
		tensorProto.Dtype = zmf.Tensor_INT8
		rawData, err = encodeInt8(data)
	case []int16:
		tensorProto.Dtype = zmf.Tensor_INT16
		rawData, err = encodeInt16(data)
	case []int32:
		tensorProto.Dtype = zmf.Tensor_INT32
		rawData, err = encodeInt32(data)
	case []int:
		tensorProto.Dtype = zmf.Tensor_INT64
		rawData, err = encodeInt(data)
	case []int64:
		tensorProto.Dtype = zmf.Tensor_INT64
		rawData, err = encodeInt64(data)
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

func encodeFloat64(data []float64) ([]byte, error) {
	rawData := make([]byte, len(data)*8)
	for i, val := range data {
		bits := math.Float64bits(val)
		binary.LittleEndian.PutUint64(rawData[i*8:], bits)
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

func encodeInt16(data []int16) ([]byte, error) {
	rawData := make([]byte, len(data)*2)
	for i, val := range data {
		binary.LittleEndian.PutUint16(rawData[i*2:], uint16(val))
	}

	return rawData, nil
}

func encodeInt32(data []int32) ([]byte, error) {
	rawData := make([]byte, len(data)*4)
	for i, val := range data {
		binary.LittleEndian.PutUint32(rawData[i*4:], uint32(val))
	}

	return rawData, nil
}

func encodeInt(data []int) ([]byte, error) {
	rawData := make([]byte, len(data)*8)
	for i, val := range data {
		binary.LittleEndian.PutUint64(rawData[i*8:], uint64(val))
	}

	return rawData, nil
}

func encodeInt64(data []int64) ([]byte, error) {
	rawData := make([]byte, len(data)*8)
	for i, val := range data {
		binary.LittleEndian.PutUint64(rawData[i*8:], uint64(val))
	}

	return rawData, nil
}
