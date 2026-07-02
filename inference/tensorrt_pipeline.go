package inference

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/tensorrt"
	"github.com/zerfoo/ztensor/tensor"
)

// TRTInferenceEngine holds a TensorRT engine and execution context for
// inference. It wraps the serialized engine, providing a Forward method
// that mirrors the graph forward pass but runs through TensorRT.
type TRTInferenceEngine struct {
	logger  *tensorrt.Logger
	runtime *tensorrt.Runtime
	engine  *tensorrt.Engine
	context *tensorrt.ExecutionContext
	stream  *cuda.Stream

	inputNames []string
	outputName string
	dynamic    bool // whether dynamic shapes are enabled
}

// Forward runs inference through TensorRT with the given input tensors.
// Input tensors must already be on GPU.
//
//nolint:errcheck // cuda.Free errors on cleanup paths are intentionally ignored
func (e *TRTInferenceEngine) Forward(inputs []*tensor.TensorNumeric[float32], outputSize int) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) != len(e.inputNames) {
		return nil, fmt.Errorf("tensorrt: expected %d inputs, got %d", len(e.inputNames), len(inputs))
	}

	// Bind input tensors and set input shapes for dynamic mode.
	for i, t := range inputs {
		if e.dynamic {
			dims := toInt32Slice(t.Shape())
			if err := e.context.SetInputShape(e.inputNames[i], dims); err != nil {
				return nil, fmt.Errorf("tensorrt: set input shape %d: %w", i, err)
			}
		}
		data := t.Data()
		if err := e.context.SetTensorAddress(e.inputNames[i], unsafe.Pointer(&data[0])); err != nil {
			return nil, fmt.Errorf("tensorrt: bind input %d: %w", i, err)
		}
	}

	// Allocate output.
	outputBytes := outputSize * 4 // float32
	outputDev, err := cuda.Malloc(outputBytes)
	if err != nil {
		return nil, fmt.Errorf("tensorrt: alloc output: %w", err)
	}

	if err := e.context.SetTensorAddress(e.outputName, outputDev); err != nil {
		cuda.Free(outputDev)
		return nil, fmt.Errorf("tensorrt: bind output: %w", err)
	}

	// Enqueue.
	if err := e.context.EnqueueV3(e.stream.Ptr()); err != nil {
		cuda.Free(outputDev)
		return nil, fmt.Errorf("tensorrt: enqueue: %w", err)
	}

	if err := e.stream.Synchronize(); err != nil {
		cuda.Free(outputDev)
		return nil, fmt.Errorf("tensorrt: sync: %w", err)
	}

	// Copy output back to CPU.
	result := make([]float32, outputSize)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), outputDev, outputBytes, cuda.MemcpyDeviceToHost); err != nil {
		cuda.Free(outputDev)
		return nil, fmt.Errorf("tensorrt: memcpy D2H: %w", err)
	}
	cuda.Free(outputDev)

	// Create result tensor.
	return tensor.New[float32]([]int{outputSize}, result)
}

// Close releases all TensorRT resources.
//
//nolint:errcheck // best-effort cleanup
func (e *TRTInferenceEngine) Close() error {
	if e.stream != nil {
		e.stream.Destroy()
	}
	if e.context != nil {
		e.context.Destroy()
	}
	if e.engine != nil {
		e.engine.Destroy()
	}
	if e.runtime != nil {
		e.runtime.Destroy()
	}
	if e.logger != nil {
		e.logger.Destroy()
	}
	return nil
}
