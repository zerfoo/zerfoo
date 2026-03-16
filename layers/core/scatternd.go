package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ScatterND scatters updates into a copy of the data tensor at indices.
type ScatterND[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (s *ScatterND[T]) OpType() string                  { return "ScatterND" }
func (s *ScatterND[T]) Attributes() map[string]any       { return nil }
func (s *ScatterND[T]) OutputShape() []int               { return nil }
func (s *ScatterND[T]) Parameters() []*graph.Parameter[T] { return nil }

func (s *ScatterND[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 3 {
		return nil, fmt.Errorf("ScatterND requires 3 inputs (data, indices, updates), got %d", len(inputs))
	}

	dataShape := inputs[0].Shape()
	indicesShape := inputs[1].Shape()

	// Compute strides for the data tensor.
	strides := make([]int, len(dataShape))
	strides[len(strides)-1] = 1
	for i := len(strides) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * dataShape[i+1]
	}

	// The last dimension of indices is the index depth.
	indexDepth := indicesShape[len(indicesShape)-1]

	// Indices are always read on CPU (small tensor).
	indicesData := inputs[1].Data()

	// Number of scatter operations.
	numScatters := len(indicesData) / indexDepth

	// GPU path: data is on GPU, keep output GPU-resident.
	if dataGS, ok := inputs[0].GetStorage().(*tensor.GPUStorage[T]); ok {
		return s.forwardGPU(dataGS, dataShape, indicesData, inputs[2], strides, indexDepth, numScatters)
	}

	// CPU path.
	data := inputs[0].Data()
	updates := inputs[2].Data()

	// Copy data to output.
	out := make([]T, len(data))
	copy(out, data)

	// Elements per scatter update.
	elemPerUpdate := len(updates) / numScatters

	for i := range numScatters {
		// Compute flat offset from multi-dimensional index.
		offset := 0
		for d := range indexDepth {
			idx := int(indicesData[i*indexDepth+d])
			offset += idx * strides[d]
		}

		// Copy update elements.
		for j := range elemPerUpdate {
			if offset+j < len(out) {
				out[offset+j] = updates[i*elemPerUpdate+j]
			}
		}
	}

	return tensor.New(dataShape, out)
}

// forwardGPU performs ScatterND with GPU-resident data. The output stays on
// GPU. Indices are read on CPU to compute flat offsets, then update slices are
// copied into the output via D2D (GPU updates) or H2D (CPU updates) memcpy.
func (s *ScatterND[T]) forwardGPU(
	dataGS *tensor.GPUStorage[T],
	dataShape []int,
	indicesData []T,
	updatesTensor *tensor.TensorNumeric[T],
	strides []int,
	indexDepth, numScatters int,
) (*tensor.TensorNumeric[T], error) {
	totalElems := dataGS.Len()

	// Allocate output GPU storage and D2D copy data into it.
	outGS, err := tensor.NewGPUStorage[T](totalElems, dataGS.DeviceID())
	if err != nil {
		return nil, fmt.Errorf("ScatterND GPU: alloc output: %w", err)
	}
	if err := outGS.CopyFromDevice(dataGS, 0, 0, totalElems); err != nil {
		_ = outGS.Free()
		return nil, fmt.Errorf("ScatterND GPU: D2D copy data: %w", err)
	}

	// Determine elements per scatter update.
	updatesLen := 1
	for _, d := range updatesTensor.Shape() {
		updatesLen *= d
	}
	elemPerUpdate := updatesLen / numScatters

	// Check if updates are also on GPU.
	updatesGS, updatesOnGPU := updatesTensor.GetStorage().(*tensor.GPUStorage[T])

	for i := range numScatters {
		// Compute flat offset from multi-dimensional index on CPU.
		offset := 0
		for d := range indexDepth {
			idx := int(indicesData[i*indexDepth+d])
			offset += idx * strides[d]
		}

		copyLen := elemPerUpdate
		if offset+copyLen > totalElems {
			copyLen = totalElems - offset
		}
		if copyLen <= 0 {
			continue
		}

		if updatesOnGPU {
			// D2D copy from updates GPU storage into output at offset.
			if err := outGS.CopyFromDevice(updatesGS, offset, i*elemPerUpdate, copyLen); err != nil {
				_ = outGS.Free()
				return nil, fmt.Errorf("ScatterND GPU: D2D copy update %d: %w", i, err)
			}
		} else {
			// H2D copy from CPU updates slice into output at offset.
			updatesData := updatesTensor.Data()
			slice := updatesData[i*elemPerUpdate : i*elemPerUpdate+copyLen]
			if err := outGS.CopyFromHost(slice, offset); err != nil {
				_ = outGS.Free()
				return nil, fmt.Errorf("ScatterND GPU: H2D copy update %d: %w", i, err)
			}
		}
	}

	return tensor.NewWithStorage[T](dataShape, outGS)
}

func (s *ScatterND[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("ScatterND backward not implemented")
}

// BuildScatterND constructs a ScatterND node from attributes.
func BuildScatterND[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &ScatterND[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*ScatterND[float32])(nil)
