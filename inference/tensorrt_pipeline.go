package inference

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/tensorrt"
	"github.com/zerfoo/zerfoo/tensor"
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

	inputNames   []string
	outputName   string
	profileIndex int  // optimization profile index; -1 means static shapes
	dynamic      bool // whether dynamic shapes are enabled
}

// buildTRTEngine converts a graph to a TensorRT engine, using the cache if
// available. Returns a ready-to-use TRTInferenceEngine.
// If dynamicShapes is non-nil, the engine supports variable-size inputs.
func buildTRTEngine(g *graph.Graph[float32], modelID string, opts *loadOptions, dynamicShapes *DynamicShapeConfig) (*TRTInferenceEngine, error) {
	if !tensorrt.Available() {
		return nil, fmt.Errorf("tensorrt pipeline: TensorRT library not available")
	}
	precision := opts.precision
	if precision == "" {
		precision = "fp32"
	}
	fp16 := precision == "fp16"

	// Build cache key including dynamic shape ranges if present.
	cacheKeyPrecision := precision
	if dynamicShapes != nil && len(dynamicShapes.InputShapes) > 0 {
		cacheKeyPrecision = fmt.Sprintf("%s|dyn_%d", precision, len(dynamicShapes.InputShapes))
		for i, sr := range dynamicShapes.InputShapes {
			cacheKeyPrecision = fmt.Sprintf("%s|%d:%v-%v-%v", cacheKeyPrecision, i, sr.Min, sr.Opt, sr.Max)
		}
	}
	cacheKey, err := TRTCacheKey(modelID, cacheKeyPrecision)
	if err != nil {
		return nil, fmt.Errorf("tensorrt pipeline: cache key: %w", err)
	}

	var serialized []byte
	var inputNames []string
	var outputName string
	profileIndex := -1

	cached, err := LoadTRTEngine(cacheKey)
	if err != nil {
		return nil, fmt.Errorf("tensorrt pipeline: cache load: %w", err)
	}

	if cached != nil {
		// Cache hit -- use the cached engine.
		serialized = cached
		// We need to discover I/O names from the engine itself.
	} else {
		// Cache miss -- convert and build.
		result, err := ConvertGraphToTRT(g, 1<<28, fp16, dynamicShapes) // 256 MB workspace
		if err != nil {
			return nil, fmt.Errorf("tensorrt pipeline: convert: %w", err)
		}
		serialized = result.serialized
		inputNames = result.inputNames
		outputName = result.outputName
		profileIndex = result.profileIndex

		// Save to cache.
		if saveErr := SaveTRTEngine(cacheKey, serialized); saveErr != nil {
			// Non-fatal: log but continue.
			_ = saveErr
		}
	}

	// Create runtime and deserialize.
	logger := tensorrt.CreateLogger(tensorrt.SeverityWarning)

	rt, err := tensorrt.CreateRuntime(logger)
	if err != nil {
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt pipeline: %w", err)
	}

	engine, err := rt.DeserializeEngine(serialized)
	if err != nil {
		rt.Destroy()
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt pipeline: %w", err)
	}

	// Discover I/O names if from cache (no conversion result).
	if inputNames == nil {
		for i := 0; i < engine.NumIOTensors(); i++ {
			name := engine.GetIOTensorName(i)
			// Heuristic: input names start with "input_".
			if len(name) >= 6 && name[:6] == "input_" {
				inputNames = append(inputNames, name)
			} else {
				outputName = name
			}
		}
	}

	ctx, err := engine.CreateExecutionContext()
	if err != nil {
		engine.Destroy()
		rt.Destroy()
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt pipeline: %w", err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		ctx.Destroy()
		engine.Destroy()
		rt.Destroy()
		logger.Destroy()
		return nil, fmt.Errorf("tensorrt pipeline: %w", err)
	}

	ie := &TRTInferenceEngine{
		logger:       logger,
		runtime:      rt,
		engine:       engine,
		context:      ctx,
		stream:       stream,
		inputNames:   inputNames,
		outputName:   outputName,
		profileIndex: profileIndex,
		dynamic:      dynamicShapes != nil && len(dynamicShapes.InputShapes) > 0,
	}

	// Set the optimization profile on the context if dynamic shapes are used.
	if ie.dynamic && profileIndex >= 0 {
		if err := ctx.SetOptimizationProfile(profileIndex); err != nil {
			ie.Close()
			return nil, fmt.Errorf("tensorrt pipeline: %w", err)
		}
	}

	return ie, nil
}

// Forward runs inference through TensorRT with the given input tensors.
// Input tensors must already be on GPU.
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
