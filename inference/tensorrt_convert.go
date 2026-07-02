package inference

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/internal/tensorrt"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/ztensor/tensor"
)

// ShapeRange defines min/opt/max dimensions for a single input tensor.
// Used with DynamicShapeConfig to support variable-size inputs.
type ShapeRange struct {
	Min []int32
	Opt []int32
	Max []int32
}

// DynamicShapeConfig specifies per-input shape ranges for TensorRT optimization
// profiles. When non-nil, the converter creates an optimization profile that
// allows variable-size inputs within the specified ranges.
type DynamicShapeConfig struct {
	// InputShapes maps input index (0-based) to its shape range.
	InputShapes []ShapeRange
}

// trtConversionResult holds the result of converting a graph to TensorRT.
type trtConversionResult struct {
	// serialized is the TensorRT engine bytes ready for deserialization.
	serialized []byte
	// inputNames maps graph input node index to TRT tensor name.
	inputNames []string
	// outputName is the TRT output tensor name.
	outputName string
	// profileIndex is the optimization profile index (-1 if no dynamic shapes).
	profileIndex int
}

// supportedTRTOps lists the operation types that can be mapped to TensorRT layers.
var supportedTRTOps = map[string]bool{
	"Input":          true,
	"Constant":       true,
	"MatMul":         true,
	"Add":            true,
	"Sub":            true,
	"Mul":            true,
	"Div":            true,
	"Relu":           true,
	"Sigmoid":        true,
	"Tanh":           true,
	"Softmax":        true,
	"Reshape":        true,
	"ReduceSum":      true,
	"Conv":           true,
	"Transpose":      true,
	"Dense":          true, // Dense = MatMul + bias Add
	"Linear":         true, // Linear = MatMul (no bias)
}

// UnsupportedOpError lists the operations that cannot be converted to TensorRT.
type UnsupportedOpError struct {
	Ops []string
}

func (e *UnsupportedOpError) Error() string {
	return fmt.Sprintf("tensorrt: unsupported operations: %v", e.Ops)
}

// ConvertGraphToTRT walks a graph in topological order and maps each node to a
// TensorRT layer. Returns serialized engine bytes or an UnsupportedOpError if
// the graph contains operations that cannot be converted.
// If dynamicShapes is non-nil, an optimization profile is created with the
// specified min/opt/max dimensions for each input.
func ConvertGraphToTRT(g *graph.Graph[float32], workspaceBytes int, fp16 bool, dynamicShapes *DynamicShapeConfig) (*trtConversionResult, error) {
	if !tensorrt.Available() {
		return nil, fmt.Errorf("tensorrt convert: TensorRT library not available")
	}
	nodes := g.Nodes()

	// Check for unsupported ops first.
	var unsupported []string
	for _, n := range nodes {
		if !supportedTRTOps[n.OpType()] {
			unsupported = append(unsupported, n.OpType())
		}
	}
	if len(unsupported) > 0 {
		return nil, &UnsupportedOpError{Ops: unsupported}
	}

	// Create TRT objects.
	logger := tensorrt.CreateLogger(tensorrt.SeverityWarning)
	defer logger.Destroy()

	builder, err := tensorrt.CreateBuilder(logger)
	if err != nil {
		return nil, fmt.Errorf("tensorrt convert: %w", err)
	}
	defer builder.Destroy()

	network, err := builder.CreateNetwork()
	if err != nil {
		return nil, fmt.Errorf("tensorrt convert: %w", err)
	}
	defer network.Destroy()

	config, err := builder.CreateBuilderConfig()
	if err != nil {
		return nil, fmt.Errorf("tensorrt convert: %w", err)
	}
	defer config.Destroy()

	if workspaceBytes > 0 {
		config.SetMemoryPoolLimit(workspaceBytes)
	}
	if fp16 {
		config.SetFlag(tensorrt.FlagFP16)
	}

	// Map from graph Node to TRT Tensor for wiring layers.
	tensorMap := make(map[graph.Node[float32]]*tensorrt.Tensor)

	var inputNames []string

	// Walk in topological order.
	for _, node := range nodes {
		deps := g.Dependencies(node)

		switch node.OpType() {
		case "Input":
			shape := node.OutputShape()
			dims := toInt32Slice(shape)
			name := fmt.Sprintf("input_%d", len(inputNames))
			t := network.AddInput(name, tensorrt.Float32, dims)
			if t == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add input %q", name)
			}
			tensorMap[node] = t
			inputNames = append(inputNames, name)

		case "Constant":
			constNode, ok := node.(*core.Constant[float32])
			if !ok {
				return nil, fmt.Errorf("tensorrt convert: expected Constant node type")
			}
			val := constNode.GetValue()
			shape := val.Shape()
			data := val.Data()
			dims := toInt32Slice(shape)
			layer := network.AddConstant(dims, tensorrt.Float32,
				unsafe.Pointer(&data[0]), int64(len(data)))
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add constant")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "MatMul":
			if len(deps) != 2 {
				return nil, fmt.Errorf("tensorrt convert: MatMul expects 2 inputs, got %d", len(deps))
			}
			t0 := tensorMap[deps[0]]
			t1 := tensorMap[deps[1]]
			layer := network.AddMatrixMultiply(t0, tensorrt.MatrixOpNone,
				t1, tensorrt.MatrixOpNone)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add MatMul")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Add":
			if len(deps) != 2 {
				return nil, fmt.Errorf("tensorrt convert: Add expects 2 inputs, got %d", len(deps))
			}
			layer := network.AddElementWise(tensorMap[deps[0]], tensorMap[deps[1]],
				tensorrt.ElementWiseSum)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Add")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Sub":
			if len(deps) != 2 {
				return nil, fmt.Errorf("tensorrt convert: Sub expects 2 inputs, got %d", len(deps))
			}
			layer := network.AddElementWise(tensorMap[deps[0]], tensorMap[deps[1]],
				tensorrt.ElementWiseSub)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Sub")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Mul":
			if len(deps) != 2 {
				return nil, fmt.Errorf("tensorrt convert: Mul expects 2 inputs, got %d", len(deps))
			}
			layer := network.AddElementWise(tensorMap[deps[0]], tensorMap[deps[1]],
				tensorrt.ElementWiseProd)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Mul")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Div":
			if len(deps) != 2 {
				return nil, fmt.Errorf("tensorrt convert: Div expects 2 inputs, got %d", len(deps))
			}
			layer := network.AddElementWise(tensorMap[deps[0]], tensorMap[deps[1]],
				tensorrt.ElementWiseDiv)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Div")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Relu":
			if len(deps) != 1 {
				return nil, fmt.Errorf("tensorrt convert: ReLU expects 1 input, got %d", len(deps))
			}
			layer := network.AddActivation(tensorMap[deps[0]], tensorrt.ActivationReLU)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add ReLU")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Sigmoid":
			if len(deps) != 1 {
				return nil, fmt.Errorf("tensorrt convert: Sigmoid expects 1 input, got %d", len(deps))
			}
			layer := network.AddActivation(tensorMap[deps[0]], tensorrt.ActivationSigmoid)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Sigmoid")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Tanh":
			if len(deps) != 1 {
				return nil, fmt.Errorf("tensorrt convert: Tanh expects 1 input, got %d", len(deps))
			}
			layer := network.AddActivation(tensorMap[deps[0]], tensorrt.ActivationTanh)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Tanh")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Softmax":
			if len(deps) != 1 {
				return nil, fmt.Errorf("tensorrt convert: Softmax expects 1 input, got %d", len(deps))
			}
			attrs := node.Attributes()
			axis := -1
			if a, ok := attrs["axis"]; ok {
				if ai, ok := a.(int); ok {
					axis = ai
				}
			}
			// Default axis: last dimension.
			if axis < 0 {
				shape := node.OutputShape()
				axis = len(shape) - 1
			}
			layer := network.AddSoftMax(tensorMap[deps[0]], axis)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Softmax")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Reshape":
			if len(deps) < 1 {
				return nil, fmt.Errorf("tensorrt convert: Reshape expects at least 1 input, got %d", len(deps))
			}
			layer := network.AddShuffle(tensorMap[deps[0]])
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Shuffle/Reshape")
			}
			targetShape := node.OutputShape()
			tensorrt.ShuffleSetReshapeDims(layer, toInt32Slice(targetShape))
			tensorMap[node] = layer.GetOutput(0)

		case "Transpose":
			if len(deps) != 1 {
				return nil, fmt.Errorf("tensorrt convert: Transpose expects 1 input, got %d", len(deps))
			}
			layer := network.AddShuffle(tensorMap[deps[0]])
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Shuffle/Transpose")
			}
			attrs := node.Attributes()
			if perm, ok := attrs["perm"]; ok {
				if permSlice, ok := perm.([]int); ok {
					tensorrt.ShuffleSetFirstTranspose(layer, toInt32Slice(permSlice))
				}
			}
			tensorMap[node] = layer.GetOutput(0)

		case "ReduceSum":
			if len(deps) != 1 {
				return nil, fmt.Errorf("tensorrt convert: ReduceSum expects 1 input, got %d", len(deps))
			}
			attrs := node.Attributes()
			axes := uint32(0)
			keepDims := true
			if a, ok := attrs["axes"]; ok {
				if axList, ok := a.([]int); ok {
					for _, ax := range axList {
						axes |= 1 << uint(ax)
					}
				}
			}
			if kd, ok := attrs["keepdims"]; ok {
				if kdi, ok := kd.(bool); ok {
					keepDims = kdi
				}
			}
			layer := network.AddReduce(tensorMap[deps[0]], tensorrt.ReduceSum, axes, keepDims)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add ReduceSum")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Conv":
			if len(deps) < 1 {
				return nil, fmt.Errorf("tensorrt convert: Conv expects at least 1 input, got %d", len(deps))
			}
			params := node.Parameters()
			if len(params) < 1 {
				return nil, fmt.Errorf("tensorrt convert: Conv has no kernel weights")
			}
			kernel := params[0].Value
			kernelShape := kernel.Shape()
			kernelData := kernel.Data()
			nbOutputMaps := kernelShape[0]
			// Spatial kernel size (H, W for 2D conv).
			spatialDims := toInt32Slice(kernelShape[2:])

			var biasPtr unsafe.Pointer
			var biasCount int64
			if len(params) > 1 {
				biasData := params[1].Value.Data()
				biasPtr = unsafe.Pointer(&biasData[0])
				biasCount = int64(len(biasData))
			}

			layer := network.AddConvolutionNd(tensorMap[deps[0]], nbOutputMaps,
				spatialDims, unsafe.Pointer(&kernelData[0]), int64(len(kernelData)),
				biasPtr, biasCount)
			if layer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add Conv")
			}
			tensorMap[node] = layer.GetOutput(0)

		case "Dense", "Linear":
			// Dense/Linear nodes embed weights and optional bias as parameters.
			if len(deps) != 1 {
				return nil, fmt.Errorf("tensorrt convert: %s expects 1 input, got %d", node.OpType(), len(deps))
			}
			params := node.Parameters()
			if len(params) < 1 {
				return nil, fmt.Errorf("tensorrt convert: %s has no weight parameters", node.OpType())
			}
			// Add weight as constant.
			wt := params[0].Value
			wShape := wt.Shape()
			wData := wt.Data()
			wLayer := network.AddConstant(toInt32Slice(wShape), tensorrt.Float32,
				unsafe.Pointer(&wData[0]), int64(len(wData)))
			if wLayer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add weight constant for %s", node.OpType())
			}
			wTensor := wLayer.GetOutput(0)

			// MatMul: input * weights^T for Dense, input * weights for Linear.
			wOp := tensorrt.MatrixOpTranspose
			if node.OpType() == "Linear" {
				wOp = tensorrt.MatrixOpNone
			}
			mmLayer := network.AddMatrixMultiply(tensorMap[deps[0]], tensorrt.MatrixOpNone,
				wTensor, wOp)
			if mmLayer == nil {
				return nil, fmt.Errorf("tensorrt convert: failed to add MatMul for %s", node.OpType())
			}
			result := mmLayer.GetOutput(0)

			// Add bias if present.
			if len(params) > 1 {
				bData := params[1].Value.Data()
				bShape := params[1].Value.Shape()
				bLayer := network.AddConstant(toInt32Slice(bShape), tensorrt.Float32,
					unsafe.Pointer(&bData[0]), int64(len(bData)))
				if bLayer == nil {
					return nil, fmt.Errorf("tensorrt convert: failed to add bias constant for %s", node.OpType())
				}
				addLayer := network.AddElementWise(result, bLayer.GetOutput(0),
					tensorrt.ElementWiseSum)
				if addLayer == nil {
					return nil, fmt.Errorf("tensorrt convert: failed to add bias for %s", node.OpType())
				}
				result = addLayer.GetOutput(0)
			}
			tensorMap[node] = result

		default:
			return nil, fmt.Errorf("tensorrt convert: unhandled op type %q", node.OpType())
		}
	}

	// Mark the output node.
	outputNode := g.Output()
	outTensor, ok := tensorMap[outputNode]
	if !ok {
		return nil, fmt.Errorf("tensorrt convert: output node not in tensor map")
	}
	network.MarkOutput(outTensor)

	// Add optimization profile for dynamic shapes.
	profileIndex := -1
	if dynamicShapes != nil && len(dynamicShapes.InputShapes) > 0 {
		if len(dynamicShapes.InputShapes) != len(inputNames) {
			return nil, fmt.Errorf("tensorrt convert: dynamic shape config has %d entries but graph has %d inputs",
				len(dynamicShapes.InputShapes), len(inputNames))
		}
		profile, err := builder.CreateOptimizationProfile()
		if err != nil {
			return nil, fmt.Errorf("tensorrt convert: %w", err)
		}
		for i, sr := range dynamicShapes.InputShapes {
			if err := profile.SetDimensions(inputNames[i], sr.Min, sr.Opt, sr.Max); err != nil {
				return nil, fmt.Errorf("tensorrt convert: set dimensions for %q: %w", inputNames[i], err)
			}
		}
		idx, err := profile.AddToConfig(config)
		if err != nil {
			return nil, fmt.Errorf("tensorrt convert: add optimization profile: %w", err)
		}
		profileIndex = idx
	}

	// Build serialized engine.
	serialized, err := builder.BuildSerializedNetwork(network, config)
	if err != nil {
		return nil, fmt.Errorf("tensorrt convert: %w", err)
	}

	// Determine output tensor name from the built engine.
	runtime, err := tensorrt.CreateRuntime(logger)
	if err != nil {
		return nil, fmt.Errorf("tensorrt convert: %w", err)
	}
	defer runtime.Destroy()

	engine, err := runtime.DeserializeEngine(serialized)
	if err != nil {
		return nil, fmt.Errorf("tensorrt convert: %w", err)
	}
	defer engine.Destroy()

	// Find the output name (not in inputNames).
	outputName := ""
	inputSet := make(map[string]bool)
	for _, name := range inputNames {
		inputSet[name] = true
	}
	for i := 0; i < engine.NumIOTensors(); i++ {
		name := engine.GetIOTensorName(i)
		if !inputSet[name] {
			outputName = name
			break
		}
	}

	return &trtConversionResult{
		serialized:   serialized,
		inputNames:   inputNames,
		outputName:   outputName,
		profileIndex: profileIndex,
	}, nil
}

// ConstantValueGetter is an interface for nodes that hold constant tensor data.
type ConstantValueGetter interface {
	GetValue() *tensor.TensorNumeric[float32]
}

func toInt32Slice(s []int) []int32 {
	r := make([]int32, len(s))
	for i, v := range s {
		r[i] = int32(v)
	}
	return r
}
