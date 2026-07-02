//go:build cuda && tensorrt

// Package tensorrt provides CGo bindings for the NVIDIA TensorRT inference
// optimization library. It wraps TensorRT's C++ API via a thin C shim
// (trt_capi.h/cpp) to enable CGo access from Go.
package tensorrt

/*
#cgo LDFLAGS: -L${SRCDIR} -ltrt_capi -lnvinfer -lstdc++
#include <stdlib.h>
#include "cshim/trt_capi.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Severity controls the minimum log level for TensorRT's internal logger.
type Severity int

const (
	SeverityInternalError Severity = C.TRT_SEVERITY_INTERNAL_ERROR
	SeverityError         Severity = C.TRT_SEVERITY_ERROR
	SeverityWarning       Severity = C.TRT_SEVERITY_WARNING
	SeverityInfo          Severity = C.TRT_SEVERITY_INFO
	SeverityVerbose       Severity = C.TRT_SEVERITY_VERBOSE
)

// DataType specifies the element type for TensorRT tensors.
type DataType int

const (
	Float32 DataType = C.TRT_FLOAT32
	Float16 DataType = C.TRT_FLOAT16
	Int8    DataType = C.TRT_INT8
	Int32   DataType = C.TRT_INT32
)

// ActivationType specifies the activation function.
type ActivationType int

const (
	ActivationReLU    ActivationType = C.TRT_ACTIVATION_RELU
	ActivationSigmoid ActivationType = C.TRT_ACTIVATION_SIGMOID
	ActivationTanh    ActivationType = C.TRT_ACTIVATION_TANH
)

// ElementWiseOp specifies the elementwise operation.
type ElementWiseOp int

const (
	ElementWiseSum  ElementWiseOp = C.TRT_ELEMENTWISE_SUM
	ElementWiseProd ElementWiseOp = C.TRT_ELEMENTWISE_PROD
	ElementWiseMax  ElementWiseOp = C.TRT_ELEMENTWISE_MAX
	ElementWiseMin  ElementWiseOp = C.TRT_ELEMENTWISE_MIN
	ElementWiseSub  ElementWiseOp = C.TRT_ELEMENTWISE_SUB
	ElementWiseDiv  ElementWiseOp = C.TRT_ELEMENTWISE_DIV
)

// MatrixOp specifies whether to transpose a matrix multiply operand.
type MatrixOp int

const (
	MatrixOpNone      MatrixOp = C.TRT_MATMUL_NONE
	MatrixOpTranspose MatrixOp = C.TRT_MATMUL_TRANSPOSE
)

// ReduceOp specifies the reduction operation.
type ReduceOp int

const (
	ReduceSum  ReduceOp = C.TRT_REDUCE_SUM
	ReduceProd ReduceOp = C.TRT_REDUCE_PROD
	ReduceMax  ReduceOp = C.TRT_REDUCE_MAX
	ReduceMin  ReduceOp = C.TRT_REDUCE_MIN
	ReduceAvg  ReduceOp = C.TRT_REDUCE_AVG
)

// BuilderFlag controls engine build options.
type BuilderFlag int

const (
	FlagFP16 BuilderFlag = C.TRT_FLAG_FP16
	FlagINT8 BuilderFlag = C.TRT_FLAG_INT8
)

// Logger wraps a TensorRT ILogger.
type Logger struct {
	ptr C.trt_logger_t
}

// CreateLogger creates a new TensorRT logger with the given minimum severity.
func CreateLogger(severity Severity) *Logger {
	return &Logger{ptr: C.trt_create_logger(C.trt_severity_t(severity))}
}

// Destroy releases the logger.
func (l *Logger) Destroy() {
	if l.ptr != nil {
		C.trt_destroy_logger(l.ptr)
		l.ptr = nil
	}
}

// Builder wraps a TensorRT IBuilder.
type Builder struct {
	ptr C.trt_builder_t
}

// CreateBuilder creates a new TensorRT builder.
func CreateBuilder(logger *Logger) (*Builder, error) {
	ptr := C.trt_create_builder(logger.ptr)
	if ptr == nil {
		return nil, fmt.Errorf("tensorrt: failed to create builder")
	}
	return &Builder{ptr: ptr}, nil
}

// Destroy releases the builder.
func (b *Builder) Destroy() {
	if b.ptr != nil {
		C.trt_destroy_builder(b.ptr)
		b.ptr = nil
	}
}

// CreateNetwork creates a new network definition with explicit batch mode.
func (b *Builder) CreateNetwork() (*NetworkDefinition, error) {
	ptr := C.trt_create_network(b.ptr)
	if ptr == nil {
		return nil, fmt.Errorf("tensorrt: failed to create network")
	}
	return &NetworkDefinition{ptr: ptr}, nil
}

// CreateBuilderConfig creates a new builder configuration.
func (b *Builder) CreateBuilderConfig() (*BuilderConfig, error) {
	ptr := C.trt_create_builder_config(b.ptr)
	if ptr == nil {
		return nil, fmt.Errorf("tensorrt: failed to create builder config")
	}
	return &BuilderConfig{ptr: ptr}, nil
}

// BuildSerializedNetwork builds an optimized engine from the network and returns
// serialized bytes. The caller must use Runtime.DeserializeEngine to load it.
func (b *Builder) BuildSerializedNetwork(network *NetworkDefinition, config *BuilderConfig) ([]byte, error) {
	mem := C.trt_builder_build_serialized_network(b.ptr, network.ptr, config.ptr)
	if mem == nil {
		return nil, fmt.Errorf("tensorrt: failed to build serialized network")
	}
	defer C.trt_destroy_host_memory(mem)

	data := C.trt_host_memory_data(mem)
	size := C.trt_host_memory_size(mem)
	if data == nil || size == 0 {
		return nil, fmt.Errorf("tensorrt: serialized network is empty")
	}

	result := make([]byte, int(size))
	copy(result, unsafe.Slice((*byte)(data), int(size)))
	return result, nil
}

// BuilderConfig wraps a TensorRT IBuilderConfig.
type BuilderConfig struct {
	ptr C.trt_builder_config_t
}

// Destroy releases the builder config.
func (c *BuilderConfig) Destroy() {
	if c.ptr != nil {
		C.trt_destroy_builder_config(c.ptr)
		c.ptr = nil
	}
}

// SetMemoryPoolLimit sets the maximum workspace memory for engine building.
func (c *BuilderConfig) SetMemoryPoolLimit(bytes int) {
	C.trt_builder_config_set_memory_pool_limit(c.ptr, C.size_t(bytes))
}

// SetFlag enables a builder flag (e.g., FP16 precision).
func (c *BuilderConfig) SetFlag(flag BuilderFlag) {
	C.trt_builder_config_set_flag(c.ptr, C.trt_builder_flag_t(flag))
}

// NetworkDefinition wraps a TensorRT INetworkDefinition.
type NetworkDefinition struct {
	ptr C.trt_network_t
}

// Destroy releases the network definition.
func (n *NetworkDefinition) Destroy() {
	if n.ptr != nil {
		C.trt_destroy_network(n.ptr)
		n.ptr = nil
	}
}

// NumInputs returns the number of network inputs.
func (n *NetworkDefinition) NumInputs() int {
	return int(C.trt_network_num_inputs(n.ptr))
}

// NumOutputs returns the number of network outputs.
func (n *NetworkDefinition) NumOutputs() int {
	return int(C.trt_network_num_outputs(n.ptr))
}

// NumLayers returns the number of network layers.
func (n *NetworkDefinition) NumLayers() int {
	return int(C.trt_network_num_layers(n.ptr))
}

// Tensor wraps a TensorRT ITensor pointer.
type Tensor struct {
	ptr C.trt_tensor_t
}

// Layer wraps a TensorRT ILayer pointer.
type Layer struct {
	ptr C.trt_layer_t
}

// GetOutput returns the output tensor at the given index.
func (l *Layer) GetOutput(index int) *Tensor {
	ptr := C.trt_layer_get_output(l.ptr, C.int(index))
	if ptr == nil {
		return nil
	}
	return &Tensor{ptr: ptr}
}

// SetName sets the layer name for debugging.
func (l *Layer) SetName(name string) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.trt_layer_set_name(l.ptr, cname)
}

// AddInput adds a network input tensor.
func (n *NetworkDefinition) AddInput(name string, dtype DataType, dims []int32) *Tensor {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	ptr := C.trt_network_add_input(n.ptr, cname, C.trt_data_type_t(dtype),
		C.int(len(dims)), (*C.int32_t)(unsafe.Pointer(&dims[0])))
	if ptr == nil {
		return nil
	}
	return &Tensor{ptr: ptr}
}

// MarkOutput marks a tensor as a network output.
func (n *NetworkDefinition) MarkOutput(tensor *Tensor) {
	C.trt_network_mark_output(n.ptr, tensor.ptr)
}

// AddActivation adds an activation layer.
func (n *NetworkDefinition) AddActivation(input *Tensor, actType ActivationType) *Layer {
	ptr := C.trt_network_add_activation(n.ptr, input.ptr,
		C.trt_activation_type_t(actType))
	if ptr == nil {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddElementWise adds an elementwise operation layer.
func (n *NetworkDefinition) AddElementWise(input1, input2 *Tensor, op ElementWiseOp) *Layer {
	ptr := C.trt_network_add_elementwise(n.ptr, input1.ptr, input2.ptr,
		C.trt_elementwise_op_t(op))
	if ptr == nil {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddMatrixMultiply adds a matrix multiplication layer.
func (n *NetworkDefinition) AddMatrixMultiply(input0 *Tensor, op0 MatrixOp,
	input1 *Tensor, op1 MatrixOp) *Layer {
	ptr := C.trt_network_add_matrix_multiply(n.ptr, input0.ptr,
		C.trt_matrix_op_t(op0), input1.ptr, C.trt_matrix_op_t(op1))
	if ptr == nil {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddSoftMax adds a softmax layer with the given axis.
func (n *NetworkDefinition) AddSoftMax(input *Tensor, axis int) *Layer {
	ptr := C.trt_network_add_softmax(n.ptr, input.ptr, C.int(axis))
	if ptr == nil {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddReduce adds a reduce layer.
func (n *NetworkDefinition) AddReduce(input *Tensor, op ReduceOp, reduceAxes uint32, keepDims bool) *Layer {
	kd := C.int(0)
	if keepDims {
		kd = 1
	}
	ptr := C.trt_network_add_reduce(n.ptr, input.ptr, C.trt_reduce_op_t(op),
		C.uint32_t(reduceAxes), kd)
	if ptr == nil {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddConstant adds a constant tensor layer.
func (n *NetworkDefinition) AddConstant(dims []int32, dtype DataType, weights unsafe.Pointer, count int64) *Layer {
	ptr := C.trt_network_add_constant(n.ptr, C.int(len(dims)),
		(*C.int32_t)(unsafe.Pointer(&dims[0])), C.trt_data_type_t(dtype),
		weights, C.int64_t(count))
	if ptr == nil {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddShuffle adds a shuffle (reshape/transpose) layer.
func (n *NetworkDefinition) AddShuffle(input *Tensor) *Layer {
	ptr := C.trt_network_add_shuffle(n.ptr, input.ptr)
	if ptr == nil {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddConvolutionNd adds an N-dimensional convolution layer.
func (n *NetworkDefinition) AddConvolutionNd(input *Tensor, nbOutputMaps int,
	kernelSize []int32, kernelWeights unsafe.Pointer, kernelCount int64,
	biasWeights unsafe.Pointer, biasCount int64) *Layer {
	ptr := C.trt_network_add_convolution_nd(n.ptr, input.ptr,
		C.int(nbOutputMaps), C.int(len(kernelSize)),
		(*C.int32_t)(unsafe.Pointer(&kernelSize[0])),
		kernelWeights, C.int64_t(kernelCount),
		biasWeights, C.int64_t(biasCount))
	if ptr == nil {
		return nil
	}
	return &Layer{ptr: ptr}
}

// ShuffleSetReshapeDims sets reshape dimensions on a shuffle layer.
func ShuffleSetReshapeDims(layer *Layer, dims []int32) {
	C.trt_shuffle_set_reshape_dims(layer.ptr, C.int(len(dims)),
		(*C.int32_t)(unsafe.Pointer(&dims[0])))
}

// ShuffleSetFirstTranspose sets the first transpose permutation on a shuffle layer.
func ShuffleSetFirstTranspose(layer *Layer, perm []int32) {
	C.trt_shuffle_set_first_transpose(layer.ptr, C.int(len(perm)),
		(*C.int32_t)(unsafe.Pointer(&perm[0])))
}

// Runtime wraps a TensorRT IRuntime for deserializing engines.
type Runtime struct {
	ptr C.trt_runtime_t
}

// CreateRuntime creates a new TensorRT runtime.
func CreateRuntime(logger *Logger) (*Runtime, error) {
	ptr := C.trt_create_runtime(logger.ptr)
	if ptr == nil {
		return nil, fmt.Errorf("tensorrt: failed to create runtime")
	}
	return &Runtime{ptr: ptr}, nil
}

// Destroy releases the runtime.
func (r *Runtime) Destroy() {
	if r.ptr != nil {
		C.trt_destroy_runtime(r.ptr)
		r.ptr = nil
	}
}

// DeserializeEngine loads a serialized engine.
func (r *Runtime) DeserializeEngine(data []byte) (*Engine, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("tensorrt: empty serialized engine data")
	}
	ptr := C.trt_deserialize_engine(r.ptr, unsafe.Pointer(&data[0]),
		C.size_t(len(data)))
	if ptr == nil {
		return nil, fmt.Errorf("tensorrt: failed to deserialize engine")
	}
	return &Engine{ptr: ptr}, nil
}

// Engine wraps a TensorRT ICudaEngine.
type Engine struct {
	ptr C.trt_engine_t
}

// Destroy releases the engine.
func (e *Engine) Destroy() {
	if e.ptr != nil {
		C.trt_destroy_engine(e.ptr)
		e.ptr = nil
	}
}

// NumIOTensors returns the number of input/output tensors.
func (e *Engine) NumIOTensors() int {
	return int(C.trt_engine_num_io_tensors(e.ptr))
}

// GetIOTensorName returns the name of the I/O tensor at the given index.
func (e *Engine) GetIOTensorName(index int) string {
	return C.GoString(C.trt_engine_get_io_tensor_name(e.ptr, C.int(index)))
}

// CreateExecutionContext creates an execution context for this engine.
func (e *Engine) CreateExecutionContext() (*ExecutionContext, error) {
	ptr := C.trt_create_execution_context(e.ptr)
	if ptr == nil {
		return nil, fmt.Errorf("tensorrt: failed to create execution context")
	}
	return &ExecutionContext{ptr: ptr}, nil
}

// ExecutionContext wraps a TensorRT IExecutionContext.
type ExecutionContext struct {
	ptr C.trt_context_t
}

// Destroy releases the execution context.
func (c *ExecutionContext) Destroy() {
	if c.ptr != nil {
		C.trt_destroy_execution_context(c.ptr)
		c.ptr = nil
	}
}

// SetTensorAddress binds a device pointer to a named tensor.
func (c *ExecutionContext) SetTensorAddress(name string, data unsafe.Pointer) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	if C.trt_context_set_tensor_address(c.ptr, cname, data) == 0 {
		return fmt.Errorf("tensorrt: failed to set tensor address for %q", name)
	}
	return nil
}

// EnqueueV3 enqueues inference on the given CUDA stream.
func (c *ExecutionContext) EnqueueV3(stream unsafe.Pointer) error {
	if C.trt_context_enqueue_v3(c.ptr, stream) == 0 {
		return fmt.Errorf("tensorrt: enqueueV3 failed")
	}
	return nil
}

// SetInputShape sets the input shape for a named tensor on the execution context.
// Required for dynamic shapes before calling EnqueueV3.
func (c *ExecutionContext) SetInputShape(name string, dims []int32) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	if C.trt_context_set_input_shape(c.ptr, cname, C.int(len(dims)),
		(*C.int32_t)(unsafe.Pointer(&dims[0]))) == 0 {
		return fmt.Errorf("tensorrt: failed to set input shape for %q", name)
	}
	return nil
}

// SetOptimizationProfile sets the active optimization profile on the context.
func (c *ExecutionContext) SetOptimizationProfile(index int) error {
	if C.trt_context_set_optimization_profile(c.ptr, C.int(index)) == 0 {
		return fmt.Errorf("tensorrt: failed to set optimization profile %d", index)
	}
	return nil
}

// OptimizationProfile wraps a TensorRT IOptimizationProfile.
type OptimizationProfile struct {
	ptr C.trt_optimization_profile_t
}

// CreateOptimizationProfile creates a new optimization profile from the builder.
func (b *Builder) CreateOptimizationProfile() (*OptimizationProfile, error) {
	ptr := C.trt_create_optimization_profile(b.ptr)
	if ptr == nil {
		return nil, fmt.Errorf("tensorrt: failed to create optimization profile")
	}
	return &OptimizationProfile{ptr: ptr}, nil
}

// SetDimensions sets the min/opt/max dimensions for a named input tensor.
func (p *OptimizationProfile) SetDimensions(inputName string, minDims, optDims, maxDims []int32) error {
	if len(minDims) != len(optDims) || len(minDims) != len(maxDims) {
		return fmt.Errorf("tensorrt: min/opt/max dims must have equal length")
	}
	cname := C.CString(inputName)
	defer C.free(unsafe.Pointer(cname))
	if C.trt_profile_set_dimensions(p.ptr, cname, C.int(len(minDims)),
		(*C.int32_t)(unsafe.Pointer(&minDims[0])),
		(*C.int32_t)(unsafe.Pointer(&optDims[0])),
		(*C.int32_t)(unsafe.Pointer(&maxDims[0]))) == 0 {
		return fmt.Errorf("tensorrt: failed to set dimensions for %q", inputName)
	}
	return nil
}

// AddToConfig adds this optimization profile to a builder config.
// Returns the profile index.
func (p *OptimizationProfile) AddToConfig(config *BuilderConfig) (int, error) {
	idx := int(C.trt_config_add_optimization_profile(config.ptr, p.ptr))
	if idx < 0 {
		return -1, fmt.Errorf("tensorrt: failed to add optimization profile")
	}
	return idx, nil
}
