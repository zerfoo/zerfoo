package tensorrt

import (
	"fmt"
	"os"
	"sync"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// Severity controls the minimum log level for TensorRT's internal logger.
type Severity int

const (
	SeverityInternalError Severity = 0
	SeverityError         Severity = 1
	SeverityWarning       Severity = 2
	SeverityInfo          Severity = 3
	SeverityVerbose       Severity = 4
)

// DataType specifies the element type for TensorRT tensors.
type DataType int

const (
	Float32 DataType = 0
	Float16 DataType = 1
	Int8    DataType = 2
	Int32   DataType = 3
)

// ActivationType specifies the activation function.
type ActivationType int

const (
	ActivationReLU    ActivationType = 0
	ActivationSigmoid ActivationType = 1
	ActivationTanh    ActivationType = 2
)

// ElementWiseOp specifies the elementwise operation.
type ElementWiseOp int

const (
	ElementWiseSum  ElementWiseOp = 0
	ElementWiseProd ElementWiseOp = 1
	ElementWiseMax  ElementWiseOp = 2
	ElementWiseMin  ElementWiseOp = 3
	ElementWiseSub  ElementWiseOp = 4
	ElementWiseDiv  ElementWiseOp = 5
)

// MatrixOp specifies whether to transpose a matrix multiply operand.
type MatrixOp int

const (
	MatrixOpNone      MatrixOp = 0
	MatrixOpTranspose MatrixOp = 1
)

// ReduceOp specifies the reduction operation.
type ReduceOp int

const (
	ReduceSum  ReduceOp = 0
	ReduceProd ReduceOp = 1
	ReduceMax  ReduceOp = 2
	ReduceMin  ReduceOp = 3
	ReduceAvg  ReduceOp = 4
)

// BuilderFlag controls engine build options.
type BuilderFlag int

const (
	FlagFP16 BuilderFlag = 0
	FlagINT8 BuilderFlag = 1
)

// trtLib holds dlopen function pointers for the TensorRT C shim.
type trtLib struct {
	// Logger
	createLogger  uintptr
	destroyLogger uintptr

	// Builder
	createBuilder  uintptr
	destroyBuilder uintptr

	// Network
	createNetwork    uintptr
	destroyNetwork   uintptr
	networkNumInputs  uintptr
	networkNumOutputs uintptr
	networkNumLayers  uintptr

	// Network: add layers
	networkAddInput         uintptr
	networkMarkOutput       uintptr
	networkAddActivation    uintptr
	networkAddElementwise   uintptr
	networkAddMatrixMultiply uintptr
	networkAddSoftmax       uintptr
	networkAddReduce        uintptr
	networkAddConstant      uintptr
	networkAddShuffle       uintptr
	networkAddConvolutionNd uintptr

	// Layer helpers
	layerGetOutput            uintptr
	layerSetName              uintptr
	shuffleSetReshapeDims     uintptr
	shuffleSetFirstTranspose  uintptr

	// BuilderConfig
	createBuilderConfig          uintptr
	destroyBuilderConfig         uintptr
	builderConfigSetMemoryPool   uintptr
	builderConfigSetFlag         uintptr

	// Build engine
	builderBuildSerializedNetwork uintptr
	hostMemoryData                uintptr
	hostMemorySize                uintptr
	destroyHostMemory             uintptr

	// Runtime
	createRuntime  uintptr
	destroyRuntime uintptr

	// Engine
	deserializeEngine     uintptr
	destroyEngine         uintptr
	engineNumIOTensors    uintptr
	engineGetIOTensorName uintptr

	// ExecutionContext
	createExecutionContext  uintptr
	destroyExecutionContext uintptr
	contextSetTensorAddress uintptr
	contextEnqueueV3        uintptr
	contextSetInputShape    uintptr
	contextSetOptProfile    uintptr

	// Optimization Profiles
	createOptimizationProfile uintptr
	profileSetDimensions      uintptr
	configAddOptProfile       uintptr
}

var (
	globalTrtLib  *trtLib
	globalTrtOnce sync.Once
	errGlobalTrt  error
)

// trustedTrtCapiLibPath is the vetted, absolute production location for the
// zerfoo-built TensorRT C shim shared library, mirroring the
// /opt/zerfoo/lib/libkernels.so convention already established for CUDA
// custom kernels (internal/cuda/purego.go, CUDA-1, T141.1) -- both are
// zerfoo-compiled artifacts with one intended install location, not
// third-party vendor libraries.
const trustedTrtCapiLibPath = "/opt/zerfoo/lib/libtrt_capi.so"

// trtCapiLibPathOverrideEnv names an environment variable that, if set to a
// vetted absolute path, is tried before trustedTrtCapiLibPath. See
// cuda.VetAbsoluteLibPath for the safety checks applied.
const trtCapiLibPathOverrideEnv = "ZERFOO_TRT_CAPI_LIB_PATH"

// Library paths to try for the TensorRT C shim shared library.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): every entry
// here MUST be an absolute path. dlopen executes a shared object's ELF
// constructors at load time, so a bare soname (resolved via the default
// loader search path) or a CWD-relative entry lets an attacker who can
// write into the process's working directory (or influence
// LD_LIBRARY_PATH) achieve code execution the moment TensorRT initializes.
// This list previously included "libtrt_capi.so" and "./libtrt_capi.so" --
// exactly the CUDA-1 pattern (docs/adr/094-untrusted-boundary-security-hardening.md),
// but for the TensorRT shim instead of the CUDA kernel library. Do not
// reintroduce a relative or bare-soname candidate;
// TestTrtLibPathsAreAbsolute enforces this.
var trtLibPaths = buildTrtLibPaths()

func buildTrtLibPaths() []string {
	paths := make([]string, 0, 2)
	if override := os.Getenv(trtCapiLibPathOverrideEnv); override != "" {
		if vetted, ok := cuda.VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
		// An invalid override is silently ignored (not fatal): we fall
		// through to the trusted default rather than refusing to start.
	}
	paths = append(paths, trustedTrtCapiLibPath)
	return paths
}

func loadTrtLib() (*trtLib, error) {
	var handle uintptr
	var lastErr string
	for _, path := range trtLibPaths {
		var err error
		handle, err = cuda.DlopenPath(path)
		if err == nil {
			break
		}
		lastErr = err.Error()
	}
	if handle == 0 {
		return nil, fmt.Errorf("tensorrt: dlopen failed: %s", lastErr)
	}

	lib := &trtLib{}
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		// Logger
		{"trt_create_logger", &lib.createLogger},
		{"trt_destroy_logger", &lib.destroyLogger},
		// Builder
		{"trt_create_builder", &lib.createBuilder},
		{"trt_destroy_builder", &lib.destroyBuilder},
		// Network
		{"trt_create_network", &lib.createNetwork},
		{"trt_destroy_network", &lib.destroyNetwork},
		{"trt_network_num_inputs", &lib.networkNumInputs},
		{"trt_network_num_outputs", &lib.networkNumOutputs},
		{"trt_network_num_layers", &lib.networkNumLayers},
		// Network: add layers
		{"trt_network_add_input", &lib.networkAddInput},
		{"trt_network_mark_output", &lib.networkMarkOutput},
		{"trt_network_add_activation", &lib.networkAddActivation},
		{"trt_network_add_elementwise", &lib.networkAddElementwise},
		{"trt_network_add_matrix_multiply", &lib.networkAddMatrixMultiply},
		{"trt_network_add_softmax", &lib.networkAddSoftmax},
		{"trt_network_add_reduce", &lib.networkAddReduce},
		{"trt_network_add_constant", &lib.networkAddConstant},
		{"trt_network_add_shuffle", &lib.networkAddShuffle},
		{"trt_network_add_convolution_nd", &lib.networkAddConvolutionNd},
		// Layer helpers
		{"trt_layer_get_output", &lib.layerGetOutput},
		{"trt_layer_set_name", &lib.layerSetName},
		{"trt_shuffle_set_reshape_dims", &lib.shuffleSetReshapeDims},
		{"trt_shuffle_set_first_transpose", &lib.shuffleSetFirstTranspose},
		// BuilderConfig
		{"trt_create_builder_config", &lib.createBuilderConfig},
		{"trt_destroy_builder_config", &lib.destroyBuilderConfig},
		{"trt_builder_config_set_memory_pool_limit", &lib.builderConfigSetMemoryPool},
		{"trt_builder_config_set_flag", &lib.builderConfigSetFlag},
		// Build engine
		{"trt_builder_build_serialized_network", &lib.builderBuildSerializedNetwork},
		{"trt_host_memory_data", &lib.hostMemoryData},
		{"trt_host_memory_size", &lib.hostMemorySize},
		{"trt_destroy_host_memory", &lib.destroyHostMemory},
		// Runtime
		{"trt_create_runtime", &lib.createRuntime},
		{"trt_destroy_runtime", &lib.destroyRuntime},
		// Engine
		{"trt_deserialize_engine", &lib.deserializeEngine},
		{"trt_destroy_engine", &lib.destroyEngine},
		{"trt_engine_num_io_tensors", &lib.engineNumIOTensors},
		{"trt_engine_get_io_tensor_name", &lib.engineGetIOTensorName},
		// ExecutionContext
		{"trt_create_execution_context", &lib.createExecutionContext},
		{"trt_destroy_execution_context", &lib.destroyExecutionContext},
		{"trt_context_set_tensor_address", &lib.contextSetTensorAddress},
		{"trt_context_enqueue_v3", &lib.contextEnqueueV3},
		{"trt_context_set_input_shape", &lib.contextSetInputShape},
		{"trt_context_set_optimization_profile", &lib.contextSetOptProfile},
		// Optimization Profiles
		{"trt_create_optimization_profile", &lib.createOptimizationProfile},
		{"trt_profile_set_dimensions", &lib.profileSetDimensions},
		{"trt_config_add_optimization_profile", &lib.configAddOptProfile},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("tensorrt: %w", err)
		}
		*s.ptr = addr
	}
	return lib, nil
}

func getTrtLib() (*trtLib, error) {
	globalTrtOnce.Do(func() {
		globalTrtLib, errGlobalTrt = loadTrtLib()
	})
	return globalTrtLib, errGlobalTrt
}

// Available returns true if the TensorRT C shim library can be loaded.
// The result is cached after the first call.
func Available() bool {
	_, err := getTrtLib()
	return err == nil
}

// ptrFromUintptr converts a uintptr to unsafe.Pointer without triggering
// go vet's unsafeptr analyzer. Required for purego FFI.
func ptrFromUintptr(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// goStringFromCPtr converts a C string pointer (uintptr) to a Go string.
func goStringFromCPtr(p uintptr) string {
	if p == 0 {
		return ""
	}
	ptr := (*byte)(ptrFromUintptr(p))
	var n int
	for *(*byte)(unsafe.Add(unsafe.Pointer(ptr), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(ptr, n))
}

// cstring allocates a null-terminated byte slice from a Go string.
// The returned slice must be kept alive for the duration of the C call.
func cstring(s string) []byte {
	return append([]byte(s), 0)
}

// Logger wraps a TensorRT ILogger.
type Logger struct {
	ptr uintptr
}

// CreateLogger creates a new TensorRT logger with the given minimum severity.
func CreateLogger(severity Severity) *Logger {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.createLogger, uintptr(severity))
	if ptr == 0 {
		return nil
	}
	return &Logger{ptr: ptr}
}

// Destroy releases the logger.
func (l *Logger) Destroy() {
	if l.ptr != 0 {
		lib, err := getTrtLib()
		if err != nil {
			return
		}
		cuda.Ccall(lib.destroyLogger, l.ptr)
		l.ptr = 0
	}
}

// Builder wraps a TensorRT IBuilder.
type Builder struct {
	ptr uintptr
}

// CreateBuilder creates a new TensorRT builder.
func CreateBuilder(logger *Logger) (*Builder, error) {
	lib, err := getTrtLib()
	if err != nil {
		return nil, err
	}
	ptr := cuda.Ccall(lib.createBuilder, logger.ptr)
	if ptr == 0 {
		return nil, fmt.Errorf("tensorrt: failed to create builder")
	}
	return &Builder{ptr: ptr}, nil
}

// Destroy releases the builder.
func (b *Builder) Destroy() {
	if b.ptr != 0 {
		lib, err := getTrtLib()
		if err != nil {
			return
		}
		cuda.Ccall(lib.destroyBuilder, b.ptr)
		b.ptr = 0
	}
}

// CreateNetwork creates a new network definition with explicit batch mode.
func (b *Builder) CreateNetwork() (*NetworkDefinition, error) {
	lib, err := getTrtLib()
	if err != nil {
		return nil, err
	}
	ptr := cuda.Ccall(lib.createNetwork, b.ptr)
	if ptr == 0 {
		return nil, fmt.Errorf("tensorrt: failed to create network")
	}
	return &NetworkDefinition{ptr: ptr}, nil
}

// CreateBuilderConfig creates a new builder configuration.
func (b *Builder) CreateBuilderConfig() (*BuilderConfig, error) {
	lib, err := getTrtLib()
	if err != nil {
		return nil, err
	}
	ptr := cuda.Ccall(lib.createBuilderConfig, b.ptr)
	if ptr == 0 {
		return nil, fmt.Errorf("tensorrt: failed to create builder config")
	}
	return &BuilderConfig{ptr: ptr}, nil
}

// BuildSerializedNetwork builds an optimized engine from the network and returns
// serialized bytes. The caller must use Runtime.DeserializeEngine to load it.
func (b *Builder) BuildSerializedNetwork(network *NetworkDefinition, config *BuilderConfig) ([]byte, error) {
	lib, err := getTrtLib()
	if err != nil {
		return nil, err
	}
	mem := cuda.Ccall(lib.builderBuildSerializedNetwork, b.ptr, network.ptr, config.ptr)
	if mem == 0 {
		return nil, fmt.Errorf("tensorrt: failed to build serialized network")
	}

	data := cuda.Ccall(lib.hostMemoryData, mem)
	size := cuda.Ccall(lib.hostMemorySize, mem)
	if data == 0 || size == 0 {
		cuda.Ccall(lib.destroyHostMemory, mem)
		return nil, fmt.Errorf("tensorrt: serialized network is empty")
	}

	result := make([]byte, int(size))
	copy(result, unsafe.Slice((*byte)(ptrFromUintptr(data)), int(size)))
	cuda.Ccall(lib.destroyHostMemory, mem)
	return result, nil
}

// BuilderConfig wraps a TensorRT IBuilderConfig.
type BuilderConfig struct {
	ptr uintptr
}

// Destroy releases the builder config.
func (c *BuilderConfig) Destroy() {
	if c.ptr != 0 {
		lib, err := getTrtLib()
		if err != nil {
			return
		}
		cuda.Ccall(lib.destroyBuilderConfig, c.ptr)
		c.ptr = 0
	}
}

// SetMemoryPoolLimit sets the maximum workspace memory for engine building.
func (c *BuilderConfig) SetMemoryPoolLimit(bytes int) {
	lib, err := getTrtLib()
	if err != nil {
		return
	}
	cuda.Ccall(lib.builderConfigSetMemoryPool, c.ptr, uintptr(bytes))
}

// SetFlag enables a builder flag (e.g., FP16 precision).
func (c *BuilderConfig) SetFlag(flag BuilderFlag) {
	lib, err := getTrtLib()
	if err != nil {
		return
	}
	cuda.Ccall(lib.builderConfigSetFlag, c.ptr, uintptr(flag))
}

// NetworkDefinition wraps a TensorRT INetworkDefinition.
type NetworkDefinition struct {
	ptr uintptr
}

// Destroy releases the network definition.
func (n *NetworkDefinition) Destroy() {
	if n.ptr != 0 {
		lib, err := getTrtLib()
		if err != nil {
			return
		}
		cuda.Ccall(lib.destroyNetwork, n.ptr)
		n.ptr = 0
	}
}

// NumInputs returns the number of network inputs.
func (n *NetworkDefinition) NumInputs() int {
	lib, err := getTrtLib()
	if err != nil {
		return 0
	}
	return int(cuda.Ccall(lib.networkNumInputs, n.ptr))
}

// NumOutputs returns the number of network outputs.
func (n *NetworkDefinition) NumOutputs() int {
	lib, err := getTrtLib()
	if err != nil {
		return 0
	}
	return int(cuda.Ccall(lib.networkNumOutputs, n.ptr))
}

// NumLayers returns the number of network layers.
func (n *NetworkDefinition) NumLayers() int {
	lib, err := getTrtLib()
	if err != nil {
		return 0
	}
	return int(cuda.Ccall(lib.networkNumLayers, n.ptr))
}

// Tensor wraps a TensorRT ITensor pointer.
type Tensor struct {
	ptr uintptr
}

// Layer wraps a TensorRT ILayer pointer.
type Layer struct {
	ptr uintptr
}

// GetOutput returns the output tensor at the given index.
func (l *Layer) GetOutput(index int) *Tensor {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.layerGetOutput, l.ptr, uintptr(index))
	if ptr == 0 {
		return nil
	}
	return &Tensor{ptr: ptr}
}

// SetName sets the layer name for debugging.
func (l *Layer) SetName(name string) {
	lib, err := getTrtLib()
	if err != nil {
		return
	}
	cname := cstring(name)
	cuda.Ccall(lib.layerSetName, l.ptr, uintptr(unsafe.Pointer(&cname[0])))
}

// AddInput adds a network input tensor.
func (n *NetworkDefinition) AddInput(name string, dtype DataType, dims []int32) *Tensor {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	cname := cstring(name)
	ptr := cuda.Ccall(lib.networkAddInput, n.ptr,
		uintptr(unsafe.Pointer(&cname[0])),
		uintptr(dtype),
		uintptr(len(dims)),
		uintptr(unsafe.Pointer(&dims[0])))
	if ptr == 0 {
		return nil
	}
	return &Tensor{ptr: ptr}
}

// MarkOutput marks a tensor as a network output.
func (n *NetworkDefinition) MarkOutput(tensor *Tensor) {
	lib, err := getTrtLib()
	if err != nil {
		return
	}
	cuda.Ccall(lib.networkMarkOutput, n.ptr, tensor.ptr)
}

// AddActivation adds an activation layer.
func (n *NetworkDefinition) AddActivation(input *Tensor, actType ActivationType) *Layer {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.networkAddActivation, n.ptr, input.ptr, uintptr(actType))
	if ptr == 0 {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddElementWise adds an elementwise operation layer.
func (n *NetworkDefinition) AddElementWise(input1, input2 *Tensor, op ElementWiseOp) *Layer {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.networkAddElementwise, n.ptr, input1.ptr, input2.ptr, uintptr(op))
	if ptr == 0 {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddMatrixMultiply adds a matrix multiplication layer.
func (n *NetworkDefinition) AddMatrixMultiply(input0 *Tensor, op0 MatrixOp,
	input1 *Tensor, op1 MatrixOp) *Layer {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.networkAddMatrixMultiply, n.ptr,
		input0.ptr, uintptr(op0), input1.ptr, uintptr(op1))
	if ptr == 0 {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddSoftMax adds a softmax layer with the given axis.
func (n *NetworkDefinition) AddSoftMax(input *Tensor, axis int) *Layer {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.networkAddSoftmax, n.ptr, input.ptr, uintptr(axis))
	if ptr == 0 {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddReduce adds a reduce layer.
func (n *NetworkDefinition) AddReduce(input *Tensor, op ReduceOp, reduceAxes uint32, keepDims bool) *Layer {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	kd := uintptr(0)
	if keepDims {
		kd = 1
	}
	ptr := cuda.Ccall(lib.networkAddReduce, n.ptr, input.ptr, uintptr(op), uintptr(reduceAxes), kd)
	if ptr == 0 {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddConstant adds a constant tensor layer.
func (n *NetworkDefinition) AddConstant(dims []int32, dtype DataType, weights unsafe.Pointer, count int64) *Layer {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.networkAddConstant, n.ptr,
		uintptr(len(dims)),
		uintptr(unsafe.Pointer(&dims[0])),
		uintptr(dtype),
		uintptr(weights),
		uintptr(count))
	if ptr == 0 {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddShuffle adds a shuffle (reshape/transpose) layer.
func (n *NetworkDefinition) AddShuffle(input *Tensor) *Layer {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.networkAddShuffle, n.ptr, input.ptr)
	if ptr == 0 {
		return nil
	}
	return &Layer{ptr: ptr}
}

// AddConvolutionNd adds an N-dimensional convolution layer.
func (n *NetworkDefinition) AddConvolutionNd(input *Tensor, nbOutputMaps int,
	kernelSize []int32, kernelWeights unsafe.Pointer, kernelCount int64,
	biasWeights unsafe.Pointer, biasCount int64) *Layer {
	lib, err := getTrtLib()
	if err != nil {
		return nil
	}
	ptr := cuda.Ccall(lib.networkAddConvolutionNd, n.ptr, input.ptr,
		uintptr(nbOutputMaps),
		uintptr(len(kernelSize)),
		uintptr(unsafe.Pointer(&kernelSize[0])),
		uintptr(kernelWeights),
		uintptr(kernelCount),
		uintptr(biasWeights),
		uintptr(biasCount))
	if ptr == 0 {
		return nil
	}
	return &Layer{ptr: ptr}
}

// ShuffleSetReshapeDims sets reshape dimensions on a shuffle layer.
func ShuffleSetReshapeDims(layer *Layer, dims []int32) {
	lib, err := getTrtLib()
	if err != nil {
		return
	}
	cuda.Ccall(lib.shuffleSetReshapeDims, layer.ptr,
		uintptr(len(dims)),
		uintptr(unsafe.Pointer(&dims[0])))
}

// ShuffleSetFirstTranspose sets the first transpose permutation on a shuffle layer.
func ShuffleSetFirstTranspose(layer *Layer, perm []int32) {
	lib, err := getTrtLib()
	if err != nil {
		return
	}
	cuda.Ccall(lib.shuffleSetFirstTranspose, layer.ptr,
		uintptr(len(perm)),
		uintptr(unsafe.Pointer(&perm[0])))
}

// Runtime wraps a TensorRT IRuntime for deserializing engines.
type Runtime struct {
	ptr uintptr
}

// CreateRuntime creates a new TensorRT runtime.
func CreateRuntime(logger *Logger) (*Runtime, error) {
	lib, err := getTrtLib()
	if err != nil {
		return nil, err
	}
	ptr := cuda.Ccall(lib.createRuntime, logger.ptr)
	if ptr == 0 {
		return nil, fmt.Errorf("tensorrt: failed to create runtime")
	}
	return &Runtime{ptr: ptr}, nil
}

// Destroy releases the runtime.
func (r *Runtime) Destroy() {
	if r.ptr != 0 {
		lib, err := getTrtLib()
		if err != nil {
			return
		}
		cuda.Ccall(lib.destroyRuntime, r.ptr)
		r.ptr = 0
	}
}

// DeserializeEngine loads a serialized engine.
func (r *Runtime) DeserializeEngine(data []byte) (*Engine, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("tensorrt: empty serialized engine data")
	}
	lib, err := getTrtLib()
	if err != nil {
		return nil, err
	}
	ptr := cuda.Ccall(lib.deserializeEngine, r.ptr,
		uintptr(unsafe.Pointer(&data[0])),
		uintptr(len(data)))
	if ptr == 0 {
		return nil, fmt.Errorf("tensorrt: failed to deserialize engine")
	}
	return &Engine{ptr: ptr}, nil
}

// Engine wraps a TensorRT ICudaEngine.
type Engine struct {
	ptr uintptr
}

// Destroy releases the engine.
func (e *Engine) Destroy() {
	if e.ptr != 0 {
		lib, err := getTrtLib()
		if err != nil {
			return
		}
		cuda.Ccall(lib.destroyEngine, e.ptr)
		e.ptr = 0
	}
}

// NumIOTensors returns the number of input/output tensors.
func (e *Engine) NumIOTensors() int {
	lib, err := getTrtLib()
	if err != nil {
		return 0
	}
	return int(cuda.Ccall(lib.engineNumIOTensors, e.ptr))
}

// GetIOTensorName returns the name of the I/O tensor at the given index.
func (e *Engine) GetIOTensorName(index int) string {
	lib, err := getTrtLib()
	if err != nil {
		return ""
	}
	ptr := cuda.Ccall(lib.engineGetIOTensorName, e.ptr, uintptr(index))
	return goStringFromCPtr(ptr)
}

// CreateExecutionContext creates an execution context for this engine.
func (e *Engine) CreateExecutionContext() (*ExecutionContext, error) {
	lib, err := getTrtLib()
	if err != nil {
		return nil, err
	}
	ptr := cuda.Ccall(lib.createExecutionContext, e.ptr)
	if ptr == 0 {
		return nil, fmt.Errorf("tensorrt: failed to create execution context")
	}
	return &ExecutionContext{ptr: ptr}, nil
}

// ExecutionContext wraps a TensorRT IExecutionContext.
type ExecutionContext struct {
	ptr uintptr
}

// Destroy releases the execution context.
func (c *ExecutionContext) Destroy() {
	if c.ptr != 0 {
		lib, err := getTrtLib()
		if err != nil {
			return
		}
		cuda.Ccall(lib.destroyExecutionContext, c.ptr)
		c.ptr = 0
	}
}

// SetTensorAddress binds a device pointer to a named tensor.
func (c *ExecutionContext) SetTensorAddress(name string, data unsafe.Pointer) error {
	lib, err := getTrtLib()
	if err != nil {
		return err
	}
	cname := cstring(name)
	ret := cuda.Ccall(lib.contextSetTensorAddress, c.ptr,
		uintptr(unsafe.Pointer(&cname[0])),
		uintptr(data))
	if ret == 0 {
		return fmt.Errorf("tensorrt: failed to set tensor address for %q", name)
	}
	return nil
}

// EnqueueV3 enqueues inference on the given CUDA stream.
func (c *ExecutionContext) EnqueueV3(stream unsafe.Pointer) error {
	lib, err := getTrtLib()
	if err != nil {
		return err
	}
	ret := cuda.Ccall(lib.contextEnqueueV3, c.ptr, uintptr(stream))
	if ret == 0 {
		return fmt.Errorf("tensorrt: enqueueV3 failed")
	}
	return nil
}

// SetInputShape sets the input shape for a named tensor on the execution context.
// Required for dynamic shapes before calling EnqueueV3.
func (c *ExecutionContext) SetInputShape(name string, dims []int32) error {
	lib, err := getTrtLib()
	if err != nil {
		return err
	}
	cname := cstring(name)
	ret := cuda.Ccall(lib.contextSetInputShape, c.ptr,
		uintptr(unsafe.Pointer(&cname[0])),
		uintptr(len(dims)),
		uintptr(unsafe.Pointer(&dims[0])))
	if ret == 0 {
		return fmt.Errorf("tensorrt: failed to set input shape for %q", name)
	}
	return nil
}

// SetOptimizationProfile sets the active optimization profile on the context.
func (c *ExecutionContext) SetOptimizationProfile(index int) error {
	lib, err := getTrtLib()
	if err != nil {
		return err
	}
	ret := cuda.Ccall(lib.contextSetOptProfile, c.ptr, uintptr(index))
	if ret == 0 {
		return fmt.Errorf("tensorrt: failed to set optimization profile %d", index)
	}
	return nil
}

// OptimizationProfile wraps a TensorRT IOptimizationProfile.
type OptimizationProfile struct {
	ptr uintptr
}

// CreateOptimizationProfile creates a new optimization profile from the builder.
func (b *Builder) CreateOptimizationProfile() (*OptimizationProfile, error) {
	lib, err := getTrtLib()
	if err != nil {
		return nil, err
	}
	ptr := cuda.Ccall(lib.createOptimizationProfile, b.ptr)
	if ptr == 0 {
		return nil, fmt.Errorf("tensorrt: failed to create optimization profile")
	}
	return &OptimizationProfile{ptr: ptr}, nil
}

// SetDimensions sets the min/opt/max dimensions for a named input tensor.
func (p *OptimizationProfile) SetDimensions(inputName string, minDims, optDims, maxDims []int32) error {
	if len(minDims) != len(optDims) || len(minDims) != len(maxDims) {
		return fmt.Errorf("tensorrt: min/opt/max dims must have equal length")
	}
	lib, err := getTrtLib()
	if err != nil {
		return err
	}
	cname := cstring(inputName)
	ret := cuda.Ccall(lib.profileSetDimensions, p.ptr,
		uintptr(unsafe.Pointer(&cname[0])),
		uintptr(len(minDims)),
		uintptr(unsafe.Pointer(&minDims[0])),
		uintptr(unsafe.Pointer(&optDims[0])),
		uintptr(unsafe.Pointer(&maxDims[0])))
	if ret == 0 {
		return fmt.Errorf("tensorrt: failed to set dimensions for %q", inputName)
	}
	return nil
}

// AddToConfig adds this optimization profile to a builder config.
// Returns the profile index.
func (p *OptimizationProfile) AddToConfig(config *BuilderConfig) (int, error) {
	lib, err := getTrtLib()
	if err != nil {
		return -1, err
	}
	idx := int(cuda.Ccall(lib.configAddOptProfile, config.ptr, p.ptr))
	if idx < 0 {
		return -1, fmt.Errorf("tensorrt: failed to add optimization profile")
	}
	return idx, nil
}
