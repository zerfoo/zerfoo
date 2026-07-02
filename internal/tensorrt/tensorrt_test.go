package tensorrt

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func TestCreateDestroyLogger(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	if l == nil {
		t.Fatal("CreateLogger returned nil")
	}
	l.Destroy()
}

func TestCreateDestroyBuilder(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	b.Destroy()
}

func TestCreateDestroyNetwork(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	n, err := b.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}

	if n.NumInputs() != 0 {
		t.Errorf("NumInputs = %d, want 0", n.NumInputs())
	}
	if n.NumOutputs() != 0 {
		t.Errorf("NumOutputs = %d, want 0", n.NumOutputs())
	}
	if n.NumLayers() != 0 {
		t.Errorf("NumLayers = %d, want 0", n.NumLayers())
	}

	n.Destroy()
}

func TestCreateDestroyBuilderConfig(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	cfg, err := b.CreateBuilderConfig()
	if err != nil {
		t.Fatalf("CreateBuilderConfig: %v", err)
	}

	cfg.SetMemoryPoolLimit(1 << 20) // 1 MB
	cfg.Destroy()
}

func TestCreateDestroyRuntime(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	rt, err := CreateRuntime(l)
	if err != nil {
		t.Fatalf("CreateRuntime: %v", err)
	}
	rt.Destroy()
}

func TestNetworkAddInput(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	n, err := b.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer n.Destroy()

	input := n.AddInput("input", Float32, []int32{1, 4})
	if input == nil {
		t.Fatal("AddInput returned nil")
	}

	if n.NumInputs() != 1 {
		t.Errorf("NumInputs = %d, want 1", n.NumInputs())
	}
}

func TestNetworkAddActivation(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	n, err := b.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer n.Destroy()

	input := n.AddInput("input", Float32, []int32{1, 4})
	if input == nil {
		t.Fatal("AddInput returned nil")
	}

	relu := n.AddActivation(input, ActivationReLU)
	if relu == nil {
		t.Fatal("AddActivation returned nil")
	}

	out := relu.GetOutput(0)
	if out == nil {
		t.Fatal("GetOutput returned nil")
	}

	n.MarkOutput(out)

	if n.NumLayers() != 1 {
		t.Errorf("NumLayers = %d, want 1", n.NumLayers())
	}
	if n.NumOutputs() != 1 {
		t.Errorf("NumOutputs = %d, want 1", n.NumOutputs())
	}
}

// TestBuildAndRunReLUNetwork builds a trivial Input -> ReLU -> Output network,
// serializes it, deserializes it, and runs inference to verify correctness.
func TestBuildAndRunReLUNetwork(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	logger := CreateLogger(SeverityWarning)
	defer logger.Destroy()

	builder, err := CreateBuilder(logger)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer builder.Destroy()

	network, err := builder.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer network.Destroy()

	// Build: input(1,4) -> ReLU -> output
	input := network.AddInput("input", Float32, []int32{1, 4})
	if input == nil {
		t.Fatal("AddInput returned nil")
	}

	relu := network.AddActivation(input, ActivationReLU)
	if relu == nil {
		t.Fatal("AddActivation returned nil")
	}
	relu.SetName("relu_0")

	out := relu.GetOutput(0)
	network.MarkOutput(out)

	// Configure builder
	config, err := builder.CreateBuilderConfig()
	if err != nil {
		t.Fatalf("CreateBuilderConfig: %v", err)
	}
	defer config.Destroy()
	config.SetMemoryPoolLimit(1 << 20) // 1 MB workspace

	// Build serialized engine
	serialized, err := builder.BuildSerializedNetwork(network, config)
	if err != nil {
		t.Fatalf("BuildSerializedNetwork: %v", err)
	}
	if len(serialized) == 0 {
		t.Fatal("serialized engine is empty")
	}

	// Deserialize
	runtime, err := CreateRuntime(logger)
	if err != nil {
		t.Fatalf("CreateRuntime: %v", err)
	}
	defer runtime.Destroy()

	engine, err := runtime.DeserializeEngine(serialized)
	if err != nil {
		t.Fatalf("DeserializeEngine: %v", err)
	}
	defer engine.Destroy()

	if engine.NumIOTensors() != 2 {
		t.Fatalf("NumIOTensors = %d, want 2", engine.NumIOTensors())
	}

	// Create execution context
	ctx, err := engine.CreateExecutionContext()
	if err != nil {
		t.Fatalf("CreateExecutionContext: %v", err)
	}
	defer ctx.Destroy()

	// Allocate device memory
	inputData := []float32{-2, -1, 1, 3}
	byteSize := len(inputData) * 4

	inputDev, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc input: %v", err)
	}
	defer func() { _ = cuda.Free(inputDev) }()

	outputDev, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc output: %v", err)
	}
	defer func() { _ = cuda.Free(outputDev) }()

	// Copy input to device
	if err := cuda.Memcpy(inputDev, unsafe.Pointer(&inputData[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	// Bind tensors
	if err := ctx.SetTensorAddress("input", inputDev); err != nil {
		t.Fatalf("SetTensorAddress(input): %v", err)
	}
	// TensorRT auto-names the output based on the layer
	outputName := engine.GetIOTensorName(1)
	if err := ctx.SetTensorAddress(outputName, outputDev); err != nil {
		t.Fatalf("SetTensorAddress(%s): %v", outputName, err)
	}

	// Run inference on default stream
	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := ctx.EnqueueV3(stream.Ptr()); err != nil {
		t.Fatalf("EnqueueV3: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	// Copy output back
	result := make([]float32, len(inputData))
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), outputDev, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	// Verify ReLU: [-2, -1, 1, 3] -> [0, 0, 1, 3]
	expected := []float32{0, 0, 1, 3}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("output[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

// TestBuildAndRunMatMulReLUNetwork builds a MatMul -> ReLU network and verifies output.
func TestBuildAndRunMatMulReLUNetwork(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	logger := CreateLogger(SeverityWarning)
	defer logger.Destroy()

	builder, err := CreateBuilder(logger)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer builder.Destroy()

	network, err := builder.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer network.Destroy()

	// Network: input(1,2) x weights(2,3) -> ReLU -> output(1,3)
	input := network.AddInput("input", Float32, []int32{1, 2})
	if input == nil {
		t.Fatal("AddInput returned nil")
	}

	// Weights as a constant: 2x3 matrix [[1, -1, 2], [0, 3, -2]]
	weights := []float32{1, -1, 2, 0, 3, -2}
	wLayer := network.AddConstant([]int32{2, 3}, Float32,
		unsafe.Pointer(&weights[0]), int64(len(weights)))
	if wLayer == nil {
		t.Fatal("AddConstant returned nil")
	}
	wTensor := wLayer.GetOutput(0)

	// MatMul: (1,2) x (2,3) = (1,3)
	matmul := network.AddMatrixMultiply(input, MatrixOpNone, wTensor, MatrixOpNone)
	if matmul == nil {
		t.Fatal("AddMatrixMultiply returned nil")
	}
	matmulOut := matmul.GetOutput(0)

	// ReLU
	relu := network.AddActivation(matmulOut, ActivationReLU)
	if relu == nil {
		t.Fatal("AddActivation returned nil")
	}
	out := relu.GetOutput(0)
	network.MarkOutput(out)

	// Build
	config, err := builder.CreateBuilderConfig()
	if err != nil {
		t.Fatalf("CreateBuilderConfig: %v", err)
	}
	defer config.Destroy()
	config.SetMemoryPoolLimit(1 << 20)

	serialized, err := builder.BuildSerializedNetwork(network, config)
	if err != nil {
		t.Fatalf("BuildSerializedNetwork: %v", err)
	}

	runtime, err := CreateRuntime(logger)
	if err != nil {
		t.Fatalf("CreateRuntime: %v", err)
	}
	defer runtime.Destroy()

	engine, err := runtime.DeserializeEngine(serialized)
	if err != nil {
		t.Fatalf("DeserializeEngine: %v", err)
	}
	defer engine.Destroy()

	ctx, err := engine.CreateExecutionContext()
	if err != nil {
		t.Fatalf("CreateExecutionContext: %v", err)
	}
	defer ctx.Destroy()

	// Input: [1, 2] -> MatMul -> [1*1+2*0, 1*(-1)+2*3, 1*2+2*(-2)] = [1, 5, -2] -> ReLU -> [1, 5, 0]
	inputData := []float32{1, 2}
	inputBytes := len(inputData) * 4
	outputBytes := 3 * 4

	inputDev, err := cuda.Malloc(inputBytes)
	if err != nil {
		t.Fatalf("Malloc input: %v", err)
	}
	defer func() { _ = cuda.Free(inputDev) }()

	outputDev, err := cuda.Malloc(outputBytes)
	if err != nil {
		t.Fatalf("Malloc output: %v", err)
	}
	defer func() { _ = cuda.Free(outputDev) }()

	if err := cuda.Memcpy(inputDev, unsafe.Pointer(&inputData[0]), inputBytes, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	if err := ctx.SetTensorAddress("input", inputDev); err != nil {
		t.Fatalf("SetTensorAddress(input): %v", err)
	}
	outputName := engine.GetIOTensorName(1)
	if err := ctx.SetTensorAddress(outputName, outputDev); err != nil {
		t.Fatalf("SetTensorAddress(%s): %v", outputName, err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := ctx.EnqueueV3(stream.Ptr()); err != nil {
		t.Fatalf("EnqueueV3: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	result := make([]float32, 3)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), outputDev, outputBytes, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	expected := []float32{1, 5, 0}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("output[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestDeserializeEngineEmptyData(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	rt, err := CreateRuntime(l)
	if err != nil {
		t.Fatalf("CreateRuntime: %v", err)
	}
	defer rt.Destroy()

	_, err = rt.DeserializeEngine(nil)
	if err == nil {
		t.Fatal("expected error for empty data")
	}
}

func TestNetworkAddElementWise(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	n, err := b.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer n.Destroy()

	a := n.AddInput("a", Float32, []int32{1, 4})
	bInput := n.AddInput("b", Float32, []int32{1, 4})

	add := n.AddElementWise(a, bInput, ElementWiseSum)
	if add == nil {
		t.Fatal("AddElementWise returned nil")
	}

	out := add.GetOutput(0)
	n.MarkOutput(out)

	if n.NumLayers() != 1 {
		t.Errorf("NumLayers = %d, want 1", n.NumLayers())
	}
}

func TestNetworkAddSoftMax(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	n, err := b.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer n.Destroy()

	input := n.AddInput("input", Float32, []int32{1, 4})
	sm := n.AddSoftMax(input, 1) // softmax over dim 1
	if sm == nil {
		t.Fatal("AddSoftMax returned nil")
	}

	out := sm.GetOutput(0)
	n.MarkOutput(out)

	if n.NumLayers() != 1 {
		t.Errorf("NumLayers = %d, want 1", n.NumLayers())
	}
}

func TestNetworkAddReduce(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	n, err := b.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer n.Destroy()

	input := n.AddInput("input", Float32, []int32{2, 3})
	// Reduce sum over axis 1 (bitmask: 1<<1 = 2)
	reduce := n.AddReduce(input, ReduceSum, 2, true)
	if reduce == nil {
		t.Fatal("AddReduce returned nil")
	}

	out := reduce.GetOutput(0)
	n.MarkOutput(out)

	if n.NumLayers() != 1 {
		t.Errorf("NumLayers = %d, want 1", n.NumLayers())
	}
}

func TestNetworkAddShuffle(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	n, err := b.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer n.Destroy()

	input := n.AddInput("input", Float32, []int32{2, 3})
	shuffle := n.AddShuffle(input)
	if shuffle == nil {
		t.Fatal("AddShuffle returned nil")
	}
	ShuffleSetReshapeDims(shuffle, []int32{3, 2})

	out := shuffle.GetOutput(0)
	n.MarkOutput(out)

	if n.NumLayers() != 1 {
		t.Errorf("NumLayers = %d, want 1", n.NumLayers())
	}
}

func TestBuilderConfigSetFlagFP16(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}
	l := CreateLogger(SeverityWarning)
	defer l.Destroy()

	b, err := CreateBuilder(l)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer b.Destroy()

	cfg, err := b.CreateBuilderConfig()
	if err != nil {
		t.Fatalf("CreateBuilderConfig: %v", err)
	}
	defer cfg.Destroy()

	// Should not panic
	cfg.SetFlag(FlagFP16)
}
