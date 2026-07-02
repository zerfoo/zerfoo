package cudnn

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func skipIfNoAvailable(t *testing.T) {
	t.Helper()
	if !Available() {
		t.Skip("cuDNN not available")
	}
}

func TestCreateDestroyHandle(t *testing.T) {
	skipIfNoAvailable(t)
	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	if err := h.Destroy(); err != nil {
		t.Fatalf("Destroy: %v", err)
	}
}

func TestHandleSetStream(t *testing.T) {
	skipIfNoAvailable(t)
	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer h.Destroy()

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer stream.Destroy()

	if err := h.SetStream(stream.Ptr()); err != nil {
		t.Fatalf("SetStream: %v", err)
	}
}

func TestCreateDestroyTensorDescriptor(t *testing.T) {
	skipIfNoAvailable(t)
	d, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor: %v", err)
	}
	if err := d.Set4d(NCHW, Float32, 1, 3, 224, 224); err != nil {
		t.Fatalf("Set4d: %v", err)
	}
	if err := d.Destroy(); err != nil {
		t.Fatalf("Destroy: %v", err)
	}
}

func TestTensorDescriptorSetNd(t *testing.T) {
	skipIfNoAvailable(t)
	d, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor: %v", err)
	}
	defer d.Destroy()

	dims := []int{2, 3, 4, 5}
	strides := []int{60, 20, 5, 1}
	if err := d.SetNd(Float32, dims, strides); err != nil {
		t.Fatalf("SetNd: %v", err)
	}
}

func TestTensorDescriptorSetNdMismatch(t *testing.T) {
	skipIfNoAvailable(t)
	d, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor: %v", err)
	}
	defer d.Destroy()

	err = d.SetNd(Float32, []int{1, 2}, []int{1})
	if err == nil {
		t.Fatal("expected error for mismatched dims/strides lengths")
	}
}

func TestCreateDestroyFilterDescriptor(t *testing.T) {
	skipIfNoAvailable(t)
	d, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatalf("CreateFilterDescriptor: %v", err)
	}
	if err := d.Set4d(Float32, NCHW, 64, 3, 3, 3); err != nil {
		t.Fatalf("Set4d: %v", err)
	}
	if err := d.Destroy(); err != nil {
		t.Fatalf("Destroy: %v", err)
	}
}

func TestCreateDestroyConvolutionDescriptor(t *testing.T) {
	skipIfNoAvailable(t)
	d, err := CreateConvolutionDescriptor()
	if err != nil {
		t.Fatalf("CreateConvolutionDescriptor: %v", err)
	}
	if err := d.Set2d(1, 1, 1, 1, 1, 1, CrossCorrelation, Float32); err != nil {
		t.Fatalf("Set2d: %v", err)
	}
	if err := d.Destroy(); err != nil {
		t.Fatalf("Destroy: %v", err)
	}
}

func TestCreateDestroyActivationDescriptor(t *testing.T) {
	skipIfNoAvailable(t)
	tests := []struct {
		name string
		mode ActivationMode
	}{
		{"ReLU", ActivationReLU},
		{"Sigmoid", ActivationSigmoid},
		{"Tanh", ActivationTanh},
		{"ClippedReLU", ActivationClippedReLU},
		{"ELU", ActivationELU},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d, err := CreateActivationDescriptor()
			if err != nil {
				t.Fatalf("CreateActivationDescriptor: %v", err)
			}
			if err := d.Set(tc.mode, NotPropagateNan, 0.0); err != nil {
				t.Fatalf("Set(%s): %v", tc.name, err)
			}
			if err := d.Destroy(); err != nil {
				t.Fatalf("Destroy: %v", err)
			}
		})
	}
}

func TestCreateDestroyPoolingDescriptor(t *testing.T) {
	skipIfNoAvailable(t)
	tests := []struct {
		name string
		mode PoolingMode
	}{
		{"Max", PoolingMax},
		{"AvgIncPad", PoolingAverageCountIncludePad},
		{"AvgExcPad", PoolingAverageCountExcludePad},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d, err := CreatePoolingDescriptor()
			if err != nil {
				t.Fatalf("CreatePoolingDescriptor: %v", err)
			}
			if err := d.Set2d(tc.mode, NotPropagateNan, 2, 2, 0, 0, 2, 2); err != nil {
				t.Fatalf("Set2d(%s): %v", tc.name, err)
			}
			if err := d.Destroy(); err != nil {
				t.Fatalf("Destroy: %v", err)
			}
		})
	}
}

// TestActivationForwardReLU tests an in-place ReLU activation on a small buffer.
func TestActivationForwardReLU(t *testing.T) {
	skipIfNoAvailable(t)
	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer h.Destroy()

	// Input: [-1, 0, 1, 2] -> expected ReLU output: [0, 0, 1, 2]
	input := []float32{-1, 0, 1, 2}
	n, c, height, w := 1, 1, 1, 4

	xDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(x): %v", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(NCHW, Float32, n, c, height, w); err != nil {
		t.Fatalf("Set4d(x): %v", err)
	}

	yDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(y): %v", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(NCHW, Float32, n, c, height, w); err != nil {
		t.Fatalf("Set4d(y): %v", err)
	}

	byteSize := len(input) * 4
	xDev, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc(x): %v", err)
	}
	defer func() { _ = cuda.Free(xDev) }()

	yDev, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc(y): %v", err)
	}
	defer func() { _ = cuda.Free(yDev) }()

	if err := cuda.Memcpy(xDev, unsafe.Pointer(&input[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	actDesc, err := CreateActivationDescriptor()
	if err != nil {
		t.Fatalf("CreateActivationDescriptor: %v", err)
	}
	defer actDesc.Destroy()
	if err := actDesc.Set(ActivationReLU, NotPropagateNan, 0.0); err != nil {
		t.Fatalf("Set(ReLU): %v", err)
	}

	if err := h.ActivationForward(actDesc, 1.0, xDesc, xDev, 0.0, yDesc, yDev); err != nil {
		t.Fatalf("ActivationForward: %v", err)
	}

	output := make([]float32, len(input))
	if err := cuda.Memcpy(unsafe.Pointer(&output[0]), yDev, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	expected := []float32{0, 0, 1, 2}
	for i, v := range output {
		if v != expected[i] {
			t.Errorf("output[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

// TestPoolingForwardMax tests a 2x2 max pooling on a 4x4 input.
func TestPoolingForwardMax(t *testing.T) {
	skipIfNoAvailable(t)
	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer h.Destroy()

	// 1x1x4x4 input -> 1x1x2x2 output with 2x2 max pool, stride 2
	input := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	n, c, inH, inW := 1, 1, 4, 4
	outH, outW := 2, 2

	xDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(x): %v", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(NCHW, Float32, n, c, inH, inW); err != nil {
		t.Fatalf("Set4d(x): %v", err)
	}

	yDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(y): %v", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(NCHW, Float32, n, c, outH, outW); err != nil {
		t.Fatalf("Set4d(y): %v", err)
	}

	poolDesc, err := CreatePoolingDescriptor()
	if err != nil {
		t.Fatalf("CreatePoolingDescriptor: %v", err)
	}
	defer poolDesc.Destroy()
	if err := poolDesc.Set2d(PoolingMax, NotPropagateNan, 2, 2, 0, 0, 2, 2); err != nil {
		t.Fatalf("Set2d: %v", err)
	}

	xBytes := len(input) * 4
	yBytes := outH * outW * 4

	xDev, err := cuda.Malloc(xBytes)
	if err != nil {
		t.Fatalf("Malloc(x): %v", err)
	}
	defer func() { _ = cuda.Free(xDev) }()

	yDev, err := cuda.Malloc(yBytes)
	if err != nil {
		t.Fatalf("Malloc(y): %v", err)
	}
	defer func() { _ = cuda.Free(yDev) }()

	if err := cuda.Memcpy(xDev, unsafe.Pointer(&input[0]), xBytes, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	if err := h.PoolingForward(poolDesc, 1.0, xDesc, xDev, 0.0, yDesc, yDev); err != nil {
		t.Fatalf("PoolingForward: %v", err)
	}

	output := make([]float32, outH*outW)
	if err := cuda.Memcpy(unsafe.Pointer(&output[0]), yDev, yBytes, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	// 2x2 max pool on 4x4: top-left max=6, top-right max=8, bottom-left max=14, bottom-right max=16
	expected := []float32{6, 8, 14, 16}
	for i, v := range output {
		if v != expected[i] {
			t.Errorf("output[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

// TestSoftmaxForward tests softmax on a small input.
func TestSoftmaxForward(t *testing.T) {
	skipIfNoAvailable(t)
	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer h.Destroy()

	// Softmax over channel dim: [1, 2, 3] -> [0.0900, 0.2447, 0.6652] approximately
	input := []float32{1, 2, 3}
	n, c, height, w := 1, 3, 1, 1

	xDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(x): %v", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(NCHW, Float32, n, c, height, w); err != nil {
		t.Fatalf("Set4d(x): %v", err)
	}

	yDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(y): %v", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(NCHW, Float32, n, c, height, w); err != nil {
		t.Fatalf("Set4d(y): %v", err)
	}

	byteSize := len(input) * 4
	xDev, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc(x): %v", err)
	}
	defer func() { _ = cuda.Free(xDev) }()

	yDev, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc(y): %v", err)
	}
	defer func() { _ = cuda.Free(yDev) }()

	if err := cuda.Memcpy(xDev, unsafe.Pointer(&input[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	if err := h.SoftmaxForward(SoftmaxAccurate, SoftmaxModeChannel, 1.0, xDesc, xDev, 0.0, yDesc, yDev); err != nil {
		t.Fatalf("SoftmaxForward: %v", err)
	}

	output := make([]float32, len(input))
	if err := cuda.Memcpy(unsafe.Pointer(&output[0]), yDev, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	// Verify softmax properties: all positive, sums to 1, monotonically increasing
	sum := float32(0)
	for i, v := range output {
		if v <= 0 {
			t.Errorf("output[%d] = %f, want > 0", i, v)
		}
		sum += v
	}
	if sum < 0.999 || sum > 1.001 {
		t.Errorf("softmax sum = %f, want ~1.0", sum)
	}
	if output[0] >= output[1] || output[1] >= output[2] {
		t.Errorf("softmax not monotonically increasing: %v", output)
	}
}
