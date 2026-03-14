package cudnn

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

const maxRelErr = 1e-4

// relError returns the relative error between got and want.
// Returns 0 when both are zero.
func relError(got, want float32) float64 {
	if got == want {
		return 0
	}
	denom := float64(math.Abs(float64(want)))
	if denom == 0 {
		denom = 1
	}
	return math.Abs(float64(got)-float64(want)) / denom
}

// checkParity verifies that each element in got matches want within maxRelErr.
func checkParity(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		re := relError(got[i], want[i])
		if re > maxRelErr {
			t.Errorf("%s[%d]: got %g, want %g (rel error %g > %g)", name, i, got[i], want[i], re, maxRelErr)
		}
	}
}

// allocDevFloat32 allocates device memory and copies host data to it.
func allocDevFloat32(t *testing.T, data []float32) unsafe.Pointer {
	t.Helper()
	byteSize := len(data) * 4
	dev, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}
	if err := cuda.Memcpy(dev, unsafe.Pointer(&data[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		_ = cuda.Free(dev)
		t.Fatalf("Memcpy H2D: %v", err)
	}
	return dev
}

// allocDevZero allocates zero-initialized device memory.
func allocDevZero(t *testing.T, count int) unsafe.Pointer {
	t.Helper()
	byteSize := count * 4
	dev, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}
	zeros := make([]float32, count)
	if err := cuda.Memcpy(dev, unsafe.Pointer(&zeros[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		_ = cuda.Free(dev)
		t.Fatalf("Memcpy H2D: %v", err)
	}
	return dev
}

// readDevFloat32 copies device memory to host.
func readDevFloat32(t *testing.T, dev unsafe.Pointer, count int) []float32 {
	t.Helper()
	out := make([]float32, count)
	byteSize := count * 4
	if err := cuda.Memcpy(unsafe.Pointer(&out[0]), dev, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}
	return out
}

func TestActivationForwardParity(t *testing.T) {
	skipIfNoAvailable(t)

	tests := []struct {
		name   string
		mode   ActivationMode
		coef   float64
		input  []float32
		cpuFn  func(float32) float32
	}{
		{
			name:  "ReLU",
			mode:  ActivationReLU,
			coef:  0,
			input: []float32{-2.5, -0.1, 0, 0.5, 3.0, -1e-6},
			cpuFn: func(x float32) float32 {
				if x > 0 {
					return x
				}
				return 0
			},
		},
		{
			name:  "Sigmoid",
			mode:  ActivationSigmoid,
			coef:  0,
			input: []float32{-3, -1, 0, 1, 3, 10},
			cpuFn: func(x float32) float32 {
				return float32(1.0 / (1.0 + math.Exp(-float64(x))))
			},
		},
		{
			name:  "Tanh",
			mode:  ActivationTanh,
			coef:  0,
			input: []float32{-3, -1, 0, 0.5, 1, 3},
			cpuFn: func(x float32) float32 {
				return float32(math.Tanh(float64(x)))
			},
		},
		{
			name:  "ClippedReLU",
			mode:  ActivationClippedReLU,
			coef:  2.0,
			input: []float32{-1, 0, 0.5, 1.5, 2.0, 5.0},
			cpuFn: func(x float32) float32 {
				if x < 0 {
					return 0
				}
				if x > 2.0 {
					return 2.0
				}
				return x
			},
		},
		{
			name:  "ELU",
			mode:  ActivationELU,
			coef:  1.0,
			input: []float32{-3, -1, 0, 0.5, 2, 5},
			cpuFn: func(x float32) float32 {
				if x >= 0 {
					return x
				}
				return float32(math.Exp(float64(x)) - 1)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h, err := CreateHandle()
			if err != nil {
				t.Fatalf("CreateHandle: %v", err)
			}
			defer h.Destroy()

			n, c, height, w := 1, 1, 1, len(tc.input)
			count := len(tc.input)

			xDesc, err := CreateTensorDescriptor()
			if err != nil {
				t.Fatalf("CreateTensorDescriptor: %v", err)
			}
			defer xDesc.Destroy()
			if err := xDesc.Set4d(NCHW, Float32, n, c, height, w); err != nil {
				t.Fatalf("Set4d: %v", err)
			}

			yDesc, err := CreateTensorDescriptor()
			if err != nil {
				t.Fatalf("CreateTensorDescriptor: %v", err)
			}
			defer yDesc.Destroy()
			if err := yDesc.Set4d(NCHW, Float32, n, c, height, w); err != nil {
				t.Fatalf("Set4d: %v", err)
			}

			xDev := allocDevFloat32(t, tc.input)
			defer func() { _ = cuda.Free(xDev) }()
			yDev := allocDevZero(t, count)
			defer func() { _ = cuda.Free(yDev) }()

			actDesc, err := CreateActivationDescriptor()
			if err != nil {
				t.Fatalf("CreateActivationDescriptor: %v", err)
			}
			defer actDesc.Destroy()
			if err := actDesc.Set(tc.mode, NotPropagateNan, tc.coef); err != nil {
				t.Fatalf("Set: %v", err)
			}

			if err := h.ActivationForward(actDesc, 1.0, xDesc, xDev, 0.0, yDesc, yDev); err != nil {
				t.Fatalf("ActivationForward: %v", err)
			}

			got := readDevFloat32(t, yDev, count)
			want := make([]float32, count)
			for i, x := range tc.input {
				want[i] = tc.cpuFn(x)
			}
			checkParity(t, tc.name, got, want)
		})
	}
}

func TestPoolingForwardParity(t *testing.T) {
	skipIfNoAvailable(t)

	tests := []struct {
		name    string
		mode    PoolingMode
		inH     int
		inW     int
		winH    int
		winW    int
		padH    int
		padW    int
		strideH int
		strideW int
		input   []float32
	}{
		{
			name:    "Max2x2",
			mode:    PoolingMax,
			inH:     4,
			inW:     4,
			winH:    2,
			winW:    2,
			strideH: 2,
			strideW: 2,
			input: []float32{
				1, 3, 2, 4,
				5, 6, 8, 7,
				9, 11, 10, 12,
				13, 15, 14, 16,
			},
		},
		{
			name:    "AvgIncPad2x2",
			mode:    PoolingAverageCountIncludePad,
			inH:     4,
			inW:     4,
			winH:    2,
			winW:    2,
			strideH: 2,
			strideW: 2,
			input: []float32{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
				13, 14, 15, 16,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h, err := CreateHandle()
			if err != nil {
				t.Fatalf("CreateHandle: %v", err)
			}
			defer h.Destroy()

			outH := (tc.inH+2*tc.padH-tc.winH)/tc.strideH + 1
			outW := (tc.inW+2*tc.padW-tc.winW)/tc.strideW + 1
			n, c := 1, 1

			xDesc, err := CreateTensorDescriptor()
			if err != nil {
				t.Fatalf("CreateTensorDescriptor: %v", err)
			}
			defer xDesc.Destroy()
			if err := xDesc.Set4d(NCHW, Float32, n, c, tc.inH, tc.inW); err != nil {
				t.Fatalf("Set4d(x): %v", err)
			}

			yDesc, err := CreateTensorDescriptor()
			if err != nil {
				t.Fatalf("CreateTensorDescriptor: %v", err)
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
			if err := poolDesc.Set2d(tc.mode, NotPropagateNan, tc.winH, tc.winW, tc.padH, tc.padW, tc.strideH, tc.strideW); err != nil {
				t.Fatalf("Set2d: %v", err)
			}

			xDev := allocDevFloat32(t, tc.input)
			defer func() { _ = cuda.Free(xDev) }()
			outCount := outH * outW
			yDev := allocDevZero(t, outCount)
			defer func() { _ = cuda.Free(yDev) }()

			if err := h.PoolingForward(poolDesc, 1.0, xDesc, xDev, 0.0, yDesc, yDev); err != nil {
				t.Fatalf("PoolingForward: %v", err)
			}

			got := readDevFloat32(t, yDev, outCount)

			// CPU reference
			want := make([]float32, outCount)
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					hStart := oh*tc.strideH - tc.padH
					wStart := ow*tc.strideW - tc.padW
					var val float64
					if tc.mode == PoolingMax {
						val = -math.MaxFloat64
					}
					count := 0
					for wh := 0; wh < tc.winH; wh++ {
						for ww := 0; ww < tc.winW; ww++ {
							ih := hStart + wh
							iw := wStart + ww
							if ih < 0 || ih >= tc.inH || iw < 0 || iw >= tc.inW {
								if tc.mode == PoolingAverageCountIncludePad {
									count++
								}
								continue
							}
							v := float64(tc.input[ih*tc.inW+iw])
							if tc.mode == PoolingMax {
								if v > val {
									val = v
								}
							} else {
								val += v
								count++
							}
						}
					}
					if tc.mode != PoolingMax && count > 0 {
						val /= float64(count)
					}
					want[oh*outW+ow] = float32(val)
				}
			}
			checkParity(t, tc.name, got, want)
		})
	}
}

func TestSoftmaxForwardParity(t *testing.T) {
	skipIfNoAvailable(t)

	tests := []struct {
		name  string
		input []float32 // Interpreted as (1, C, 1, 1) -- softmax over C
	}{
		{
			name:  "Small",
			input: []float32{1, 2, 3},
		},
		{
			name:  "Negative",
			input: []float32{-1, 0, 1, 2},
		},
		{
			name:  "Large",
			input: []float32{100, 101, 102},
		},
		{
			name:  "Single",
			input: []float32{42},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h, err := CreateHandle()
			if err != nil {
				t.Fatalf("CreateHandle: %v", err)
			}
			defer h.Destroy()

			c := len(tc.input)
			n, height, w := 1, 1, 1

			xDesc, err := CreateTensorDescriptor()
			if err != nil {
				t.Fatalf("CreateTensorDescriptor: %v", err)
			}
			defer xDesc.Destroy()
			if err := xDesc.Set4d(NCHW, Float32, n, c, height, w); err != nil {
				t.Fatalf("Set4d: %v", err)
			}

			yDesc, err := CreateTensorDescriptor()
			if err != nil {
				t.Fatalf("CreateTensorDescriptor: %v", err)
			}
			defer yDesc.Destroy()
			if err := yDesc.Set4d(NCHW, Float32, n, c, height, w); err != nil {
				t.Fatalf("Set4d: %v", err)
			}

			xDev := allocDevFloat32(t, tc.input)
			defer func() { _ = cuda.Free(xDev) }()
			yDev := allocDevZero(t, c)
			defer func() { _ = cuda.Free(yDev) }()

			if err := h.SoftmaxForward(SoftmaxAccurate, SoftmaxModeChannel, 1.0, xDesc, xDev, 0.0, yDesc, yDev); err != nil {
				t.Fatalf("SoftmaxForward: %v", err)
			}

			got := readDevFloat32(t, yDev, c)

			// CPU reference: stable softmax
			maxVal := float64(tc.input[0])
			for _, v := range tc.input[1:] {
				if float64(v) > maxVal {
					maxVal = float64(v)
				}
			}
			want := make([]float32, c)
			sumExp := 0.0
			for _, v := range tc.input {
				sumExp += math.Exp(float64(v) - maxVal)
			}
			for i, v := range tc.input {
				want[i] = float32(math.Exp(float64(v)-maxVal) / sumExp)
			}
			checkParity(t, tc.name, got, want)
		})
	}
}

func TestConvolutionForwardParity(t *testing.T) {
	skipIfNoAvailable(t)

	// Simple 1x1x4x4 input, 1x1x3x3 filter, pad=1, stride=1 -> 1x1x4x4 output
	input := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	filter := []float32{
		0, 0, 0,
		0, 1, 0,
		0, 0, 0,
	}
	n, cIn, inH, inW := 1, 1, 4, 4
	cOut, fH, fW := 1, 3, 3
	padH, padW := 1, 1
	strideH, strideW := 1, 1
	outH := (inH+2*padH-fH)/strideH + 1
	outW := (inW+2*padW-fW)/strideW + 1
	outCount := n * cOut * outH * outW

	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer h.Destroy()

	xDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(x): %v", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(NCHW, Float32, n, cIn, inH, inW); err != nil {
		t.Fatalf("Set4d(x): %v", err)
	}

	yDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(y): %v", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(NCHW, Float32, n, cOut, outH, outW); err != nil {
		t.Fatalf("Set4d(y): %v", err)
	}

	wDesc, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatalf("CreateFilterDescriptor: %v", err)
	}
	defer wDesc.Destroy()
	if err := wDesc.Set4d(Float32, NCHW, cOut, cIn, fH, fW); err != nil {
		t.Fatalf("Set4d(w): %v", err)
	}

	convDesc, err := CreateConvolutionDescriptor()
	if err != nil {
		t.Fatalf("CreateConvolutionDescriptor: %v", err)
	}
	defer convDesc.Destroy()
	if err := convDesc.Set2d(padH, padW, strideH, strideW, 1, 1, CrossCorrelation, Float32); err != nil {
		t.Fatalf("Set2d: %v", err)
	}

	xDev := allocDevFloat32(t, input)
	defer func() { _ = cuda.Free(xDev) }()
	wDev := allocDevFloat32(t, filter)
	defer func() { _ = cuda.Free(wDev) }()
	yDev := allocDevZero(t, outCount)
	defer func() { _ = cuda.Free(yDev) }()

	wsSize, err := h.GetConvolutionForwardWorkspaceSize(xDesc, wDesc, convDesc, yDesc, ConvFwdAlgoImplicitGemm)
	if err != nil {
		t.Fatalf("GetConvolutionForwardWorkspaceSize: %v", err)
	}
	var wsDev unsafe.Pointer
	if wsSize > 0 {
		wsDev, err = cuda.Malloc(wsSize)
		if err != nil {
			t.Fatalf("Malloc workspace: %v", err)
		}
		defer func() { _ = cuda.Free(wsDev) }()
	}

	if err := h.ConvolutionForward(1.0, xDesc, xDev, wDesc, wDev, convDesc,
		ConvFwdAlgoImplicitGemm, wsDev, wsSize, 0.0, yDesc, yDev); err != nil {
		t.Fatalf("ConvolutionForward: %v", err)
	}

	got := readDevFloat32(t, yDev, outCount)

	// CPU reference: cross-correlation with identity kernel should reproduce input
	want := make([]float32, outCount)
	for onn := 0; onn < n; onn++ {
		for oc := 0; oc < cOut; oc++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					var sum float64
					for ic := 0; ic < cIn; ic++ {
						for kh := 0; kh < fH; kh++ {
							for kw := 0; kw < fW; kw++ {
								ih := oh*strideH - padH + kh
								iw := ow*strideW - padW + kw
								if ih < 0 || ih >= inH || iw < 0 || iw >= inW {
									continue
								}
								xi := onn*cIn*inH*inW + ic*inH*inW + ih*inW + iw
								wi := oc*cIn*fH*fW + ic*fH*fW + kh*fW + kw
								sum += float64(input[xi]) * float64(filter[wi])
							}
						}
					}
					yi := onn*cOut*outH*outW + oc*outH*outW + oh*outW + ow
					want[yi] = float32(sum)
				}
			}
		}
	}
	checkParity(t, "ConvForward", got, want)
}

func TestBatchNormInferenceParity(t *testing.T) {
	skipIfNoAvailable(t)

	// 1x2x2x2 input, spatial batch norm over 2 channels
	input := []float32{
		// Channel 0: 2x2
		1, 2, 3, 4,
		// Channel 1: 2x2
		5, 6, 7, 8,
	}
	n, c, height, w := 1, 2, 2, 2
	count := n * c * height * w

	scale := []float32{2.0, 0.5}
	bias := []float32{1.0, -1.0}
	mean := []float32{2.5, 6.5}
	variance := []float32{1.25, 1.25}
	epsilon := 1e-5

	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer h.Destroy()

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

	bnDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor(bn): %v", err)
	}
	defer bnDesc.Destroy()
	if err := bnDesc.Set4d(NCHW, Float32, 1, c, 1, 1); err != nil {
		t.Fatalf("Set4d(bn): %v", err)
	}

	xDev := allocDevFloat32(t, input)
	defer func() { _ = cuda.Free(xDev) }()
	yDev := allocDevZero(t, count)
	defer func() { _ = cuda.Free(yDev) }()
	scaleDev := allocDevFloat32(t, scale)
	defer func() { _ = cuda.Free(scaleDev) }()
	biasDev := allocDevFloat32(t, bias)
	defer func() { _ = cuda.Free(biasDev) }()
	meanDev := allocDevFloat32(t, mean)
	defer func() { _ = cuda.Free(meanDev) }()
	varDev := allocDevFloat32(t, variance)
	defer func() { _ = cuda.Free(varDev) }()

	if err := h.BatchNormalizationForwardInference(
		BatchNormSpatial,
		1.0, 0.0,
		xDesc, xDev,
		yDesc, yDev,
		bnDesc,
		scaleDev, biasDev,
		meanDev, varDev,
		epsilon,
	); err != nil {
		t.Fatalf("BatchNormalizationForwardInference: %v", err)
	}

	got := readDevFloat32(t, yDev, count)

	// CPU reference: y = scale * (x - mean) / sqrt(var + eps) + bias
	want := make([]float32, count)
	spatial := height * w
	for ch := 0; ch < c; ch++ {
		invStd := 1.0 / math.Sqrt(float64(variance[ch])+epsilon)
		for j := 0; j < spatial; j++ {
			idx := ch*spatial + j
			norm := (float64(input[idx]) - float64(mean[ch])) * invStd
			want[idx] = float32(float64(scale[ch])*norm + float64(bias[ch]))
		}
	}
	checkParity(t, "BatchNormInference", got, want)
}
