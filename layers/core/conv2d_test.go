package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func newConv2dOnesFloat32(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = 1
	}
	t, _ := tensor.New[float32](shape, data)
	return t
}

func shapeEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestConv2d_ForwardShapeAndValues exercises stride/kernel combinations using
// a table-driven approach to avoid structural code duplication (dupl linter).
func TestConv2d_ForwardShapeAndValues(t *testing.T) {
	type tc struct {
		name       string
		inputShape []int
		kernShape  []int
		strides    []int
		wantShape  []int
		wantVal    float32
	}
	cases := []tc{
		{
			name:       "5x5_3x3_stride1",
			inputShape: []int{1, 1, 5, 5},
			kernShape:  []int{1, 1, 3, 3},
			strides:    []int{1, 1},
			wantShape:  []int{1, 1, 3, 3},
			wantVal:    9, // 3*3 sum of ones
		},
		{
			name:       "4x4_2x2_stride2",
			inputShape: []int{1, 1, 4, 4},
			kernShape:  []int{1, 1, 2, 2},
			strides:    []int{2, 2},
			wantShape:  []int{1, 1, 2, 2},
			wantVal:    4, // 2*2 sum of ones
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](&ops)

			x := newConv2dOnesFloat32(c.inputShape)
			w := newConv2dOnesFloat32(c.kernShape)

			conv := NewConv2d[float32](engine, &ops, c.strides, []int{0, 0, 0, 0}, []int{1, 1}, 1)
			out, err := conv.Forward(context.Background(), x, w)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			if !shapeEq(out.Shape(), c.wantShape) {
				t.Fatalf("shape mismatch: got %v want %v", out.Shape(), c.wantShape)
			}
			for i, v := range out.Data() {
				if v != c.wantVal {
					t.Errorf("out[%d] = %v, want %v", i, v, c.wantVal)
				}
			}
		})
	}
}

// TestConv2d_WithBias: same as AllOnes test but adds bias=1 to each output channel.
func TestConv2d_WithBias(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x := newConv2dOnesFloat32([]int{1, 1, 5, 5})
	w := newConv2dOnesFloat32([]int{1, 1, 3, 3})
	b, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	out, err := conv.Forward(context.Background(), x, w, b)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	for i, v := range out.Data() {
		if v != 10 {
			t.Errorf("out[%d] = %v, want 10", i, v)
		}
	}
}

// TestConv2d_TwoOutputChannels: kernel [2,1,3,3] produces two output channels.
func TestConv2d_TwoOutputChannels(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x := newConv2dOnesFloat32([]int{1, 1, 5, 5})
	// First kernel all-ones (each out = 9), second kernel all-twos (each out = 18).
	wData := make([]float32, 2*1*3*3)
	for i := range 9 {
		wData[i] = 1
	}
	for i := 9; i < 18; i++ {
		wData[i] = 2
	}
	w, _ := tensor.New[float32]([]int{2, 1, 3, 3}, wData)

	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	out, err := conv.Forward(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	want := []int{1, 2, 3, 3}
	if !shapeEq(out.Shape(), want) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), want)
	}
	data := out.Data()
	// First 9 values are channel 0 (all 9), next 9 are channel 1 (all 18).
	for i, v := range data[:9] {
		if v != 9 {
			t.Errorf("ch0[%d] = %v, want 9", i, v)
		}
	}
	for i, v := range data[9:] {
		if v != 18 {
			t.Errorf("ch1[%d] = %v, want 18", i, v)
		}
	}
}

func TestConv2d_InvalidInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)

	_, err := conv.Forward(context.Background())
	if err == nil {
		t.Fatal("expected error for 0 inputs")
	}
}

func TestConv2d_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	if conv.OpType() != "Conv" {
		t.Errorf("OpType = %q, want %q", conv.OpType(), "Conv")
	}
}

func TestConv2d_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	conv := NewConv2d[float32](engine, &ops, []int{2, 3}, []int{1, 1, 1, 1}, []int{2, 2}, 4)
	attrs := conv.Attributes()
	if attrs == nil {
		t.Fatal("Attributes returned nil")
	}
	if _, ok := attrs["strides"]; !ok {
		t.Error("missing strides attribute")
	}
	if _, ok := attrs["pads"]; !ok {
		t.Error("missing pads attribute")
	}
	if _, ok := attrs["dilations"]; !ok {
		t.Error("missing dilations attribute")
	}
	if _, ok := attrs["group"]; !ok {
		t.Error("missing group attribute")
	}
}

func TestConv2d_ParametersAndBackward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	if conv.Parameters() != nil {
		t.Error("Parameters should be nil")
	}
	grads, err := conv.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("Backward should return nil grads for inference-only layer")
	}
}

func TestBuildConv2d_Defaults(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	// Build with empty attributes (should use defaults).
	node, err := BuildConv2d[float32](engine, &ops, "conv", nil, map[string]interface{}{})
	if err != nil {
		t.Fatalf("BuildConv2d failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildConv2d returned nil node")
	}
}

func TestBuildConv2d_WithAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	attrs := map[string]interface{}{
		"strides":   []int64{2, 2},
		"pads":      []int64{1, 1, 1, 1},
		"dilations": []int64{1, 1},
		"group":     int64(1),
	}
	node, err := BuildConv2d[float32](engine, &ops, "conv", nil, attrs)
	if err != nil {
		t.Fatalf("BuildConv2d failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildConv2d returned nil node")
	}
	if node.OpType() != "Conv" {
		t.Errorf("OpType = %q, want Conv", node.OpType())
	}
}

// TestConv2d_OutputShape verifies OutputShape is populated after Forward.
func TestConv2d_OutputShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x := newConv2dOnesFloat32([]int{1, 1, 5, 5})
	w := newConv2dOnesFloat32([]int{1, 1, 3, 3})

	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	if _, err := conv.Forward(context.Background(), x, w); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	want := []int{1, 1, 3, 3}
	if !shapeEq(conv.OutputShape(), want) {
		t.Errorf("OutputShape = %v, want %v", conv.OutputShape(), want)
	}
}

// conv2dReference computes 2D convolution using a direct nested loop (reference implementation).
// Used to verify im2col + MatMul parity.
func conv2dReference(
	xData []float32, xShape []int,
	wData []float32, wShape []int,
	bData []float32,
	strides [2]int, pads [4]int, dilations [2]int,
	groups int,
) ([]float32, []int) {
	n, cIn, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3]
	cOut, cInG, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]
	sH, sW := strides[0], strides[1]
	padT, padL := pads[0], pads[1]
	dH, dW := dilations[0], dilations[1]

	outH := (inH+pads[0]+pads[2]-dH*(kH-1)-1)/sH + 1
	outW := (inW+pads[1]+pads[3]-dW*(kW-1)-1)/sW + 1

	out := make([]float32, n*cOut*outH*outW)
	cOutPerGroup := cOut / groups

	for ni := range n {
		for g := range groups {
			icOff := g * cInG
			ocOff := g * cOutPerGroup
			for oc := range cOutPerGroup {
				absOC := ocOff + oc
				for oh := range outH {
					for ow := range outW {
						var val float32
						for ic := range cInG {
							for kh := range kH {
								for kw := range kW {
									ih := oh*sH - padT + kh*dH
									iw := ow*sW - padL + kw*dW
									if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
										xIdx := ni*cIn*inH*inW + (icOff+ic)*inH*inW + ih*inW + iw
										wIdx := absOC*cInG*kH*kW + ic*kH*kW + kh*kW + kw
										val += xData[xIdx] * wData[wIdx]
									}
								}
							}
						}
						idx := ni*cOut*outH*outW + absOC*outH*outW + oh*outW + ow
						out[idx] = val
					}
				}
			}
		}
	}

	if bData != nil {
		for ni := range n {
			for oc := range cOut {
				for oh := range outH {
					for ow := range outW {
						idx := ni*cOut*outH*outW + oc*outH*outW + oh*outW + ow
						out[idx] += bData[oc]
					}
				}
			}
		}
	}

	return out, []int{n, cOut, outH, outW}
}

// newConv2dSeqFloat32 creates a tensor with sequential values 0.01, 0.02, ...
func newConv2dSeqFloat32(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i+1) * 0.01
	}
	t, _ := tensor.New[float32](shape, data)
	return t
}

// TestConv2d_Im2colParity verifies the im2col+MatMul implementation matches
// a direct nested-loop reference across various configurations.
func TestConv2d_Im2colParity(t *testing.T) {
	const tol = 1e-5

	type tc struct {
		name      string
		xShape    []int
		wShape    []int
		bShape    []int // nil means no bias
		strides   [2]int
		pads      [2]int // symmetric: [padH, padW] -> [padH, padW, padH, padW]
		dilations [2]int
		groups    int
	}
	cases := []tc{
		{
			name:      "3x3_kernel_no_pad",
			xShape:    []int{1, 1, 5, 5},
			wShape:    []int{1, 1, 3, 3},
			strides:   [2]int{1, 1},
			pads:      [2]int{0, 0},
			dilations: [2]int{1, 1},
			groups:    1,
		},
		{
			name:      "3x3_kernel_pad1",
			xShape:    []int{1, 1, 5, 5},
			wShape:    []int{1, 1, 3, 3},
			strides:   [2]int{1, 1},
			pads:      [2]int{1, 1},
			dilations: [2]int{1, 1},
			groups:    1,
		},
		{
			name:      "3x3_stride2",
			xShape:    []int{1, 1, 7, 7},
			wShape:    []int{2, 1, 3, 3},
			strides:   [2]int{2, 2},
			pads:      [2]int{0, 0},
			dilations: [2]int{1, 1},
			groups:    1,
		},
		{
			name:      "3x3_dilation2",
			xShape:    []int{1, 1, 7, 7},
			wShape:    []int{1, 1, 3, 3},
			strides:   [2]int{1, 1},
			pads:      [2]int{0, 0},
			dilations: [2]int{2, 2},
			groups:    1,
		},
		{
			name:      "with_bias",
			xShape:    []int{1, 2, 5, 5},
			wShape:    []int{3, 2, 3, 3},
			bShape:    []int{3},
			strides:   [2]int{1, 1},
			pads:      [2]int{1, 1},
			dilations: [2]int{1, 1},
			groups:    1,
		},
		{
			name:      "batch2_multichannel",
			xShape:    []int{2, 3, 6, 6},
			wShape:    []int{4, 3, 3, 3},
			strides:   [2]int{1, 1},
			pads:      [2]int{0, 0},
			dilations: [2]int{1, 1},
			groups:    1,
		},
		{
			name:      "groups2",
			xShape:    []int{1, 4, 5, 5},
			wShape:    []int{4, 2, 3, 3},
			strides:   [2]int{1, 1},
			pads:      [2]int{0, 0},
			dilations: [2]int{1, 1},
			groups:    2,
		},
		{
			name:      "5x5_kernel_pad2_stride2",
			xShape:    []int{1, 1, 8, 8},
			wShape:    []int{2, 1, 5, 5},
			strides:   [2]int{2, 2},
			pads:      [2]int{2, 2},
			dilations: [2]int{1, 1},
			groups:    1,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](&ops)

			x := newConv2dSeqFloat32(c.xShape)
			w := newConv2dSeqFloat32(c.wShape)

			pads4 := []int{c.pads[0], c.pads[1], c.pads[0], c.pads[1]}
			conv := NewConv2d[float32](engine, &ops,
				[]int{c.strides[0], c.strides[1]},
				pads4,
				[]int{c.dilations[0], c.dilations[1]},
				c.groups,
			)

			inputs := []*tensor.TensorNumeric[float32]{x, w}
			var bData []float32
			if c.bShape != nil {
				b := newConv2dSeqFloat32(c.bShape)
				inputs = append(inputs, b)
				bData = b.Data()
			}

			out, err := conv.Forward(context.Background(), inputs...)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			refData, refShape := conv2dReference(
				x.Data(), c.xShape,
				w.Data(), c.wShape,
				bData,
				c.strides, [4]int{c.pads[0], c.pads[1], c.pads[0], c.pads[1]}, c.dilations,
				c.groups,
			)

			if !shapeEq(out.Shape(), refShape) {
				t.Fatalf("shape mismatch: got %v want %v", out.Shape(), refShape)
			}

			gotData := out.Data()
			for i := range gotData {
				diff := math.Abs(float64(gotData[i] - refData[i]))
				if diff > tol {
					t.Errorf("out[%d] = %v, ref = %v, diff = %v > tol %v", i, gotData[i], refData[i], diff, tol)
				}
			}
		})
	}
}
