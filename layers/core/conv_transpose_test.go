package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestConvTranspose3d_TinyScatter(t *testing.T) {
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	// X = [[v]] (single element), ones kernel 2x2x2, stride1 pad0 -> [1,1,2,2,2]
	// all = v (each kernel tap deposits v*1 at a distinct output position).
	x, _ := tensor.New[float64]([]int{1, 1, 1, 1, 1}, []float64{2.5})
	w := func() *tensor.TensorNumeric[float64] {
		d := make([]float64, 8)
		for i := range d {
			d[i] = 1
		}
		tt, _ := tensor.New[float64]([]int{1, 1, 2, 2, 2}, d)
		return tt
	}()
	ct := NewConvTranspose3d[float64](engine, ops, []int{1, 1, 1}, []int{0, 0, 0, 0, 0, 0}, []int{1, 1, 1}, []int{0, 0, 0}, 1)
	out, err := ct.Forward(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if !shapeEq(out.Shape(), []int{1, 1, 2, 2, 2}) {
		t.Fatalf("shape = %v, want [1 1 2 2 2]", out.Shape())
	}
	for i, v := range out.Data() {
		if math.Abs(v-2.5) > 1e-9 {
			t.Fatalf("out[%d] = %v, want 2.5", i, v)
		}
	}
}

func TestConvTranspose3d_StridedUpsampleShape(t *testing.T) {
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	// stride 2, kernel 2, input 2^3 ones -> output 4^3, each position covered
	// exactly once (no overlap) -> all ones.
	xd := make([]float64, 8)
	for i := range xd {
		xd[i] = 1
	}
	x, _ := tensor.New[float64]([]int{1, 1, 2, 2, 2}, xd)
	wd := make([]float64, 8)
	for i := range wd {
		wd[i] = 1
	}
	w, _ := tensor.New[float64]([]int{1, 1, 2, 2, 2}, wd)
	ct := NewConvTranspose3d[float64](engine, ops, []int{2, 2, 2}, []int{0, 0, 0, 0, 0, 0}, []int{1, 1, 1}, []int{0, 0, 0}, 1)
	out, err := ct.Forward(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if !shapeEq(out.Shape(), []int{1, 1, 4, 4, 4}) {
		t.Fatalf("shape = %v, want [1 1 4 4 4]", out.Shape())
	}
	for i, v := range out.Data() {
		if math.Abs(v-1) > 1e-9 {
			t.Fatalf("out[%d] = %v, want 1", i, v)
		}
	}
}

// flipSwapWeight converts a conv_transpose weight Wt[Cin,Cout,kD,kH,kW] into the
// equivalent conv weight Wc[Cout,Cin,kD,kH,kW] (swap in/out channels, flip the
// kernel spatially). With stride1/dilation1 and full padding (k-1), a plain
// convolution with Wc reproduces the transposed convolution -- an independent
// cross-check of ConvTranspose3d against the naive-verified Conv3d.
func flipSwapWeight(wt []float64, cIn, cOut, kD, kH, kW int) []float64 {
	wc := make([]float64, cOut*cIn*kD*kH*kW)
	for cin := range cIn {
		for cout := range cOut {
			for kd := range kD {
				for kh := range kH {
					for kw := range kW {
						src := (((cin*cOut+cout)*kD+(kD-1-kd))*kH+(kH-1-kh))*kW + (kW - 1 - kw)
						dst := (((cout*cIn+cin)*kD+kd)*kH+kh)*kW + kw
						wc[dst] = wt[src]
					}
				}
			}
		}
	}
	return wc
}

func TestConvTranspose3d_AdjointMatchesConv3d(t *testing.T) {
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	const (
		n, cIn, cOut    = 2, 3, 4
		inD, inH, inW   = 3, 4, 3
		kD, kH, kW      = 2, 3, 2
	)
	xData := seqData(n*cIn*inD*inH*inW, 0.55)
	wtData := seqData(cIn*cOut*kD*kH*kW, 1.7) // [Cin,Cout,kD,kH,kW]

	x, _ := tensor.New[float64]([]int{n, cIn, inD, inH, inW}, xData)
	wt, _ := tensor.New[float64]([]int{cIn, cOut, kD, kH, kW}, wtData)

	// Transposed conv (stride1, pad0, dilation1).
	ct := NewConvTranspose3d[float64](engine, ops, []int{1, 1, 1}, []int{0, 0, 0, 0, 0, 0}, []int{1, 1, 1}, []int{0, 0, 0}, 1)
	ctOut, err := ct.Forward(context.Background(), x, wt)
	if err != nil {
		t.Fatalf("ConvTranspose3d Forward: %v", err)
	}

	// Equivalent plain conv: flipped/swapped weight, full padding (k-1).
	wcData := flipSwapWeight(wtData, cIn, cOut, kD, kH, kW)
	wc, _ := tensor.New[float64]([]int{cOut, cIn, kD, kH, kW}, wcData)
	conv := NewConv3d[float64](engine, ops, []int{1, 1, 1},
		[]int{kD - 1, kH - 1, kW - 1, kD - 1, kH - 1, kW - 1}, []int{1, 1, 1}, 1)
	convOut, err := conv.Forward(context.Background(), x, wc)
	if err != nil {
		t.Fatalf("Conv3d Forward: %v", err)
	}

	if !shapeEq(ctOut.Shape(), convOut.Shape()) {
		t.Fatalf("shape mismatch: convT %v vs conv %v", ctOut.Shape(), convOut.Shape())
	}
	a, b := ctOut.Data(), convOut.Data()
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-9 {
			t.Fatalf("adjoint mismatch at %d: convT %v vs conv %v", i, a[i], b[i])
		}
	}
}
