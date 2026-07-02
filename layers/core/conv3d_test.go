package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// naiveConv3d is an independent, direct nested-loop reference (groups=1) used
// to validate the im2col+MatMul Conv3d.Forward without needing torch.
func naiveConv3d(
	x []float64, n, cIn, inD, inH, inW int,
	w []float64, cOut, kD, kH, kW int,
	bias []float64,
	sD, sH, sW, padDb, padHb, padWb, dD, dH, dW int,
) ([]float64, int, int, int) {
	outD := (inD+2*padDb-dD*(kD-1)-1)/sD + 1 // tests use symmetric begin/end pads
	outH := (inH+2*padHb-dH*(kH-1)-1)/sH + 1
	outW := (inW+2*padWb-dW*(kW-1)-1)/sW + 1
	out := make([]float64, n*cOut*outD*outH*outW)
	for ni := range n {
		for oc := range cOut {
			b := 0.0
			if bias != nil {
				b = bias[oc]
			}
			for od := range outD {
				for oh := range outH {
					for ow := range outW {
						acc := b
						for ic := range cIn {
							for kd := range kD {
								id := od*sD - padDb + kd*dD
								if id < 0 || id >= inD {
									continue
								}
								for kh := range kH {
									ih := oh*sH - padHb + kh*dH
									if ih < 0 || ih >= inH {
										continue
									}
									for kw := range kW {
										iw := ow*sW - padWb + kw*dW
										if iw < 0 || iw >= inW {
											continue
										}
										xv := x[(((ni*cIn+ic)*inD+id)*inH+ih)*inW+iw]
										wv := w[(((oc*cIn+ic)*kD+kd)*kH+kh)*kW+kw]
										acc += xv * wv
									}
								}
							}
						}
						out[(((ni*cOut+oc)*outD+od)*outH+oh)*outW+ow] = acc
					}
				}
			}
		}
	}
	return out, outD, outH, outW
}

func seqData(n int, seed float64) []float64 {
	d := make([]float64, n)
	s := seed
	for i := range d {
		// deterministic, varied values in ~[-1,1]
		s = math.Mod(s*1.123456789+0.314159, 2.0)
		d[i] = s - 1.0
	}
	return d
}

func TestConv3d_OnesForward(t *testing.T) {
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	mkOnes := func(shape []int) *tensor.TensorNumeric[float64] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float64, size)
		for i := range data {
			data[i] = 1
		}
		tt, _ := tensor.New[float64](shape, data)
		return tt
	}
	x := mkOnes([]int{1, 1, 3, 3, 3})
	w := mkOnes([]int{1, 1, 2, 2, 2})
	conv := NewConv3d[float64](engine, ops, []int{1, 1, 1}, []int{0, 0, 0, 0, 0, 0}, []int{1, 1, 1}, 1)
	out, err := conv.Forward(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if !shapeEq(out.Shape(), []int{1, 1, 2, 2, 2}) {
		t.Fatalf("shape = %v, want [1 1 2 2 2]", out.Shape())
	}
	for i, v := range out.Data() {
		if math.Abs(v-8) > 1e-9 { // 2*2*2 sum of ones
			t.Fatalf("out[%d] = %v, want 8", i, v)
		}
	}
}

func TestConv3d_MatchesNaiveReference(t *testing.T) {
	type tc struct {
		name                      string
		n, cIn, inD, inH, inW     int
		cOut, kD, kH, kW          int
		sD, sH, sW                int
		padD, padH, padW          int
		dD, dH, dW                int
		bias                      bool
	}
	cases := []tc{
		{"valid_stride1", 2, 3, 4, 5, 6, 4, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false},
		{"strided_padded_bias", 1, 2, 5, 5, 5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, true},
		{"dilated", 1, 2, 6, 6, 6, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, true},
	}
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			xData := seqData(c.n*c.cIn*c.inD*c.inH*c.inW, 0.7)
			wData := seqData(c.cOut*c.cIn*c.kD*c.kH*c.kW, 1.3)
			var bData []float64
			if c.bias {
				bData = seqData(c.cOut, 2.1)
			}
			x, _ := tensor.New[float64]([]int{c.n, c.cIn, c.inD, c.inH, c.inW}, xData)
			w, _ := tensor.New[float64]([]int{c.cOut, c.cIn, c.kD, c.kH, c.kW}, wData)
			inputs := []*tensor.TensorNumeric[float64]{x, w}
			if c.bias {
				b, _ := tensor.New[float64]([]int{c.cOut}, bData)
				inputs = append(inputs, b)
			}
			conv := NewConv3d[float64](engine, ops,
				[]int{c.sD, c.sH, c.sW},
				[]int{c.padD, c.padH, c.padW, c.padD, c.padH, c.padW},
				[]int{c.dD, c.dH, c.dW}, 1)
			out, err := conv.Forward(context.Background(), inputs...)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			want, oD, oH, oW := naiveConv3d(xData, c.n, c.cIn, c.inD, c.inH, c.inW,
				wData, c.cOut, c.kD, c.kH, c.kW, bData,
				c.sD, c.sH, c.sW, c.padD, c.padH, c.padW, c.dD, c.dH, c.dW)
			if !shapeEq(out.Shape(), []int{c.n, c.cOut, oD, oH, oW}) {
				t.Fatalf("shape = %v, want [%d %d %d %d %d]", out.Shape(), c.n, c.cOut, oD, oH, oW)
			}
			got := out.Data()
			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-9 {
					t.Fatalf("mismatch at %d: got %v want %v", i, got[i], want[i])
				}
			}
		})
	}
}
