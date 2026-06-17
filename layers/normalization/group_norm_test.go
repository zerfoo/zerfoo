package normalization

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func gnSeq(n int, seed float64) []float64 {
	d := make([]float64, n)
	s := seed
	for i := range d {
		s = math.Mod(s*1.123456789+0.314159, 2.0)
		d[i] = s - 1.0
	}
	return d
}

// naiveGroupNorm is an independent reference: per (sample, group) mean/var over
// the (C/groups)*S elements, then per-channel affine.
func naiveGroupNorm(x []float64, n, c, s, groups int, eps float64, scale, bias []float64) []float64 {
	cg := c / groups
	m := cg * s
	out := make([]float64, n*c*s)
	for ni := range n {
		for gi := range groups {
			// mean/var over the group's m elements.
			mean := 0.0
			for cc := range cg {
				ch := gi*cg + cc
				for si := range s {
					mean += x[(ni*c+ch)*s+si]
				}
			}
			mean /= float64(m)
			varr := 0.0
			for cc := range cg {
				ch := gi*cg + cc
				for si := range s {
					d := x[(ni*c+ch)*s+si] - mean
					varr += d * d
				}
			}
			varr /= float64(m)
			inv := 1.0 / math.Sqrt(varr+eps)
			for cc := range cg {
				ch := gi*cg + cc
				for si := range s {
					idx := (ni*c+ch)*s + si
					y := (x[idx] - mean) * inv
					if scale != nil {
						y *= scale[ch]
					}
					if bias != nil {
						y += bias[ch]
					}
					out[idx] = y
				}
			}
		}
	}
	return out
}

func TestGroupNorm_ForwardMatchesNaive(t *testing.T) {
	type tc struct {
		name   string
		shape  []int
		groups int
	}
	cases := []tc{
		{"2d_no_spatial", []int{2, 4, 3}, 2},
		{"4d_image", []int{2, 6, 2, 2}, 3},
		{"5d_video", []int{1, 8, 2, 2, 2}, 4},
	}
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	const eps = 1e-5
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			n, ch := c.shape[0], c.shape[1]
			s := 1
			for _, d := range c.shape[2:] {
				s *= d
			}
			xData := gnSeq(n*ch*s, 0.6)
			scaleData := gnSeq(ch, 1.1)
			biasData := gnSeq(ch, 2.2)
			x, _ := tensor.New[float64](c.shape, xData)
			scaleT, _ := tensor.New[float64]([]int{ch}, scaleData)
			biasT, _ := tensor.New[float64]([]int{ch}, biasData)
			scale, _ := graph.NewParameter[float64]("scale", scaleT, tensor.New[float64])
			bias, _ := graph.NewParameter[float64]("bias", biasT, tensor.New[float64])
			gn := NewGroupNormalizationWithParams[float64](engine, ops, c.groups, eps, scale, bias)
			out, err := gn.Forward(context.Background(), x)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			want := naiveGroupNorm(xData, n, ch, s, c.groups, eps, scaleData, biasData)
			got := out.Data()
			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-9 {
					t.Fatalf("mismatch at %d: got %v want %v", i, got[i], want[i])
				}
			}
		})
	}
}

func TestGroupNorm_BackwardFiniteDiff(t *testing.T) {
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	const (
		n, ch, s, groups = 2, 4, 3, 2
		eps              = 1e-5
		h                = 1e-6
	)
	shape := []int{n, ch, s}
	xData := gnSeq(n*ch*s, 0.42)
	scaleData := gnSeq(ch, 1.7)
	biasData := gnSeq(ch, 0.9)
	dOut := gnSeq(n*ch*s, 3.3) // arbitrary upstream gradient

	// forwardData runs a fresh layer and returns the flat output.
	forwardData := func(x, sc, bi []float64) []float64 {
		xt, _ := tensor.New[float64](shape, append([]float64(nil), x...))
		st, _ := tensor.New[float64]([]int{ch}, append([]float64(nil), sc...))
		bt, _ := tensor.New[float64]([]int{ch}, append([]float64(nil), bi...))
		sp, _ := graph.NewParameter[float64]("scale", st, tensor.New[float64])
		bp, _ := graph.NewParameter[float64]("bias", bt, tensor.New[float64])
		gn := NewGroupNormalizationWithParams[float64](engine, ops, groups, eps, sp, bp)
		out, err := gn.Forward(context.Background(), xt)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		return out.Data()
	}
	lossAt := func(x, sc, bi []float64) float64 {
		y := forwardData(x, sc, bi)
		l := 0.0
		for i := range y {
			l += dOut[i] * y[i]
		}
		return l
	}

	// Analytic gradients.
	xt, _ := tensor.New[float64](shape, append([]float64(nil), xData...))
	st, _ := tensor.New[float64]([]int{ch}, append([]float64(nil), scaleData...))
	bt, _ := tensor.New[float64]([]int{ch}, append([]float64(nil), biasData...))
	sp, _ := graph.NewParameter[float64]("scale", st, tensor.New[float64])
	bp, _ := graph.NewParameter[float64]("bias", bt, tensor.New[float64])
	gn := NewGroupNormalizationWithParams[float64](engine, ops, groups, eps, sp, bp)
	if _, err := gn.Forward(context.Background(), xt); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	dOutT, _ := tensor.New[float64](shape, append([]float64(nil), dOut...))
	grads, err := gn.Backward(context.Background(), 0, dOutT, xt)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	dXAnalytic := grads[0].Data()

	central := func(perturb func(i int, delta float64) ([]float64, []float64, []float64), i int) float64 {
		xp, sp, bp := perturb(i, +h)
		lp := lossAt(xp, sp, bp)
		xm, sm, bm := perturb(i, -h)
		lm := lossAt(xm, sm, bm)
		return (lp - lm) / (2 * h)
	}

	// dX check.
	for i := range xData {
		num := central(func(j int, delta float64) ([]float64, []float64, []float64) {
			xp := append([]float64(nil), xData...)
			xp[j] += delta
			return xp, scaleData, biasData
		}, i)
		if math.Abs(num-dXAnalytic[i]) > 1e-5 {
			t.Fatalf("dX[%d]: analytic %v vs finite-diff %v", i, dXAnalytic[i], num)
		}
	}
	// dScale / dBias checks.
	dScale := sp.Gradient.Data()
	dBias := bp.Gradient.Data()
	for cidx := range ch {
		numS := central(func(j int, delta float64) ([]float64, []float64, []float64) {
			scp := append([]float64(nil), scaleData...)
			scp[j] += delta
			return xData, scp, biasData
		}, cidx)
		if math.Abs(numS-dScale[cidx]) > 1e-5 {
			t.Fatalf("dScale[%d]: analytic %v vs finite-diff %v", cidx, dScale[cidx], numS)
		}
		numB := central(func(j int, delta float64) ([]float64, []float64, []float64) {
			bip := append([]float64(nil), biasData...)
			bip[j] += delta
			return xData, scaleData, bip
		}, cidx)
		if math.Abs(numB-dBias[cidx]) > 1e-5 {
			t.Fatalf("dBias[%d]: analytic %v vs finite-diff %v", cidx, dBias[cidx], numB)
		}
	}
}
