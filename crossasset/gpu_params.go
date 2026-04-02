package crossasset

import (
	"github.com/zerfoo/ztensor/tensor"
)

// gpuCALayer holds float32 tensor weights for one cross-attention transformer layer.
type gpuCALayer struct {
	qW, kW, vW *tensor.TensorNumeric[float32] // [DModel, DModel]
	outW       *tensor.TensorNumeric[float32]  // [DModel, DModel]
	lnGamma    *tensor.TensorNumeric[float32]  // [1, DModel]
	lnBeta     *tensor.TensorNumeric[float32]  // [1, DModel]

	ffnW1    *tensor.TensorNumeric[float32] // [DModel, 4*DModel]
	ffnB1    *tensor.TensorNumeric[float32] // [1, 4*DModel]
	ffnW2    *tensor.TensorNumeric[float32] // [4*DModel, DModel]
	ffnB2    *tensor.TensorNumeric[float32] // [1, DModel]
	ffnGamma *tensor.TensorNumeric[float32] // [1, DModel]
	ffnBeta  *tensor.TensorNumeric[float32] // [1, DModel]
}

// gpuCAParams holds all model parameters as float32 tensors for GPU training.
type gpuCAParams struct {
	inputW []*tensor.TensorNumeric[float32] // [NSources] each [FeaturesPerSource, DModel]
	inputB []*tensor.TensorNumeric[float32] // [NSources] each [1, DModel]
	layers []gpuCALayer
	headW  *tensor.TensorNumeric[float32] // [DModel, 3]
	headB  *tensor.TensorNumeric[float32] // [1, 3]
}

// gpuCAGrads mirrors gpuCAParams for gradient accumulation.
type gpuCAGrads = gpuCAParams

// extractGPUParams converts the Model's float64 weights to float32 tensors.
func extractGPUParams(m *Model) (*gpuCAParams, error) {
	cfg := m.config
	dm := cfg.DModel
	fps := cfg.FeaturesPerSource
	ffnDim := 4 * dm

	p := &gpuCAParams{
		inputW: make([]*tensor.TensorNumeric[float32], cfg.NSources),
		inputB: make([]*tensor.TensorNumeric[float32], cfg.NSources),
		layers: make([]gpuCALayer, cfg.NLayers),
	}

	// Input projections: Model stores inputW[s] as flat [FeaturesPerSource * DModel].
	for s := range cfg.NSources {
		wF32 := f64ToF32(m.inputW[s])
		w, err := tensor.New([]int{fps, dm}, wF32)
		if err != nil {
			return nil, err
		}
		p.inputW[s] = w

		bF32 := f64ToF32(m.inputB[s])
		b, err := tensor.New([]int{1, dm}, bF32)
		if err != nil {
			return nil, err
		}
		p.inputB[s] = b
	}

	// Per-layer weights.
	for li, l := range m.layers {
		gl := &p.layers[li]
		var err error

		gl.qW, err = newF32Mat(l.qW, dm, dm)
		if err != nil {
			return nil, err
		}
		gl.kW, err = newF32Mat(l.kW, dm, dm)
		if err != nil {
			return nil, err
		}
		gl.vW, err = newF32Mat(l.vW, dm, dm)
		if err != nil {
			return nil, err
		}
		gl.outW, err = newF32Mat(l.outW, dm, dm)
		if err != nil {
			return nil, err
		}

		gl.lnGamma, err = newF32Row(l.lnGamma, dm)
		if err != nil {
			return nil, err
		}
		gl.lnBeta, err = newF32Row(l.lnBeta, dm)
		if err != nil {
			return nil, err
		}

		gl.ffnW1, err = newF32Mat(l.ffnW1, dm, ffnDim)
		if err != nil {
			return nil, err
		}
		gl.ffnB1, err = newF32Row(l.ffnB1, ffnDim)
		if err != nil {
			return nil, err
		}
		gl.ffnW2, err = newF32Mat(l.ffnW2, ffnDim, dm)
		if err != nil {
			return nil, err
		}
		gl.ffnB2, err = newF32Row(l.ffnB2, dm)
		if err != nil {
			return nil, err
		}

		gl.ffnGamma, err = newF32Row(l.ffnGamma, dm)
		if err != nil {
			return nil, err
		}
		gl.ffnBeta, err = newF32Row(l.ffnBeta, dm)
		if err != nil {
			return nil, err
		}
	}

	// Classification head: Model stores headW as flat [DModel * 3].
	hwF32 := f64ToF32(m.headW)
	var err error
	p.headW, err = tensor.New([]int{dm, 3}, hwF32)
	if err != nil {
		return nil, err
	}
	hbF32 := f64ToF32(m.headB)
	p.headB, err = tensor.New([]int{1, 3}, hbF32)
	if err != nil {
		return nil, err
	}

	return p, nil
}

// allocGrads creates zero-valued gradient tensors matching each parameter shape.
func allocGrads(p *gpuCAParams) (*gpuCAGrads, error) {
	g := &gpuCAGrads{
		inputW: make([]*tensor.TensorNumeric[float32], len(p.inputW)),
		inputB: make([]*tensor.TensorNumeric[float32], len(p.inputB)),
		layers: make([]gpuCALayer, len(p.layers)),
	}

	for s := range p.inputW {
		var err error
		g.inputW[s], err = tensor.New[float32](p.inputW[s].Shape(), nil)
		if err != nil {
			return nil, err
		}
		g.inputB[s], err = tensor.New[float32](p.inputB[s].Shape(), nil)
		if err != nil {
			return nil, err
		}
	}

	for li := range p.layers {
		pl := &p.layers[li]
		gl := &g.layers[li]
		fields := []struct {
			src  *tensor.TensorNumeric[float32]
			dst  **tensor.TensorNumeric[float32]
		}{
			{pl.qW, &gl.qW}, {pl.kW, &gl.kW}, {pl.vW, &gl.vW}, {pl.outW, &gl.outW},
			{pl.lnGamma, &gl.lnGamma}, {pl.lnBeta, &gl.lnBeta},
			{pl.ffnW1, &gl.ffnW1}, {pl.ffnB1, &gl.ffnB1},
			{pl.ffnW2, &gl.ffnW2}, {pl.ffnB2, &gl.ffnB2},
			{pl.ffnGamma, &gl.ffnGamma}, {pl.ffnBeta, &gl.ffnBeta},
		}
		for _, f := range fields {
			var err error
			*f.dst, err = tensor.New[float32](f.src.Shape(), nil)
			if err != nil {
				return nil, err
			}
		}
	}

	var err error
	g.headW, err = tensor.New[float32](p.headW.Shape(), nil)
	if err != nil {
		return nil, err
	}
	g.headB, err = tensor.New[float32](p.headB.Shape(), nil)
	if err != nil {
		return nil, err
	}

	return g, nil
}


// writeBackParams copies float32 GPU params back to the Model's float64 weights.
func writeBackParams(m *Model, p *gpuCAParams) {
	cfg := m.config
	dm := cfg.DModel
	ffnDim := 4 * dm

	for s := range cfg.NSources {
		f32ToF64(p.inputW[s].Data(), m.inputW[s])
		f32ToF64(p.inputB[s].Data(), m.inputB[s])
	}

	for li := range m.layers {
		l := &m.layers[li]
		gl := &p.layers[li]
		f32ToF64(gl.qW.Data(), l.qW)
		f32ToF64(gl.kW.Data(), l.kW)
		f32ToF64(gl.vW.Data(), l.vW)
		f32ToF64(gl.outW.Data(), l.outW)
		f32ToF64Slice(gl.lnGamma.Data(), l.lnGamma, dm)
		f32ToF64Slice(gl.lnBeta.Data(), l.lnBeta, dm)
		f32ToF64(gl.ffnW1.Data(), l.ffnW1)
		f32ToF64Slice(gl.ffnB1.Data(), l.ffnB1, ffnDim)
		f32ToF64(gl.ffnW2.Data(), l.ffnW2)
		f32ToF64Slice(gl.ffnB2.Data(), l.ffnB2, dm)
		f32ToF64Slice(gl.ffnGamma.Data(), l.ffnGamma, dm)
		f32ToF64Slice(gl.ffnBeta.Data(), l.ffnBeta, dm)
	}

	f32ToF64(p.headW.Data(), m.headW)
	f32ToF64Slice(p.headB.Data(), m.headB, 3)
}

// --- helpers ---

func f64ToF32(src []float64) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = float32(v)
	}
	return dst
}

func f32ToF64(src []float32, dst []float64) {
	for i, v := range src {
		dst[i] = float64(v)
	}
}

func f32ToF64Slice(src []float32, dst []float64, n int) {
	for i := range n {
		dst[i] = float64(src[i])
	}
}

func newF32Mat(src []float64, rows, cols int) (*tensor.TensorNumeric[float32], error) {
	return tensor.New([]int{rows, cols}, f64ToF32(src))
}

func newF32Row(src []float64, cols int) (*tensor.TensorNumeric[float32], error) {
	return tensor.New([]int{1, cols}, f64ToF32(src))
}
