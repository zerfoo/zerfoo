package crossasset

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// cpuLayerNodes wraps layer objects for one transformer layer. Each node
// caches forward-pass intermediates and provides a Backward method that
// computes gradients via the Engine.
type cpuLayerNodes struct {
	qProj   *core.Linear[float32]
	kProj   *core.Linear[float32]
	vProj   *core.Linear[float32]
	outProj *core.Linear[float32]
	ffn1    *core.Linear[float32]
	gelu    *activations.Gelu[float32]
	ffn2    *core.Linear[float32]
	sdpa    *attention.ScaledDotProductAttention[float32]
}

// cpuLayerCache stores forward-pass intermediates produced by the node-based
// forward pass. All fields are tensors.
type cpuLayerCache struct {
	xIn    *tensor.TensorNumeric[float32] // [ns, dm]
	q, k, v *tensor.TensorNumeric[float32] // [ns, dm]
	// Q/K/V reshaped for SDPA: [nHeads, ns, headDim]
	qHeads, kHeads, vHeads *tensor.TensorNumeric[float32]
	attnOut *tensor.TensorNumeric[float32] // [nHeads, ns, headDim]
	concat  *tensor.TensorNumeric[float32] // [ns, dm]
	projOut *tensor.TensorNumeric[float32] // [ns, dm]
	res1    *tensor.TensorNumeric[float32] // [ns, dm]
	normed  *tensor.TensorNumeric[float32] // [ns, dm]
	// LN1 intermediates for manual backward.
	ln1NormedInput *tensor.TensorNumeric[float32] // (res1 - mean) / std
	ln1Std         *tensor.TensorNumeric[float32] // sqrt(var + eps) [ns, 1]
	// FFN intermediates.
	ffnPre  *tensor.TensorNumeric[float32] // [ns, ffnDim] after linear + bias
	ffnAct  *tensor.TensorNumeric[float32] // [ns, ffnDim] after GELU
	ffnOutT *tensor.TensorNumeric[float32] // [ns, dm] after linear + bias
	res2    *tensor.TensorNumeric[float32] // [ns, dm]
	// LN2 intermediates for manual backward.
	ln2NormedInput *tensor.TensorNumeric[float32]
	ln2Std         *tensor.TensorNumeric[float32] // [ns, 1]
}

// buildLayerNodes creates layer node objects from a layer's weight slices.
// The underlying weight data is shared (not copied).
func buildLayerNodes(l *layer, dm, nHeads int) (*cpuLayerNodes, error) {
	headDim := dm / nHeads
	ffnDim := dm * 4

	mkParam := func(name string, shape []int, data []float32) (*graph.Parameter[float32], error) {
		t, err := tensor.New[float32](shape, data)
		if err != nil {
			return nil, fmt.Errorf("buildLayerNodes: %s: %w", name, err)
		}
		return graph.NewParameter[float32](name, t, tensor.New[float32])
	}

	qP, err := mkParam("q_w", []int{dm, dm}, l.qW)
	if err != nil {
		return nil, err
	}
	kP, err := mkParam("k_w", []int{dm, dm}, l.kW)
	if err != nil {
		return nil, err
	}
	vP, err := mkParam("v_w", []int{dm, dm}, l.vW)
	if err != nil {
		return nil, err
	}
	outP, err := mkParam("out_w", []int{dm, dm}, l.outW)
	if err != nil {
		return nil, err
	}
	ffn1P, err := mkParam("ffn1_w", []int{dm, ffnDim}, l.ffnW1)
	if err != nil {
		return nil, err
	}
	ffn2P, err := mkParam("ffn2_w", []int{ffnDim, dm}, l.ffnW2)
	if err != nil {
		return nil, err
	}

	return &cpuLayerNodes{
		qProj:   core.NewLinearFromParam(cpuEngine, qP),
		kProj:   core.NewLinearFromParam(cpuEngine, kP),
		vProj:   core.NewLinearFromParam(cpuEngine, vP),
		outProj: core.NewLinearFromParam(cpuEngine, outP),
		ffn1:    core.NewLinearFromParam(cpuEngine, ffn1P),
		ffn2:    core.NewLinearFromParam(cpuEngine, ffn2P),
		gelu:    activations.NewGelu(cpuEngine, cpuOps),
		sdpa:    attention.NewBidirectionalSDPA[float32](cpuEngine, headDim),
	}, nil
}

// forwardLayerCached runs one transformer layer forward through layer nodes,
// caching all intermediates needed by backwardLayer.
func (m *Model) forwardLayerCached(x [][]float32, l layer) ([][]float32, *cpuLayerCache) {
	ns := m.config.NSources
	dm := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dm / nHeads
	ffnDim := dm * 4
	ctx := context.Background()

	nodes, err := buildLayerNodes(&l, dm, nHeads)
	panicOnErr("forwardLayerCached: build nodes", err)

	cache := &cpuLayerCache{}

	// Flatten [ns][dm] → tensor [ns, dm].
	cache.xIn = slicesToTensor(x, ns, dm)

	// Q, K, V projections via Linear nodes.
	cache.q, err = nodes.qProj.Forward(ctx, cache.xIn)
	panicOnErr("Q forward", err)
	cache.k, err = nodes.kProj.Forward(ctx, cache.xIn)
	panicOnErr("K forward", err)
	cache.v, err = nodes.vProj.Forward(ctx, cache.xIn)
	panicOnErr("V forward", err)

	// Reshape Q/K/V [ns, dm] → [nHeads, ns, headDim] for SDPA.
	cache.qHeads = reshapeForHeads(cache.q, ns, nHeads, headDim)
	cache.kHeads = reshapeForHeads(cache.k, ns, nHeads, headDim)
	cache.vHeads = reshapeForHeads(cache.v, ns, nHeads, headDim)

	// Scaled dot-product attention (bidirectional).
	cache.attnOut, err = nodes.sdpa.Forward(ctx, cache.qHeads, cache.kHeads, cache.vHeads, nil)
	panicOnErr("SDPA forward", err)

	// Reshape [nHeads, ns, headDim] → [ns, dm].
	cache.concat = reshapeFromHeads(cache.attnOut, ns, nHeads, headDim)

	// Output projection.
	cache.projOut, err = nodes.outProj.Forward(ctx, cache.concat)
	panicOnErr("out proj forward", err)

	// Residual: xIn + projOut.
	cache.res1, err = cpuEngine.Add(ctx, cache.xIn, cache.projOut)
	panicOnErr("res1", err)

	// LayerNorm 1 (manual, caching intermediates for backward).
	cache.normed, cache.ln1NormedInput, cache.ln1Std = layerNormCached(ctx, cache.res1, l.lnGamma, l.lnBeta, ns, dm)

	// FFN: Linear1 + bias1.
	ffnLinOut, err := nodes.ffn1.Forward(ctx, cache.normed)
	panicOnErr("FFN1 forward", err)
	b1, err := tensor.New[float32]([]int{1, ffnDim}, l.ffnB1)
	panicOnErr("ffnB1 tensor", err)
	cache.ffnPre, err = cpuEngine.Add(ctx, ffnLinOut, b1)
	panicOnErr("ffnB1 add", err)

	// GELU.
	cache.ffnAct, err = nodes.gelu.Forward(ctx, cache.ffnPre)
	panicOnErr("GELU forward", err)

	// Linear2 + bias2.
	ffn2LinOut, err := nodes.ffn2.Forward(ctx, cache.ffnAct)
	panicOnErr("FFN2 forward", err)
	b2, err := tensor.New[float32]([]int{1, dm}, l.ffnB2)
	panicOnErr("ffnB2 tensor", err)
	cache.ffnOutT, err = cpuEngine.Add(ctx, ffn2LinOut, b2)
	panicOnErr("ffnB2 add", err)

	// Residual: normed + ffnOut.
	cache.res2, err = cpuEngine.Add(ctx, cache.normed, cache.ffnOutT)
	panicOnErr("res2", err)

	// LayerNorm 2 (manual).
	var outT *tensor.TensorNumeric[float32]
	outT, cache.ln2NormedInput, cache.ln2Std = layerNormCached(ctx, cache.res2, l.ffnGamma, l.ffnBeta, ns, dm)

	return tensorToSlices(outT, ns, dm), cache
}

// backwardLayer computes gradients for one transformer layer using Backward
// methods of Linear, Gelu, and SDPA nodes, plus manual LayerNorm backward.
// dx is [ns][dm]. Returns gradient w.r.t. input and accumulates into dl.
func (m *Model) backwardLayer(dx [][]float32, cache *cpuLayerCache, l *layer, dl *layer) [][]float32 {
	ns := m.config.NSources
	dm := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dm / nHeads
	ffnDim := dm * 4
	ctx := context.Background()
	mode := types.FullBackprop

	nodes, err := buildLayerNodes(l, dm, nHeads)
	panicOnErr("backwardLayer: build nodes", err)

	// Replay forward through nodes to populate their internal caches.
	replayForward(ctx, nodes, cache, l, ns, dm, ffnDim)

	dxT := slicesToTensor(dx, ns, dm)

	// ---- LN2 backward (manual) ----
	dRes2 := layerNormBackward(ctx, dxT, cache.ln2NormedInput, cache.ln2Std,
		l.ffnGamma, dl.ffnGamma, dl.ffnBeta, ns, dm)

	// Residual: res2 = normed + ffnOutT → both get dRes2.
	dFFNOutB := dRes2

	// Bias2 backward.
	dBias2, err := cpuEngine.ReduceSum(ctx, dFFNOutB, 0, false)
	panicOnErr("dBias2", err)
	addToSlice(dl.ffnB2, dBias2.Data())

	// FFN2 backward (Linear node).
	dFFNAct, err := nodes.ffn2.Backward(ctx, mode, dFFNOutB, cache.ffnAct)
	panicOnErr("FFN2 backward", err)

	// GELU backward.
	dFFNPre, err := nodes.gelu.Backward(ctx, mode, dFFNAct[0])
	panicOnErr("GELU backward", err)

	// Bias1 backward.
	dBias1, err := cpuEngine.ReduceSum(ctx, dFFNPre[0], 0, false)
	panicOnErr("dBias1", err)
	addToSlice(dl.ffnB1, dBias1.Data())

	// FFN1 backward.
	dNormedFFN, err := nodes.ffn1.Backward(ctx, mode, dFFNPre[0], cache.normed)
	panicOnErr("FFN1 backward", err)

	// Combine: dNormed = dNormedFFN + dRes2 (from residual).
	dNormed, err := cpuEngine.Add(ctx, dNormedFFN[0], dRes2)
	panicOnErr("dNormed combine", err)

	// ---- LN1 backward (manual) ----
	dRes1 := layerNormBackward(ctx, dNormed, cache.ln1NormedInput, cache.ln1Std,
		l.lnGamma, dl.lnGamma, dl.lnBeta, ns, dm)

	// Residual: res1 = xIn + projOut → both get dRes1.
	dProjOut := dRes1

	// Output projection backward.
	dConcat, err := nodes.outProj.Backward(ctx, mode, dProjOut, cache.concat)
	panicOnErr("out proj backward", err)

	// Reshape dConcat [ns, dm] → [nHeads, ns, headDim].
	dConcatHeads := reshapeForHeads(dConcat[0], ns, nHeads, headDim)

	// SDPA backward → [dQ, dK, dV] in [nHeads, ns, headDim].
	dQKV, err := nodes.sdpa.Backward(ctx, mode, dConcatHeads, nil, nil, nil)
	panicOnErr("SDPA backward", err)

	// Reshape back to [ns, dm].
	dQ := reshapeFromHeads(dQKV[0], ns, nHeads, headDim)
	dK := reshapeFromHeads(dQKV[1], ns, nHeads, headDim)
	dV := reshapeFromHeads(dQKV[2], ns, nHeads, headDim)

	// Q/K/V projection backward.
	dXQ, err := nodes.qProj.Backward(ctx, mode, dQ, cache.xIn)
	panicOnErr("Q backward", err)
	dXK, err := nodes.kProj.Backward(ctx, mode, dK, cache.xIn)
	panicOnErr("K backward", err)
	dXV, err := nodes.vProj.Backward(ctx, mode, dV, cache.xIn)
	panicOnErr("V backward", err)

	// Combine: dX = dXQ + dXK + dXV + dRes1 (residual).
	dXTotal, err := cpuEngine.Add(ctx, dXQ[0], dXK[0])
	panicOnErr("dX combine", err)
	dXTotal, err = cpuEngine.Add(ctx, dXTotal, dXV[0])
	panicOnErr("dX combine", err)
	dXTotal, err = cpuEngine.Add(ctx, dXTotal, dRes1)
	panicOnErr("dX combine", err)

	// Extract weight gradients from Linear nodes into dl.
	extractLinearGrad(nodes.qProj, dl.qW)
	extractLinearGrad(nodes.kProj, dl.kW)
	extractLinearGrad(nodes.vProj, dl.vW)
	extractLinearGrad(nodes.outProj, dl.outW)
	extractLinearGrad(nodes.ffn1, dl.ffnW1)
	extractLinearGrad(nodes.ffn2, dl.ffnW2)

	return tensorToSlices(dXTotal, ns, dm)
}

// replayForward re-runs forward through nodes to populate their internal
// caches for backward (e.g., Linear caches its input, Gelu caches input).
func replayForward(
	ctx context.Context,
	nodes *cpuLayerNodes,
	cache *cpuLayerCache,
	l *layer,
	ns, dm, ffnDim int,
) {
	var err error

	_, err = nodes.qProj.Forward(ctx, cache.xIn)
	panicOnErr("replay Q", err)
	_, err = nodes.kProj.Forward(ctx, cache.xIn)
	panicOnErr("replay K", err)
	_, err = nodes.vProj.Forward(ctx, cache.xIn)
	panicOnErr("replay V", err)
	_, err = nodes.sdpa.Forward(ctx, cache.qHeads, cache.kHeads, cache.vHeads, nil)
	panicOnErr("replay SDPA", err)
	_, err = nodes.outProj.Forward(ctx, cache.concat)
	panicOnErr("replay out proj", err)
	_, err = nodes.ffn1.Forward(ctx, cache.normed)
	panicOnErr("replay FFN1", err)
	_, err = nodes.gelu.Forward(ctx, cache.ffnPre)
	panicOnErr("replay GELU", err)
	_, err = nodes.ffn2.Forward(ctx, cache.ffnAct)
	panicOnErr("replay FFN2", err)
}

// layerNormCached computes layer normalization using the engine and returns
// (output, normedInput, std) for use in manual backward.
func layerNormCached(
	ctx context.Context,
	x *tensor.TensorNumeric[float32],
	gamma, beta []float32,
	ns, dm int,
) (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) {
	const eps = 1e-5

	// mean = ReduceSum(x, axis=1, keepDims=true) / dm
	sum, err := cpuEngine.ReduceSum(ctx, x, 1, true)
	panicOnErr("LN sum", err)
	mean, err := cpuEngine.DivScalar(ctx, sum, float32(dm))
	panicOnErr("LN mean", err)

	// xMinusMean = x - mean
	xMM, err := cpuEngine.Sub(ctx, x, mean)
	panicOnErr("LN sub", err)

	// var = ReduceSum(xMM^2, axis=1, keepDims=true) / dm
	sq, err := cpuEngine.Mul(ctx, xMM, xMM)
	panicOnErr("LN sq", err)
	varSum, err := cpuEngine.ReduceSum(ctx, sq, 1, true)
	panicOnErr("LN varSum", err)
	variance, err := cpuEngine.DivScalar(ctx, varSum, float32(dm))
	panicOnErr("LN var", err)

	// std = sqrt(var + eps)
	varEps, err := cpuEngine.AddScalar(ctx, variance, eps)
	panicOnErr("LN varEps", err)
	std, err := cpuEngine.Sqrt(ctx, varEps)
	panicOnErr("LN std", err)

	// normedInput = xMM / std
	normed, err := cpuEngine.Div(ctx, xMM, std)
	panicOnErr("LN normed", err)

	// output = normed * gamma + beta
	gT, err := tensor.New[float32]([]int{1, dm}, gamma)
	panicOnErr("LN gamma", err)
	bT, err := tensor.New[float32]([]int{1, dm}, beta)
	panicOnErr("LN beta", err)
	scaled, err := cpuEngine.Mul(ctx, normed, gT)
	panicOnErr("LN scale", err)
	out, err := cpuEngine.Add(ctx, scaled, bT)
	panicOnErr("LN shift", err)

	return out, normed, std
}

// layerNormBackward computes the gradient through layer normalization.
// Accumulates gamma/beta gradients into dGamma/dBeta slices.
// Returns dX of shape [ns, dm].
func layerNormBackward(
	ctx context.Context,
	dOut *tensor.TensorNumeric[float32],
	normedInput *tensor.TensorNumeric[float32], // cached (x-mean)/std
	std *tensor.TensorNumeric[float32], // cached sqrt(var+eps) [ns, 1]
	gamma []float32,
	dGamma, dBeta []float32,
	ns, dm int,
) *tensor.TensorNumeric[float32] {
	gT, err := tensor.New[float32]([]int{1, dm}, gamma)
	panicOnErr("LN bwd gamma", err)

	// dGamma += sum(dOut * normedInput, axis=0)
	dOutNorm, err := cpuEngine.Mul(ctx, dOut, normedInput)
	panicOnErr("LN bwd dOutNorm", err)
	dG, err := cpuEngine.ReduceSum(ctx, dOutNorm, 0, false)
	panicOnErr("LN bwd dGamma", err)
	addToSlice(dGamma, dG.Data())

	// dBeta += sum(dOut, axis=0)
	dB, err := cpuEngine.ReduceSum(ctx, dOut, 0, false)
	panicOnErr("LN bwd dBeta", err)
	addToSlice(dBeta, dB.Data())

	// dNormed = dOut * gamma
	dNormed, err := cpuEngine.Mul(ctx, dOut, gT)
	panicOnErr("LN bwd dNormed", err)

	n := float32(dm)

	// Standard LayerNorm backward:
	// dX = (1/std) * (dNormed - mean(dNormed) - normedInput * mean(dNormed * normedInput))
	//
	// Where mean() is along the feature axis (axis=1).

	// mean(dNormed, axis=1, keepDims=true)
	sumDN, err := cpuEngine.ReduceSum(ctx, dNormed, 1, true)
	panicOnErr("LN bwd sumDN", err)
	meanDN, err := cpuEngine.DivScalar(ctx, sumDN, n)
	panicOnErr("LN bwd meanDN", err)

	// mean(dNormed * normedInput, axis=1, keepDims=true)
	dNorm2, err := cpuEngine.Mul(ctx, dNormed, normedInput)
	panicOnErr("LN bwd dNorm2", err)
	sumDN2, err := cpuEngine.ReduceSum(ctx, dNorm2, 1, true)
	panicOnErr("LN bwd sumDN2", err)
	meanDN2, err := cpuEngine.DivScalar(ctx, sumDN2, n)
	panicOnErr("LN bwd meanDN2", err)

	// normedInput * mean(dNormed * normedInput)
	termB, err := cpuEngine.Mul(ctx, normedInput, meanDN2)
	panicOnErr("LN bwd termB", err)

	// dNormed - meanDN - termB
	sub1, err := cpuEngine.Sub(ctx, dNormed, meanDN)
	panicOnErr("LN bwd sub1", err)
	sub2, err := cpuEngine.Sub(ctx, sub1, termB)
	panicOnErr("LN bwd sub2", err)

	// dX = sub2 / std
	dX, err := cpuEngine.Div(ctx, sub2, std)
	panicOnErr("LN bwd dX", err)

	return dX
}

// reshapeForHeads converts [ns, dm] → [nHeads, ns, headDim].
func reshapeForHeads(t *tensor.TensorNumeric[float32], ns, nHeads, headDim int) *tensor.TensorNumeric[float32] {
	data := t.Data()
	dm := nHeads * headDim
	out := make([]float32, nHeads*ns*headDim)
	for s := range ns {
		for h := range nHeads {
			copy(out[h*ns*headDim+s*headDim:], data[s*dm+h*headDim:s*dm+h*headDim+headDim])
		}
	}
	r, err := tensor.New[float32]([]int{nHeads, ns, headDim}, out)
	panicOnErr("reshapeForHeads", err)
	return r
}

// reshapeFromHeads converts [nHeads, ns, headDim] → [ns, dm].
func reshapeFromHeads(t *tensor.TensorNumeric[float32], ns, nHeads, headDim int) *tensor.TensorNumeric[float32] {
	data := t.Data()
	dm := nHeads * headDim
	out := make([]float32, ns*dm)
	for s := range ns {
		for h := range nHeads {
			copy(out[s*dm+h*headDim:], data[h*ns*headDim+s*headDim:h*ns*headDim+s*headDim+headDim])
		}
	}
	r, err := tensor.New[float32]([]int{ns, dm}, out)
	panicOnErr("reshapeFromHeads", err)
	return r
}

// extractLinearGrad accumulates gradient data from a Linear node's weight
// parameter into the flat gradient slice.
func extractLinearGrad(lin *core.Linear[float32], dst []float32) {
	for _, p := range lin.Parameters() {
		addToSlice(dst, p.Gradient.Data())
		p.ClearGradient()
	}
}

// addToSlice adds src element-wise into dst.
func addToSlice(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// slicesToTensor converts [][]float32 [n][m] to a tensor [n, m].
func slicesToTensor(s [][]float32, n, m int) *tensor.TensorNumeric[float32] {
	flat := make([]float32, n*m)
	for i := range n {
		copy(flat[i*m:], s[i])
	}
	t, err := tensor.New[float32]([]int{n, m}, flat)
	panicOnErr("slicesToTensor", err)
	return t
}

// tensorToSlices converts a tensor [n, m] to [][]float32 [n][m].
func tensorToSlices(t *tensor.TensorNumeric[float32], n, m int) [][]float32 {
	data := t.Data()
	out := make([][]float32, n)
	for i := range n {
		out[i] = make([]float32, m)
		copy(out[i], data[i*m:(i+1)*m])
	}
	return out
}

func panicOnErr(label string, err error) {
	if err != nil {
		panic(fmt.Sprintf("crossasset.%s: %v", label, err))
	}
}

// zeroLayer creates a layer with all-zero weights of the correct dimensions.
func zeroLayer(dm int) layer {
	ffnDim := dm * 4
	return layer{
		qW:       make([]float32, dm*dm),
		kW:       make([]float32, dm*dm),
		vW:       make([]float32, dm*dm),
		outW:     make([]float32, dm*dm),
		lnGamma:  make([]float32, dm),
		lnBeta:   make([]float32, dm),
		ffnW1:    make([]float32, dm*ffnDim),
		ffnB1:    make([]float32, ffnDim),
		ffnW2:    make([]float32, ffnDim*dm),
		ffnB2:    make([]float32, dm),
		ffnGamma: make([]float32, dm),
		ffnBeta:  make([]float32, dm),
	}
}
