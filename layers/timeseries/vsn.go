package timeseries

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// GRN implements a Gated Residual Network:
//
//	GRN(x) = LayerNorm(x + ELU(W1*x + b1) * sigmoid(W2*x + b2))
//
// where LayerNorm is approximated as mean-subtraction and variance-normalization.
type GRN[T tensor.Numeric] struct {
	name   string
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	w1     *graph.Parameter[T] // [inputDim, hiddenDim]
	b1     *graph.Parameter[T] // [1, hiddenDim]
	w2     *graph.Parameter[T] // [inputDim, hiddenDim]
	b2     *graph.Parameter[T] // [1, hiddenDim]
	wOut   *graph.Parameter[T] // [hiddenDim, outputDim]
	ln     *normalization.LayerNormalization[T]

	inputDim  int
	hiddenDim int
	outputDim int
}

// NewGRN creates a new Gated Residual Network layer.
func NewGRN[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputDim, hiddenDim, outputDim int,
) (*GRN[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if inputDim <= 0 || hiddenDim <= 0 || outputDim <= 0 {
		return nil, fmt.Errorf("dimensions must be positive, got input=%d hidden=%d output=%d", inputDim, hiddenDim, outputDim)
	}

	makeParam := func(suffix string, shape []int) (*graph.Parameter[T], error) {
		data := make([]T, shape[0]*shape[1])
		scale := 1.0 / math.Sqrt(float64(shape[0]))
		for i := range data {
			data[i] = T(rand.Float64() * scale)
		}
		t, err := tensor.New[T](shape, data)
		if err != nil {
			return nil, err
		}
		return graph.NewParameter[T](name+"_"+suffix, t, tensor.New[T])
	}

	w1, err := makeParam("w1", []int{inputDim, hiddenDim})
	if err != nil {
		return nil, err
	}
	b1, err := makeParam("b1", []int{1, hiddenDim})
	if err != nil {
		return nil, err
	}
	w2, err := makeParam("w2", []int{inputDim, hiddenDim})
	if err != nil {
		return nil, err
	}
	b2, err := makeParam("b2", []int{1, hiddenDim})
	if err != nil {
		return nil, err
	}
	wOut, err := makeParam("wout", []int{hiddenDim, outputDim})
	if err != nil {
		return nil, err
	}

	ln, err := normalization.NewLayerNormalization[T](engine, outputDim)
	if err != nil {
		return nil, fmt.Errorf("grn layernorm: %w", err)
	}

	return &GRN[T]{
		name:      name,
		engine:    engine,
		ops:       ops,
		w1:        w1,
		b1:        b1,
		w2:        w2,
		b2:        b2,
		wOut:      wOut,
		ln:        ln,
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		outputDim: outputDim,
	}, nil
}

// Forward computes GRN(x) = LayerNorm(residual + ELU(W1*x + b1) * sigmoid(W2*x + b2))
// projected through wOut to outputDim.
// Input x: [batch, inputDim]. Output: [batch, outputDim].
func (g *GRN[T]) Forward(ctx context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// h1 = W1*x + b1
	h1, err := g.engine.MatMul(ctx, x, g.w1.Value)
	if err != nil {
		return nil, fmt.Errorf("grn w1 matmul: %w", err)
	}
	h1, err = g.engine.Add(ctx, h1, g.b1.Value)
	if err != nil {
		return nil, fmt.Errorf("grn b1 add: %w", err)
	}

	// h2 = W2*x + b2
	h2, err := g.engine.MatMul(ctx, x, g.w2.Value)
	if err != nil {
		return nil, fmt.Errorf("grn w2 matmul: %w", err)
	}
	h2, err = g.engine.Add(ctx, h2, g.b2.Value)
	if err != nil {
		return nil, fmt.Errorf("grn b2 add: %w", err)
	}

	// ELU(h1)
	eluH1, err := g.engine.UnaryOp(ctx, h1, func(v T) T {
		if float64(v) >= 0 {
			return v
		}
		return T(math.Exp(float64(v)) - 1)
	})
	if err != nil {
		return nil, fmt.Errorf("grn elu: %w", err)
	}

	// sigmoid(h2) — delegate to canonical Sigmoid Node (T124.2.3) so the
	// math is shared with layers/activations.
	sigH2, err := activations.NewSigmoid(g.engine, g.ops).Forward(ctx, h2)
	if err != nil {
		return nil, fmt.Errorf("grn sigmoid: %w", err)
	}

	// gated = ELU(h1) * sigmoid(h2)
	gated, err := g.engine.Mul(ctx, eluH1, sigH2)
	if err != nil {
		return nil, fmt.Errorf("grn gate mul: %w", err)
	}

	// Project gated to outputDim: [batch, hiddenDim] @ [hiddenDim, outputDim] -> [batch, outputDim]
	projected, err := g.engine.MatMul(ctx, gated, g.wOut.Value)
	if err != nil {
		return nil, fmt.Errorf("grn wout matmul: %w", err)
	}

	// Residual connection: if inputDim == outputDim, add x; otherwise skip residual.
	var res *tensor.TensorNumeric[T]
	if g.inputDim == g.outputDim {
		res, err = g.engine.Add(ctx, projected, x)
		if err != nil {
			return nil, fmt.Errorf("grn residual add: %w", err)
		}
	} else {
		res = projected
	}

	// LayerNorm: normalize over the last dimension.
	normalized, err := g.ln.Forward(ctx, res)
	if err != nil {
		return nil, fmt.Errorf("grn layernorm: %w", err)
	}

	return normalized, nil
}

// Parameters returns the trainable parameters.
func (g *GRN[T]) Parameters() []*graph.Parameter[T] {
	params := []*graph.Parameter[T]{g.w1, g.b1, g.w2, g.b2, g.wOut}
	params = append(params, g.ln.Parameters()...)
	return params
}

// VSN implements a Variable Selection Network for the Temporal Fusion Transformer.
//
// Each of N input variables is projected to d_model via a learned linear projection.
// The flat concatenation of all variable embeddings is passed through a GRN and
// softmax to produce N importance weights. The output is the weighted sum of the
// variable embeddings.
type VSN[T tensor.Numeric] struct {
	name      string
	engine    compute.Engine[T]
	ops       numeric.Arithmetic[T]
	numVars   int
	dModel    int
	varProj   []*graph.Parameter[T] // one [varInputDim, dModel] per variable
	selectGRN *GRN[T]

	varInputDim int // dimension of each input variable
}

// NewVSN creates a new Variable Selection Network.
// numVars is the number of input variables.
// varInputDim is the input dimension of each variable.
// dModel is the projection/output dimension.
func NewVSN[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	numVars, varInputDim, dModel int,
) (*VSN[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if numVars <= 0 {
		return nil, fmt.Errorf("numVars must be positive, got %d", numVars)
	}
	if varInputDim <= 0 {
		return nil, fmt.Errorf("varInputDim must be positive, got %d", varInputDim)
	}
	if dModel <= 0 {
		return nil, fmt.Errorf("dModel must be positive, got %d", dModel)
	}

	varProj := make([]*graph.Parameter[T], numVars)
	scale := 1.0 / math.Sqrt(float64(varInputDim))
	for i := 0; i < numVars; i++ {
		data := make([]T, varInputDim*dModel)
		for j := range data {
			data[j] = T(rand.Float64() * scale)
		}
		t, err := tensor.New[T]([]int{varInputDim, dModel}, data)
		if err != nil {
			return nil, err
		}
		p, err := graph.NewParameter[T](fmt.Sprintf("%s_var%d_proj", name, i), t, tensor.New[T])
		if err != nil {
			return nil, err
		}
		varProj[i] = p
	}

	// GRN takes the flat concatenation of all variable embeddings (numVars * dModel)
	// and outputs numVars importance weights.
	grnInputDim := numVars * dModel
	grnHiddenDim := dModel
	selectGRN, err := NewGRN[T](name+"_select_grn", engine, ops, grnInputDim, grnHiddenDim, numVars)
	if err != nil {
		return nil, err
	}

	return &VSN[T]{
		name:        name,
		engine:      engine,
		ops:         ops,
		numVars:     numVars,
		dModel:      dModel,
		varProj:     varProj,
		selectGRN:   selectGRN,
		varInputDim: varInputDim,
	}, nil
}

// OpType returns the operation type of the layer.
func (v *VSN[T]) OpType() string {
	return "VSN"
}

// Attributes returns the attributes of the layer.
func (v *VSN[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_vars":      v.numVars,
		"var_input_dim": v.varInputDim,
		"d_model":       v.dModel,
	}
}

// OutputShape returns the output shape of the layer.
func (v *VSN[T]) OutputShape() []int {
	return []int{-1, v.dModel} // [batch, d_model]
}

// Forward computes the variable selection network.
// inputs is a slice of N tensors, each [batch, varInputDim].
// Returns (weighted_embedding [batch, dModel], importance_weights [numVars], error).
func (v *VSN[T]) Forward(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], []float32, error) {
	if len(inputs) != v.numVars {
		return nil, nil, fmt.Errorf("VSN expects %d inputs, got %d", v.numVars, len(inputs))
	}

	batch := inputs[0].Shape()[0]

	// Project each variable to d_model and collect embeddings.
	embeddings := make([]*tensor.TensorNumeric[T], v.numVars)
	for i, inp := range inputs {
		shape := inp.Shape()
		if len(shape) != 2 || shape[0] != batch || shape[1] != v.varInputDim {
			return nil, nil, fmt.Errorf("input[%d] shape %v, want [%d, %d]", i, shape, batch, v.varInputDim)
		}
		// [batch, varInputDim] @ [varInputDim, dModel] -> [batch, dModel]
		emb, err := v.engine.MatMul(ctx, inp, v.varProj[i].Value)
		if err != nil {
			return nil, nil, fmt.Errorf("var %d projection: %w", i, err)
		}
		embeddings[i] = emb
	}

	// Concatenate embeddings: [batch, numVars * dModel]
	flatEmb, err := v.engine.Concat(ctx, embeddings, 1)
	if err != nil {
		return nil, nil, fmt.Errorf("concat embeddings: %w", err)
	}

	// GRN -> [batch, numVars]
	grnOut, err := v.selectGRN.Forward(ctx, flatEmb)
	if err != nil {
		return nil, nil, fmt.Errorf("select grn: %w", err)
	}

	// Softmax along last axis -> importance weights [batch, numVars]
	weights, err := v.engine.Softmax(ctx, grnOut, 1)
	if err != nil {
		return nil, nil, fmt.Errorf("softmax weights: %w", err)
	}

	// Split weights [batch, numVars] along axis 1 into numVars * [batch, 1].
	weightCols, err := v.engine.Split(ctx, weights, v.numVars, 1)
	if err != nil {
		return nil, nil, fmt.Errorf("split weights: %w", err)
	}

	// Compute weighted sum of embeddings using engine ops.
	outData := make([]T, batch*v.dModel)
	output, err := tensor.New[T]([]int{batch, v.dModel}, outData)
	if err != nil {
		return nil, nil, err
	}

	for i := 0; i < v.numVars; i++ {
		// weightCols[i] is [batch, 1], broadcast-multiply with embedding[i] [batch, dModel]
		scaled, err := v.engine.Mul(ctx, weightCols[i], embeddings[i])
		if err != nil {
			return nil, nil, fmt.Errorf("scale var %d: %w", i, err)
		}
		output, err = v.engine.Add(ctx, output, scaled)
		if err != nil {
			return nil, nil, fmt.Errorf("accumulate var %d: %w", i, err)
		}
	}

	// Compute mean importance weights across batch via engine ReduceMean.
	meanWeights, err := v.engine.ReduceMean(ctx, weights, 0, false) // [numVars]
	if err != nil {
		return nil, nil, fmt.Errorf("mean weights: %w", err)
	}
	mwData := meanWeights.Data()
	importanceWeights := make([]float32, v.numVars)
	for i := range importanceWeights {
		importanceWeights[i] = float32(mwData[i])
	}

	return output, importanceWeights, nil
}

// Backward computes gradients for the VSN layer.
func (v *VSN[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != v.numVars {
		return nil, fmt.Errorf("VSN backward expects %d inputs, got %d", v.numVars, len(inputs))
	}

	// Recompute forward embeddings and weights for backward.
	embeddings := make([]*tensor.TensorNumeric[T], v.numVars)
	for i, inp := range inputs {
		emb, err := v.engine.MatMul(ctx, inp, v.varProj[i].Value)
		if err != nil {
			return nil, err
		}
		embeddings[i] = emb
	}

	flatEmb, err := v.engine.Concat(ctx, embeddings, 1)
	if err != nil {
		return nil, err
	}

	grnOut, err := v.selectGRN.Forward(ctx, flatEmb)
	if err != nil {
		return nil, err
	}

	weights, err := v.engine.Softmax(ctx, grnOut, 1)
	if err != nil {
		return nil, err
	}

	// Split weights for backward.
	weightCols, err := v.engine.Split(ctx, weights, v.numVars, 1) // numVars * [batch, 1]
	if err != nil {
		return nil, err
	}

	// Gradient w.r.t. each variable projection.
	inputGrads := make([]*tensor.TensorNumeric[T], v.numVars)
	for i := 0; i < v.numVars; i++ {
		// dEmb_i = weight_i * outputGradient
		dEmb, err := v.engine.Mul(ctx, weightCols[i], outputGradient)
		if err != nil {
			return nil, err
		}

		// Gradient w.r.t. varProj[i]: input[i]^T @ dEmb
		inputT, err := v.engine.Transpose(ctx, inputs[i], []int{1, 0})
		if err != nil {
			return nil, err
		}
		dw, err := v.engine.MatMul(ctx, inputT, dEmb)
		if err != nil {
			return nil, err
		}
		v.varProj[i].Gradient, err = v.engine.Add(ctx, v.varProj[i].Gradient, dw)
		if err != nil {
			return nil, err
		}

		// Gradient w.r.t. input[i]: dEmb @ varProj[i]^T
		projT, err := v.engine.Transpose(ctx, v.varProj[i].Value, []int{1, 0})
		if err != nil {
			return nil, err
		}
		dx, err := v.engine.MatMul(ctx, dEmb, projT)
		if err != nil {
			return nil, err
		}
		inputGrads[i] = dx
	}

	return inputGrads, nil
}

// Parameters returns the trainable parameters.
func (v *VSN[T]) Parameters() []*graph.Parameter[T] {
	params := make([]*graph.Parameter[T], 0, len(v.varProj)+len(v.selectGRN.Parameters()))
	params = append(params, v.varProj...)
	params = append(params, v.selectGRN.Parameters()...)
	return params
}

// SetName sets the name of the layer.
func (v *VSN[T]) SetName(name string) {
	v.name = name
	for i, p := range v.varProj {
		p.Name = fmt.Sprintf("%s_var%d_proj", name, i)
	}
}

// Name returns the name of the layer.
func (v *VSN[T]) Name() string {
	return v.name
}
