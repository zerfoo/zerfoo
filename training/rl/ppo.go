package rl

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// PPOConfig holds hyperparameters for the PPO agent.
type PPOConfig struct {
	StateDim     int
	ActionDim    int
	HiddenDim    int
	ClipRatio    float64
	Gamma        float64
	Lambda       float64
	NEpochs      int
	BatchSize    int
	LearningRate float64
}

// DefaultPPOConfig returns a PPOConfig with sensible defaults.
func DefaultPPOConfig(stateDim, actionDim int) PPOConfig {
	return PPOConfig{
		StateDim:     stateDim,
		ActionDim:    actionDim,
		HiddenDim:    64,
		ClipRatio:    0.2,
		Gamma:        0.99,
		Lambda:       0.95,
		NEpochs:      4,
		BatchSize:    64,
		LearningRate: 3e-4,
	}
}

// linearLayer holds parameters for a single dense layer used with functional.Linear.
type linearLayer struct {
	weight *graph.Parameter[float64] // [outDim, inDim]
	bias   *graph.Parameter[float64] // [outDim]
	inDim  int
	outDim int
}

func newLinearLayer(name string, inDim, outDim int) (*linearLayer, error) {
	scale := math.Sqrt(2.0 / float64(inDim+outDim))
	wData := make([]float64, outDim*inDim)
	for i := range wData {
		wData[i] = rand.NormFloat64() * scale
	}
	wTensor, err := tensor.New[float64]([]int{outDim, inDim}, wData)
	if err != nil {
		return nil, fmt.Errorf("rl: create weight tensor: %w", err)
	}
	wParam, err := graph.NewParameter[float64](name+".weight", wTensor, tensor.New[float64])
	if err != nil {
		return nil, fmt.Errorf("rl: create weight parameter: %w", err)
	}

	bData := make([]float64, outDim)
	bTensor, err := tensor.New[float64]([]int{outDim}, bData)
	if err != nil {
		return nil, fmt.Errorf("rl: create bias tensor: %w", err)
	}
	bParam, err := graph.NewParameter[float64](name+".bias", bTensor, tensor.New[float64])
	if err != nil {
		return nil, fmt.Errorf("rl: create bias parameter: %w", err)
	}

	return &linearLayer{weight: wParam, bias: bParam, inDim: inDim, outDim: outDim}, nil
}

// forward computes y = functional.Linear(x, weight, bias).
func (l *linearLayer) forward(ctx context.Context, engine compute.Engine[float64],
	x *tensor.TensorNumeric[float64]) (*tensor.TensorNumeric[float64], error) {
	return functional.Linear(ctx, engine, x, l.weight.Value, l.bias.Value)
}

// params returns the trainable parameters.
func (l *linearLayer) params() []*graph.Parameter[float64] {
	return []*graph.Parameter[float64]{l.weight, l.bias}
}

// policyNet is a 2-layer MLP that outputs action means and log standard deviations.
type policyNet struct {
	hidden  *linearLayer
	mean    *linearLayer
	logStdP *graph.Parameter[float64] // learnable per-action-dim log std (state-independent)
	engine  compute.Engine[float64]
}

func newPolicyNet(engine compute.Engine[float64], stateDim, hiddenDim, actionDim int) (*policyNet, error) {
	hidden, err := newLinearLayer("policy.hidden", stateDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	mean, err := newLinearLayer("policy.mean", hiddenDim, actionDim)
	if err != nil {
		return nil, err
	}

	logStdData := make([]float64, actionDim)
	for i := range logStdData {
		logStdData[i] = -0.5 // initial std ~ 0.6
	}
	logStdTensor, err := tensor.New[float64]([]int{actionDim}, logStdData)
	if err != nil {
		return nil, err
	}
	logStdParam, err := graph.NewParameter[float64]("policy.logStd", logStdTensor, tensor.New[float64])
	if err != nil {
		return nil, err
	}

	return &policyNet{
		hidden:  hidden,
		mean:    mean,
		logStdP: logStdParam,
		engine:  engine,
	}, nil
}

// policyForwardResult stores intermediate activations for backprop.
type policyForwardResult struct {
	input      *tensor.TensorNumeric[float64]
	hiddenPre  *tensor.TensorNumeric[float64]
	hiddenPost *tensor.TensorNumeric[float64]
	meanOut    *tensor.TensorNumeric[float64]
	logStd     []float64
}

func (p *policyNet) forwardFull(ctx context.Context, state []float64) (policyForwardResult, error) {
	xT, err := tensor.New[float64]([]int{1, len(state)}, state)
	if err != nil {
		return policyForwardResult{}, err
	}
	hiddenPre, err := p.hidden.forward(ctx, p.engine, xT)
	if err != nil {
		return policyForwardResult{}, err
	}
	hiddenPost, err := p.engine.Tanh(ctx, hiddenPre)
	if err != nil {
		return policyForwardResult{}, err
	}
	meanOut, err := p.mean.forward(ctx, p.engine, hiddenPost)
	if err != nil {
		return policyForwardResult{}, err
	}
	ls := make([]float64, len(p.logStdP.Value.Data()))
	copy(ls, p.logStdP.Value.Data())
	return policyForwardResult{
		input: xT, hiddenPre: hiddenPre, hiddenPost: hiddenPost, meanOut: meanOut, logStd: ls,
	}, nil
}

func (p *policyNet) forward(ctx context.Context, state []float64) (mean, logStd []float64, err error) {
	r, err := p.forwardFull(ctx, state)
	if err != nil {
		return nil, nil, err
	}
	return r.meanOut.Data(), r.logStd, nil
}

func (p *policyNet) params() []*graph.Parameter[float64] {
	params := p.hidden.params()
	params = append(params, p.mean.params()...)
	params = append(params, p.logStdP)
	return params
}

// valueNet is a 2-layer MLP that outputs a scalar state value.
type valueNet struct {
	hidden *linearLayer
	out    *linearLayer
	engine compute.Engine[float64]
}

func newValueNet(engine compute.Engine[float64], stateDim, hiddenDim int) (*valueNet, error) {
	hidden, err := newLinearLayer("value.hidden", stateDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	out, err := newLinearLayer("value.out", hiddenDim, 1)
	if err != nil {
		return nil, err
	}
	return &valueNet{hidden: hidden, out: out, engine: engine}, nil
}

type valueForwardResult struct {
	input      *tensor.TensorNumeric[float64]
	hiddenPre  *tensor.TensorNumeric[float64]
	hiddenPost *tensor.TensorNumeric[float64]
	value      float64
}

func (v *valueNet) forwardFull(ctx context.Context, state []float64) (valueForwardResult, error) {
	xT, err := tensor.New[float64]([]int{1, len(state)}, state)
	if err != nil {
		return valueForwardResult{}, err
	}
	hiddenPre, err := v.hidden.forward(ctx, v.engine, xT)
	if err != nil {
		return valueForwardResult{}, err
	}
	hiddenPost, err := v.engine.Tanh(ctx, hiddenPre)
	if err != nil {
		return valueForwardResult{}, err
	}
	outT, err := v.out.forward(ctx, v.engine, hiddenPost)
	if err != nil {
		return valueForwardResult{}, err
	}
	return valueForwardResult{input: xT, hiddenPre: hiddenPre, hiddenPost: hiddenPost, value: outT.Data()[0]}, nil
}

func (v *valueNet) forward(ctx context.Context, state []float64) (float64, error) {
	r, err := v.forwardFull(ctx, state)
	if err != nil {
		return 0, err
	}
	return r.value, nil
}

func (v *valueNet) params() []*graph.Parameter[float64] {
	params := v.hidden.params()
	params = append(params, v.out.params()...)
	return params
}

// PPO implements the Proximal Policy Optimization agent with clipped surrogate
// objective and Generalized Advantage Estimation (GAE).
type PPO struct {
	config    PPOConfig
	policy    *policyNet
	value     *valueNet
	engine    compute.Engine[float64]
	policyOpt *optimizer.SGD[float64]
	valueOpt  *optimizer.SGD[float64]
}

// NewPPO creates a PPO agent with the given configuration.
func NewPPO(cfg PPOConfig) *PPO {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	policy, err := newPolicyNet(engine, cfg.StateDim, cfg.HiddenDim, cfg.ActionDim)
	if err != nil {
		panic(fmt.Sprintf("rl: failed to create policy net: %v", err))
	}
	value, err := newValueNet(engine, cfg.StateDim, cfg.HiddenDim)
	if err != nil {
		panic(fmt.Sprintf("rl: failed to create value net: %v", err))
	}
	policyOpt := optimizer.NewSGD[float64](engine, numeric.Float64Ops{}, float32(cfg.LearningRate))
	valueOpt := optimizer.NewSGD[float64](engine, numeric.Float64Ops{}, float32(cfg.LearningRate))
	return &PPO{
		config: cfg, policy: policy, value: value, engine: engine,
		policyOpt: policyOpt, valueOpt: valueOpt,
	}
}

// Act selects an action by sampling from the Gaussian policy.
func (p *PPO) Act(state State) Action {
	ctx := context.Background()
	mean, logStd, err := p.policy.forward(ctx, state)
	if err != nil {
		return make(Action, p.config.ActionDim)
	}
	action := make(Action, p.config.ActionDim)
	for i := range action {
		std := math.Exp(logStd[i])
		action[i] = mean[i] + rand.NormFloat64()*std
	}
	return action
}

func logProb(action, mean, logStd []float64) float64 {
	lp := 0.0
	for i := range action {
		std := math.Exp(logStd[i])
		diff := action[i] - mean[i]
		lp += -0.5*math.Log(2*math.Pi) - logStd[i] - 0.5*(diff*diff)/(std*std)
	}
	return lp
}

func logProbGrad(action, mean, logStd []float64) (dMean, dLogStd []float64) {
	dMean = make([]float64, len(action))
	dLogStd = make([]float64, len(action))
	for i := range action {
		std := math.Exp(logStd[i])
		diff := action[i] - mean[i]
		dMean[i] = diff / (std * std)
		dLogStd[i] = (diff*diff)/(std*std) - 1.0
	}
	return dMean, dLogStd
}

// Learn performs PPO updates on the given batch of sequential experiences.
func (p *PPO) Learn(batch []Experience) error {
	n := len(batch)
	if n == 0 {
		return fmt.Errorf("rl: PPO.Learn called with empty batch")
	}
	ctx := context.Background()
	values := make([]float64, n+1)
	for i := 0; i < n; i++ {
		v, err := p.value.forward(ctx, batch[i].State)
		if err != nil {
			return fmt.Errorf("rl: PPO value forward: %w", err)
		}
		values[i] = v
	}
	if batch[n-1].Done {
		values[n] = 0
	} else {
		v, err := p.value.forward(ctx, batch[n-1].NextState)
		if err != nil {
			return fmt.Errorf("rl: PPO value forward (bootstrap): %w", err)
		}
		values[n] = v
	}
	advantages := make([]float64, n)
	returns := make([]float64, n)
	gae := 0.0
	for t := n - 1; t >= 0; t-- {
		var nextVal float64
		if batch[t].Done {
			nextVal = 0
			gae = 0
		} else {
			nextVal = values[t+1]
		}
		delta := batch[t].Reward + p.config.Gamma*nextVal - values[t]
		gae = delta + p.config.Gamma*p.config.Lambda*gae
		advantages[t] = gae
		returns[t] = gae + values[t]
	}
	advMean, advStd := meanStd(advantages)
	for i := range advantages {
		advantages[i] = (advantages[i] - advMean) / (advStd + 1e-8)
	}
	oldLogProbs := make([]float64, n)
	for i := 0; i < n; i++ {
		mean, logStd, err := p.policy.forward(ctx, batch[i].State)
		if err != nil {
			return fmt.Errorf("rl: PPO policy forward (old): %w", err)
		}
		oldLogProbs[i] = logProb(batch[i].Action, mean, logStd)
	}
	batchSize := p.config.BatchSize
	if batchSize > n {
		batchSize = n
	}
	for epoch := 0; epoch < p.config.NEpochs; epoch++ {
		indices := make([]int, n)
		for i := range indices {
			indices[i] = i
		}
		rand.Shuffle(n, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		for start := 0; start < n; start += batchSize {
			end := start + batchSize
			if end > n {
				end = n
			}
			if err := p.updateMinibatch(ctx, batch, indices[start:end], oldLogProbs, advantages, returns); err != nil {
				return fmt.Errorf("rl: PPO minibatch update (epoch %d): %w", epoch, err)
			}
		}
	}
	return nil
}

func (p *PPO) updateMinibatch(ctx context.Context, batch []Experience, indices []int,
	oldLogProbs, advantages, returns []float64) error {
	clip := p.config.ClipRatio
	mbSize := float64(len(indices))
	pol := p.policy
	val := p.value
	for _, param := range pol.params() {
		param.ClearGradient()
	}
	for _, param := range val.params() {
		param.ClearGradient()
	}
	for _, idx := range indices {
		exp := batch[idx]
		pfr, err := pol.forwardFull(ctx, exp.State)
		if err != nil {
			return err
		}
		newLP := logProb(exp.Action, pfr.meanOut.Data(), pfr.logStd)
		ratio := math.Exp(newLP - oldLogProbs[idx])
		adv := advantages[idx]
		clipped := math.Max(1-clip, math.Min(1+clip, ratio))
		surr1 := ratio * adv
		surr2 := clipped * adv
		var dRatio float64
		if surr1 <= surr2 {
			dRatio = -adv / mbSize
		} else {
			dRatio = 0
		}
		dLogProb := dRatio * ratio
		dMean, dLogStd := logProbGrad(exp.Action, pfr.meanOut.Data(), pfr.logStd)
		actionDim := pol.mean.outDim
		gradMeanOut := make([]float64, actionDim)
		for i := range gradMeanOut {
			gradMeanOut[i] = dLogProb * dMean[i]
		}
		logStdGrad := pol.logStdP.Gradient.Data()
		for i := range logStdGrad {
			logStdGrad[i] += dLogProb * dLogStd[i]
		}
		gradMeanOutT, err := tensor.New[float64]([]int{1, actionDim}, gradMeanOut)
		if err != nil {
			return err
		}
		gradHiddenPost, err := p.engine.MatMul(ctx, gradMeanOutT, pol.mean.weight.Value)
		if err != nil {
			return err
		}
		gradMeanOutCol, err := p.engine.Reshape(ctx, gradMeanOutT, []int{actionDim, 1})
		if err != nil {
			return err
		}
		dwMean, err := p.engine.MatMul(ctx, gradMeanOutCol, pfr.hiddenPost)
		if err != nil {
			return err
		}
		if err := pol.mean.weight.AddGradient(dwMean); err != nil {
			return err
		}
		gradMeanBias, err := p.engine.Reshape(ctx, gradMeanOutT, []int{actionDim})
		if err != nil {
			return err
		}
		if err := pol.mean.bias.AddGradient(gradMeanBias); err != nil {
			return err
		}
		gradHiddenPre, err := p.engine.TanhPrime(ctx, pfr.hiddenPre, gradHiddenPost)
		if err != nil {
			return err
		}
		hiddenDim := pol.hidden.outDim
		gradHiddenPreCol, err := p.engine.Reshape(ctx, gradHiddenPre, []int{hiddenDim, 1})
		if err != nil {
			return err
		}
		dwHidden, err := p.engine.MatMul(ctx, gradHiddenPreCol, pfr.input)
		if err != nil {
			return err
		}
		if err := pol.hidden.weight.AddGradient(dwHidden); err != nil {
			return err
		}
		gradHiddenBias, err := p.engine.Reshape(ctx, gradHiddenPre, []int{hiddenDim})
		if err != nil {
			return err
		}
		if err := pol.hidden.bias.AddGradient(gradHiddenBias); err != nil {
			return err
		}
		vfr, err := val.forwardFull(ctx, exp.State)
		if err != nil {
			return err
		}
		diff := vfr.value - returns[idx]
		dValue := 2.0 * diff / mbSize
		dValueT, err := tensor.New[float64]([]int{1, 1}, []float64{dValue})
		if err != nil {
			return err
		}
		gradVHiddenPost, err := p.engine.MatMul(ctx, dValueT, val.out.weight.Value)
		if err != nil {
			return err
		}
		dValueCol, err := p.engine.Reshape(ctx, dValueT, []int{1, 1})
		if err != nil {
			return err
		}
		dwOut, err := p.engine.MatMul(ctx, dValueCol, vfr.hiddenPost)
		if err != nil {
			return err
		}
		if err := val.out.weight.AddGradient(dwOut); err != nil {
			return err
		}
		dValueBias, err := p.engine.Reshape(ctx, dValueT, []int{1})
		if err != nil {
			return err
		}
		if err := val.out.bias.AddGradient(dValueBias); err != nil {
			return err
		}
		gradVHiddenPre, err := p.engine.TanhPrime(ctx, vfr.hiddenPre, gradVHiddenPost)
		if err != nil {
			return err
		}
		vHiddenDim := val.hidden.outDim
		gradVHiddenPreCol, err := p.engine.Reshape(ctx, gradVHiddenPre, []int{vHiddenDim, 1})
		if err != nil {
			return err
		}
		dwVHidden, err := p.engine.MatMul(ctx, gradVHiddenPreCol, vfr.input)
		if err != nil {
			return err
		}
		if err := val.hidden.weight.AddGradient(dwVHidden); err != nil {
			return err
		}
		gradVHiddenBias, err := p.engine.Reshape(ctx, gradVHiddenPre, []int{vHiddenDim})
		if err != nil {
			return err
		}
		if err := val.hidden.bias.AddGradient(gradVHiddenBias); err != nil {
			return err
		}
	}
	clipGradNorm(1.0, collectGradSlices(pol.params())...)
	clipGradNorm(1.0, collectGradSlices(val.params())...)
	if err := p.policyOpt.Step(ctx, pol.params()); err != nil {
		return fmt.Errorf("rl: PPO policy SGD step: %w", err)
	}
	if err := p.valueOpt.Step(ctx, val.params()); err != nil {
		return fmt.Errorf("rl: PPO value SGD step: %w", err)
	}
	return nil
}

func collectGradSlices(params []*graph.Parameter[float64]) [][]float64 {
	slices := make([][]float64, len(params))
	for i, p := range params {
		slices[i] = p.Gradient.Data()
	}
	return slices
}

func clipGradNorm(maxNorm float64, grads ...[]float64) {
	norm := 0.0
	for _, gs := range grads {
		for _, g := range gs {
			norm += g * g
		}
	}
	norm = math.Sqrt(norm)
	if norm > maxNorm {
		scale := maxNorm / norm
		for _, gs := range grads {
			for i := range gs {
				gs[i] *= scale
			}
		}
	}
}

func meanStd(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return 0, 1
	}
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	mean := sum / float64(len(xs))
	sumSq := 0.0
	for _, x := range xs {
		d := x - mean
		sumSq += d * d
	}
	std := math.Sqrt(sumSq / float64(len(xs)))
	return mean, std
}
