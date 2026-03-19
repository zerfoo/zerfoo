package rl

import (
	"fmt"
	"math"
	"math/rand/v2"
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

// mlpLayer is a single dense layer: y = x*W + b.
type mlpLayer struct {
	weights []float64 // [inDim * outDim], row-major
	biases  []float64 // [outDim]
	inDim   int
	outDim  int
}

func newMLPLayer(inDim, outDim int) mlpLayer {
	scale := math.Sqrt(2.0 / float64(inDim+outDim))
	w := make([]float64, inDim*outDim)
	for i := range w {
		w[i] = rand.NormFloat64() * scale
	}
	b := make([]float64, outDim)
	return mlpLayer{weights: w, biases: b, inDim: inDim, outDim: outDim}
}

// forward computes y = x*W + b and returns y along with the pre-activation
// input x (needed for backprop).
func (l *mlpLayer) forward(x []float64) []float64 {
	out := make([]float64, l.outDim)
	for j := 0; j < l.outDim; j++ {
		s := l.biases[j]
		for i := 0; i < l.inDim; i++ {
			s += x[i] * l.weights[i*l.outDim+j]
		}
		out[j] = s
	}
	return out
}

// backward computes gradients given dL/dy (gradOutput) and the input x.
// It accumulates into wGrad and bGrad and returns dL/dx.
func (l *mlpLayer) backward(x, gradOutput, wGrad, bGrad []float64) []float64 {
	gradInput := make([]float64, l.inDim)
	for j := 0; j < l.outDim; j++ {
		bGrad[j] += gradOutput[j]
		for i := 0; i < l.inDim; i++ {
			wGrad[i*l.outDim+j] += x[i] * gradOutput[j]
			gradInput[i] += l.weights[i*l.outDim+j] * gradOutput[j]
		}
	}
	return gradInput
}

// policyNet is a 2-layer MLP that outputs action means and log standard deviations.
type policyNet struct {
	hidden mlpLayer
	mean   mlpLayer
	logStd []float64 // learnable per-action-dim log std (state-independent)
}

func newPolicyNet(stateDim, hiddenDim, actionDim int) *policyNet {
	logStd := make([]float64, actionDim)
	for i := range logStd {
		logStd[i] = -0.5 // initial std ~ 0.6
	}
	return &policyNet{
		hidden: newMLPLayer(stateDim, hiddenDim),
		mean:   newMLPLayer(hiddenDim, actionDim),
		logStd: logStd,
	}
}

// policyForwardResult stores intermediate activations for backprop.
type policyForwardResult struct {
	input      []float64
	hiddenPre  []float64
	hiddenPost []float64
	meanOut    []float64
	logStd     []float64
}

func (p *policyNet) forwardFull(state []float64) policyForwardResult {
	hiddenPre := p.hidden.forward(state)
	hiddenPost := make([]float64, len(hiddenPre))
	for i, v := range hiddenPre {
		hiddenPost[i] = math.Tanh(v)
	}
	meanOut := p.mean.forward(hiddenPost)
	ls := make([]float64, len(p.logStd))
	copy(ls, p.logStd)
	return policyForwardResult{
		input:      state,
		hiddenPre:  hiddenPre,
		hiddenPost: hiddenPost,
		meanOut:    meanOut,
		logStd:     ls,
	}
}

func (p *policyNet) forward(state []float64) (mean, logStd []float64) {
	r := p.forwardFull(state)
	return r.meanOut, r.logStd
}

// valueNet is a 2-layer MLP that outputs a scalar state value.
type valueNet struct {
	hidden mlpLayer
	out    mlpLayer
}

func newValueNet(stateDim, hiddenDim int) *valueNet {
	return &valueNet{
		hidden: newMLPLayer(stateDim, hiddenDim),
		out:    newMLPLayer(hiddenDim, 1),
	}
}

// valueForwardResult stores activations for backprop.
type valueForwardResult struct {
	input      []float64
	hiddenPre  []float64
	hiddenPost []float64
	value      float64
}

func (v *valueNet) forwardFull(state []float64) valueForwardResult {
	hiddenPre := v.hidden.forward(state)
	hiddenPost := make([]float64, len(hiddenPre))
	for i, val := range hiddenPre {
		hiddenPost[i] = math.Tanh(val)
	}
	out := v.out.forward(hiddenPost)
	return valueForwardResult{
		input:      state,
		hiddenPre:  hiddenPre,
		hiddenPost: hiddenPost,
		value:      out[0],
	}
}

func (v *valueNet) forward(state []float64) float64 {
	return v.forwardFull(state).value
}

// PPO implements the Proximal Policy Optimization agent with clipped surrogate
// objective and Generalized Advantage Estimation (GAE).
type PPO struct {
	config PPOConfig
	policy *policyNet
	value  *valueNet
}

// NewPPO creates a PPO agent with the given configuration.
func NewPPO(cfg PPOConfig) *PPO {
	return &PPO{
		config: cfg,
		policy: newPolicyNet(cfg.StateDim, cfg.HiddenDim, cfg.ActionDim),
		value:  newValueNet(cfg.StateDim, cfg.HiddenDim),
	}
}

// Act selects an action by sampling from the Gaussian policy.
func (p *PPO) Act(state State) Action {
	mean, logStd := p.policy.forward(state)
	action := make(Action, p.config.ActionDim)
	for i := range action {
		std := math.Exp(logStd[i])
		action[i] = mean[i] + rand.NormFloat64()*std
	}
	return action
}

// logProb computes the log-probability of action under a Gaussian(mean, exp(logStd)).
func logProb(action, mean, logStd []float64) float64 {
	lp := 0.0
	for i := range action {
		std := math.Exp(logStd[i])
		diff := action[i] - mean[i]
		lp += -0.5*math.Log(2*math.Pi) - logStd[i] - 0.5*(diff*diff)/(std*std)
	}
	return lp
}

// logProbGrad computes dLogProb/dMean and dLogProb/dLogStd.
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

	// Compute values for each state and the bootstrap value.
	values := make([]float64, n+1)
	for i := 0; i < n; i++ {
		values[i] = p.value.forward(batch[i].State)
	}
	if batch[n-1].Done {
		values[n] = 0
	} else {
		values[n] = p.value.forward(batch[n-1].NextState)
	}

	// Compute GAE advantages and returns.
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

	// Normalize advantages.
	advMean, advStd := meanStd(advantages)
	for i := range advantages {
		advantages[i] = (advantages[i] - advMean) / (advStd + 1e-8)
	}

	// Compute old log-probs under current policy (before updates).
	oldLogProbs := make([]float64, n)
	for i := 0; i < n; i++ {
		mean, logStd := p.policy.forward(batch[i].State)
		oldLogProbs[i] = logProb(batch[i].Action, mean, logStd)
	}

	// Multiple epochs of minibatch updates.
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
			p.updateMinibatch(batch, indices[start:end], oldLogProbs, advantages, returns)
		}
	}

	return nil
}

// updateMinibatch performs a single analytical gradient update step.
func (p *PPO) updateMinibatch(batch []Experience, indices []int, oldLogProbs, advantages, returns []float64) {
	lr := p.config.LearningRate
	clip := p.config.ClipRatio
	mbSize := float64(len(indices))

	pol := p.policy
	val := p.value
	hiddenDim := pol.hidden.outDim
	actionDim := pol.mean.outDim
	stateDim := pol.hidden.inDim

	// Policy gradient accumulators.
	pHiddenWGrad := make([]float64, stateDim*hiddenDim)
	pHiddenBGrad := make([]float64, hiddenDim)
	pMeanWGrad := make([]float64, hiddenDim*actionDim)
	pMeanBGrad := make([]float64, actionDim)
	pLogStdGrad := make([]float64, actionDim)

	// Value gradient accumulators.
	vStateDim := val.hidden.inDim
	vHiddenDim := val.hidden.outDim
	vHiddenWGrad := make([]float64, vStateDim*vHiddenDim)
	vHiddenBGrad := make([]float64, vHiddenDim)
	vOutWGrad := make([]float64, vHiddenDim)
	vOutBGrad := make([]float64, 1)

	for _, idx := range indices {
		exp := batch[idx]

		// --- Policy gradient ---
		pfr := pol.forwardFull(exp.State)
		newLP := logProb(exp.Action, pfr.meanOut, pfr.logStd)
		ratio := math.Exp(newLP - oldLogProbs[idx])
		adv := advantages[idx]

		clipped := math.Max(1-clip, math.Min(1+clip, ratio))
		surr1 := ratio * adv
		surr2 := clipped * adv

		// Gradient flows through ratio only when the unclipped term is the min.
		// When the clipped term is the min, the gradient w.r.t. ratio is zero
		// because clipped doesn't depend on ratio at the boundary.
		var dRatio float64
		if surr1 <= surr2 {
			// Unclipped term is the min (or equal) — gradient flows.
			dRatio = -adv / mbSize
		} else {
			// Clipped term is the min — gradient is zero.
			dRatio = 0
		}

		// dRatio/dLogProb = ratio (since ratio = exp(newLP - oldLP))
		dLogProb := dRatio * ratio

		// dLogProb/dMean, dLogProb/dLogStd
		dMean, dLogStd := logProbGrad(exp.Action, pfr.meanOut, pfr.logStd)

		// Gradient through mean output layer.
		gradMeanOut := make([]float64, actionDim)
		for i := range gradMeanOut {
			gradMeanOut[i] = dLogProb * dMean[i]
		}

		// logStd gradient.
		for i := range pLogStdGrad {
			pLogStdGrad[i] += dLogProb * dLogStd[i]
		}

		// Backprop through mean layer.
		gradHiddenPost := pol.mean.backward(pfr.hiddenPost, gradMeanOut, pMeanWGrad, pMeanBGrad)

		// Backprop through tanh.
		gradHiddenPre := make([]float64, hiddenDim)
		for i := range gradHiddenPre {
			t := pfr.hiddenPost[i]
			gradHiddenPre[i] = gradHiddenPost[i] * (1 - t*t)
		}

		// Backprop through hidden layer.
		pol.hidden.backward(pfr.input, gradHiddenPre, pHiddenWGrad, pHiddenBGrad)

		// --- Value gradient ---
		vfr := val.forwardFull(exp.State)
		diff := vfr.value - returns[idx]
		dValue := 2.0 * diff / mbSize // dMSE/dValue

		gradOutInput := val.out.backward(vfr.hiddenPost, []float64{dValue}, vOutWGrad, vOutBGrad)

		gradVHiddenPre := make([]float64, vHiddenDim)
		for i := range gradVHiddenPre {
			t := vfr.hiddenPost[i]
			gradVHiddenPre[i] = gradOutInput[i] * (1 - t*t)
		}
		val.hidden.backward(vfr.input, gradVHiddenPre, vHiddenWGrad, vHiddenBGrad)
	}

	// Clip gradients by global norm to stabilize training.
	clipGradNorm(1.0, pHiddenWGrad, pHiddenBGrad, pMeanWGrad, pMeanBGrad, pLogStdGrad)
	clipGradNorm(1.0, vHiddenWGrad, vHiddenBGrad, vOutWGrad, vOutBGrad)

	// Apply gradients.
	applyGrad(pol.hidden.weights, pHiddenWGrad, lr)
	applyGrad(pol.hidden.biases, pHiddenBGrad, lr)
	applyGrad(pol.mean.weights, pMeanWGrad, lr)
	applyGrad(pol.mean.biases, pMeanBGrad, lr)
	applyGrad(pol.logStd, pLogStdGrad, lr)
	applyGrad(val.hidden.weights, vHiddenWGrad, lr)
	applyGrad(val.hidden.biases, vHiddenBGrad, lr)
	applyGrad(val.out.weights, vOutWGrad, lr)
	applyGrad(val.out.biases, vOutBGrad, lr)
}

// clipGradNorm rescales all gradient slices so their combined L2 norm
// does not exceed maxNorm. Nil slices are skipped.
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

// applyGrad performs params -= lr * grad.
func applyGrad(params, grad []float64, lr float64) {
	for i := range params {
		params[i] -= lr * grad[i]
	}
}

// meanStd returns the mean and standard deviation of a float64 slice.
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
