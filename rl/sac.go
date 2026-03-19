package rl

import (
	"math"
	"math/rand/v2"
)

// SACConfig holds hyperparameters for the SAC agent.
type SACConfig struct {
	Gamma         float64 // Discount factor (default 0.99).
	Tau           float64 // Soft update coefficient for target networks (default 0.005).
	LearningRate  float64 // Learning rate for actor and critic networks.
	AlphaLR       float64 // Learning rate for the entropy temperature parameter.
	StateDim      int     // Dimensionality of the state space.
	ActionDim     int     // Dimensionality of the action space.
	HiddenDim     int     // Width of hidden layers in actor and critic networks.
	BatchSize     int     // Mini-batch size for learning.
	InitAlpha     float64 // Initial entropy temperature.
	TargetEntropy float64 // Target entropy for automatic alpha tuning (typically -ActionDim).
}

// mlp is a simple two-hidden-layer feedforward network: in -> hidden -> hidden -> out.
type mlp struct {
	w1, b1 []float64 // First layer: inDim -> hiddenDim
	w2, b2 []float64 // Second layer: hiddenDim -> hiddenDim
	w3, b3 []float64 // Output layer: hiddenDim -> outDim

	inDim, hiddenDim, outDim int
}

func newMLP(inDim, hiddenDim, outDim int) *mlp {
	m := &mlp{
		w1: make([]float64, inDim*hiddenDim),
		b1: make([]float64, hiddenDim),
		w2: make([]float64, hiddenDim*hiddenDim),
		b2: make([]float64, hiddenDim),
		w3: make([]float64, hiddenDim*outDim),
		b3: make([]float64, outDim),

		inDim:     inDim,
		hiddenDim: hiddenDim,
		outDim:    outDim,
	}
	// Xavier initialisation.
	xavierInit(m.w1, inDim, hiddenDim)
	xavierInit(m.w2, hiddenDim, hiddenDim)
	xavierInit(m.w3, hiddenDim, outDim)
	return m
}

func xavierInit(w []float64, fanIn, fanOut int) {
	scale := math.Sqrt(2.0 / float64(fanIn+fanOut))
	for i := range w {
		w[i] = rand.NormFloat64() * scale
	}
}

// forward computes the output of the MLP with ReLU activations on hidden layers.
func (m *mlp) forward(input []float64) []float64 {
	h1 := linearReLU(input, m.w1, m.b1, m.inDim, m.hiddenDim)
	h2 := linearReLU(h1, m.w2, m.b2, m.hiddenDim, m.hiddenDim)
	return linearRaw(h2, m.w3, m.b3, m.hiddenDim, m.outDim)
}

func linearReLU(x, w, b []float64, inDim, outDim int) []float64 {
	out := make([]float64, outDim)
	for j := range outDim {
		sum := b[j]
		for i := range inDim {
			sum += x[i] * w[i*outDim+j]
		}
		if sum > 0 {
			out[j] = sum
		}
	}
	return out
}

func linearRaw(x, w, b []float64, inDim, outDim int) []float64 {
	out := make([]float64, outDim)
	for j := range outDim {
		sum := b[j]
		for i := range inDim {
			sum += x[i] * w[i*outDim+j]
		}
		out[j] = sum
	}
	return out
}

// forwardWithGrad computes forward pass and returns hidden activations for backprop.
type mlpCache struct {
	input, h1Pre, h1, h2Pre, h2, output []float64
}

func (m *mlp) forwardCached(input []float64) (*mlpCache, []float64) {
	c := &mlpCache{input: input}

	c.h1Pre = linearRawSlice(input, m.w1, m.b1, m.inDim, m.hiddenDim)
	c.h1 = reluVec(c.h1Pre)

	c.h2Pre = linearRawSlice(c.h1, m.w2, m.b2, m.hiddenDim, m.hiddenDim)
	c.h2 = reluVec(c.h2Pre)

	c.output = linearRaw(c.h2, m.w3, m.b3, m.hiddenDim, m.outDim)
	return c, c.output
}

func linearRawSlice(x, w, b []float64, inDim, outDim int) []float64 {
	out := make([]float64, outDim)
	for j := range outDim {
		sum := b[j]
		for i := range inDim {
			sum += x[i] * w[i*outDim+j]
		}
		out[j] = sum
	}
	return out
}

func reluVec(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

// backward computes gradients for the MLP given dOutput (gradient of loss w.r.t. output).
// It accumulates into the provided gradient slices (caller zeroes them per batch).
func (m *mlp) backward(c *mlpCache, dOutput []float64,
	dw1, db1, dw2, db2, dw3, db3 []float64) []float64 {

	// Output layer: dL/dw3, dL/db3
	for j := range m.outDim {
		db3[j] += dOutput[j]
		for i := range m.hiddenDim {
			dw3[i*m.outDim+j] += c.h2[i] * dOutput[j]
		}
	}

	// Backprop through output layer to h2
	dh2 := make([]float64, m.hiddenDim)
	for i := range m.hiddenDim {
		for j := range m.outDim {
			dh2[i] += m.w3[i*m.outDim+j] * dOutput[j]
		}
	}

	// ReLU derivative on h2
	dh2Pre := make([]float64, m.hiddenDim)
	for i := range m.hiddenDim {
		if c.h2Pre[i] > 0 {
			dh2Pre[i] = dh2[i]
		}
	}

	// Second hidden layer
	for j := range m.hiddenDim {
		db2[j] += dh2Pre[j]
		for i := range m.hiddenDim {
			dw2[i*m.hiddenDim+j] += c.h1[i] * dh2Pre[j]
		}
	}

	dh1 := make([]float64, m.hiddenDim)
	for i := range m.hiddenDim {
		for j := range m.hiddenDim {
			dh1[i] += m.w2[i*m.hiddenDim+j] * dh2Pre[j]
		}
	}

	// ReLU derivative on h1
	dh1Pre := make([]float64, m.hiddenDim)
	for i := range m.hiddenDim {
		if c.h1Pre[i] > 0 {
			dh1Pre[i] = dh1[i]
		}
	}

	// First hidden layer
	for j := range m.hiddenDim {
		db1[j] += dh1Pre[j]
		for i := range m.inDim {
			dw1[i*m.hiddenDim+j] += c.input[i] * dh1Pre[j]
		}
	}

	// Gradient w.r.t. input (for actor backprop through critic)
	dInput := make([]float64, m.inDim)
	for i := range m.inDim {
		for j := range m.hiddenDim {
			dInput[i] += m.w1[i*m.hiddenDim+j] * dh1Pre[j]
		}
	}
	return dInput
}

// sgdUpdate applies gradient descent: param -= lr * grad, then zeroes grad.
func sgdUpdate(param, grad []float64, lr float64) {
	for i := range param {
		param[i] -= lr * grad[i]
		grad[i] = 0
	}
}

// softUpdate performs Polyak averaging: target = tau*source + (1-tau)*target.
func softUpdate(target, source []float64, tau float64) {
	for i := range target {
		target[i] = tau*source[i] + (1-tau)*target[i]
	}
}

func copySlice(src []float64) []float64 {
	dst := make([]float64, len(src))
	copy(dst, src)
	return dst
}

func clampLogStd(ls float64) float64 {
	if ls < -20 {
		return -20
	}
	if ls > 2 {
		return 2
	}
	return ls
}

func copyMLP(src *mlp) *mlp {
	return &mlp{
		w1: copySlice(src.w1), b1: copySlice(src.b1),
		w2: copySlice(src.w2), b2: copySlice(src.b2),
		w3: copySlice(src.w3), b3: copySlice(src.b3),
		inDim: src.inDim, hiddenDim: src.hiddenDim, outDim: src.outDim,
	}
}

// SAC implements the Soft Actor-Critic algorithm with twin Q-networks
// and automatic entropy temperature tuning.
type SAC struct {
	config SACConfig

	// Actor network: state -> [mean; logStd] (2*actionDim outputs)
	actor *mlp

	// Twin critics: (state, action) -> Q-value
	critic1, critic2         *mlp
	targetCritic1, targetCritic2 *mlp

	// Entropy temperature
	logAlpha float64
	alpha    float64

	// Gradient buffers for actor
	actorDw1, actorDb1 []float64
	actorDw2, actorDb2 []float64
	actorDw3, actorDb3 []float64

	// Gradient buffers for critic1
	c1Dw1, c1Db1 []float64
	c1Dw2, c1Db2 []float64
	c1Dw3, c1Db3 []float64

	// Gradient buffers for critic2
	c2Dw1, c2Db1 []float64
	c2Dw2, c2Db2 []float64
	c2Dw3, c2Db3 []float64
}

// NewSAC creates a new SAC agent with the given configuration.
func NewSAC(cfg SACConfig) *SAC {
	if cfg.Gamma == 0 {
		cfg.Gamma = 0.99
	}
	if cfg.Tau == 0 {
		cfg.Tau = 0.005
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 3e-4
	}
	if cfg.AlphaLR == 0 {
		cfg.AlphaLR = 3e-4
	}
	if cfg.HiddenDim == 0 {
		cfg.HiddenDim = 64
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 64
	}
	if cfg.InitAlpha == 0 {
		cfg.InitAlpha = 0.2
	}
	if cfg.TargetEntropy == 0 {
		cfg.TargetEntropy = -float64(cfg.ActionDim)
	}

	criticIn := cfg.StateDim + cfg.ActionDim

	s := &SAC{
		config:   cfg,
		actor:    newMLP(cfg.StateDim, cfg.HiddenDim, 2*cfg.ActionDim),
		critic1:  newMLP(criticIn, cfg.HiddenDim, 1),
		critic2:  newMLP(criticIn, cfg.HiddenDim, 1),
		logAlpha: math.Log(cfg.InitAlpha),
		alpha:    cfg.InitAlpha,
	}

	s.targetCritic1 = copyMLP(s.critic1)
	s.targetCritic2 = copyMLP(s.critic2)

	// Allocate gradient buffers for actor.
	s.actorDw1 = make([]float64, len(s.actor.w1))
	s.actorDb1 = make([]float64, len(s.actor.b1))
	s.actorDw2 = make([]float64, len(s.actor.w2))
	s.actorDb2 = make([]float64, len(s.actor.b2))
	s.actorDw3 = make([]float64, len(s.actor.w3))
	s.actorDb3 = make([]float64, len(s.actor.b3))

	// Allocate gradient buffers for critics.
	s.c1Dw1 = make([]float64, len(s.critic1.w1))
	s.c1Db1 = make([]float64, len(s.critic1.b1))
	s.c1Dw2 = make([]float64, len(s.critic1.w2))
	s.c1Db2 = make([]float64, len(s.critic1.b2))
	s.c1Dw3 = make([]float64, len(s.critic1.w3))
	s.c1Db3 = make([]float64, len(s.critic1.b3))

	s.c2Dw1 = make([]float64, len(s.critic2.w1))
	s.c2Db1 = make([]float64, len(s.critic2.b1))
	s.c2Dw2 = make([]float64, len(s.critic2.w2))
	s.c2Db2 = make([]float64, len(s.critic2.b2))
	s.c2Dw3 = make([]float64, len(s.critic2.w3))
	s.c2Db3 = make([]float64, len(s.critic2.b3))

	return s
}

// Alpha returns the current entropy temperature.
func (s *SAC) Alpha() float64 { return s.alpha }

// sampleAction uses the reparameterisation trick: action = tanh(mean + std*noise).
// Returns the squashed action and the log-probability.
func (s *SAC) sampleAction(state State) (Action, float64) {
	out := s.actor.forward(state)
	actionDim := s.config.ActionDim
	mean := out[:actionDim]
	logStd := out[actionDim:]

	action := make(Action, actionDim)
	logProb := 0.0

	for i := range actionDim {
		ls := clampLogStd(logStd[i])
		std := math.Exp(ls)

		noise := rand.NormFloat64()
		u := mean[i] + std*noise // pre-squash

		// Squash through tanh.
		a := math.Tanh(u)
		action[i] = a

		// Log probability with tanh correction:
		// log pi(a|s) = log N(u; mean, std) - log(1 - tanh(u)^2)
		logProb += -0.5*(noise*noise) - ls - 0.5*math.Log(2*math.Pi)
		// Correction for tanh squashing, with small epsilon for stability.
		logProb -= math.Log(math.Max(1-a*a, 1e-6))
	}

	return action, logProb
}

// Act selects an action for the given state using the current policy.
func (s *SAC) Act(state State) Action {
	action, _ := s.sampleAction(state)
	return action
}

// Learn updates actor, twin critics, and entropy temperature from a batch of experiences.
func (s *SAC) Learn(batch []Experience) error {
	if len(batch) == 0 {
		return nil
	}

	cfg := s.config
	actionDim := cfg.ActionDim

	// ---- Update critics ----
	for _, exp := range batch {
		// Sample next action from current policy for the next state.
		nextAction, nextLogProb := s.sampleAction(exp.NextState)

		// Compute target Q using target networks.
		nextInput := append(copySlice(exp.NextState), nextAction...)
		tq1 := s.targetCritic1.forward(nextInput)[0]
		tq2 := s.targetCritic2.forward(nextInput)[0]
		minTQ := tq1
		if tq2 < tq1 {
			minTQ = tq2
		}

		doneVal := 0.0
		if !exp.Done {
			doneVal = 1.0
		}
		target := exp.Reward + cfg.Gamma*doneVal*(minTQ-s.alpha*nextLogProb)

		// Critic 1 update.
		curInput := append(copySlice(exp.State), exp.Action...)
		cache1, q1 := s.critic1.forwardCached(curInput)
		dq1 := []float64{2 * (q1[0] - target) / float64(len(batch))}
		s.critic1.backward(cache1, dq1, s.c1Dw1, s.c1Db1, s.c1Dw2, s.c1Db2, s.c1Dw3, s.c1Db3)

		// Critic 2 update.
		cache2, q2 := s.critic2.forwardCached(curInput)
		dq2 := []float64{2 * (q2[0] - target) / float64(len(batch))}
		s.critic2.backward(cache2, dq2, s.c2Dw1, s.c2Db1, s.c2Dw2, s.c2Db2, s.c2Dw3, s.c2Db3)
	}

	sgdUpdate(s.critic1.w1, s.c1Dw1, cfg.LearningRate)
	sgdUpdate(s.critic1.b1, s.c1Db1, cfg.LearningRate)
	sgdUpdate(s.critic1.w2, s.c1Dw2, cfg.LearningRate)
	sgdUpdate(s.critic1.b2, s.c1Db2, cfg.LearningRate)
	sgdUpdate(s.critic1.w3, s.c1Dw3, cfg.LearningRate)
	sgdUpdate(s.critic1.b3, s.c1Db3, cfg.LearningRate)

	sgdUpdate(s.critic2.w1, s.c2Dw1, cfg.LearningRate)
	sgdUpdate(s.critic2.b1, s.c2Db1, cfg.LearningRate)
	sgdUpdate(s.critic2.w2, s.c2Dw2, cfg.LearningRate)
	sgdUpdate(s.critic2.b2, s.c2Db2, cfg.LearningRate)
	sgdUpdate(s.critic2.w3, s.c2Dw3, cfg.LearningRate)
	sgdUpdate(s.critic2.b3, s.c2Db3, cfg.LearningRate)

	// ---- Update actor ----
	// Actor loss: E[alpha * logProb - min(Q1, Q2)(s, a)]
	// Use reparameterisation trick with numerical dQ/da via finite differences.
	const eps = 1e-4
	totalLogProb := 0.0
	invBatch := 1.0 / float64(len(batch))

	for _, exp := range batch {
		// Forward through actor to get mean and log_std.
		actorCache, actorOut := s.actor.forwardCached(exp.State)
		mean := actorOut[:actionDim]
		logStd := actorOut[actionDim:]

		action := make([]float64, actionDim)
		noises := make([]float64, actionDim)
		logProb := 0.0

		for i := range actionDim {
			ls := clampLogStd(logStd[i])
			std := math.Exp(ls)
			noise := rand.NormFloat64()
			noises[i] = noise
			u := mean[i] + std*noise
			a := math.Tanh(u)
			action[i] = a

			logProb += -0.5*(noise*noise) - ls - 0.5*math.Log(2*math.Pi)
			logProb -= math.Log(math.Max(1-a*a, 1e-6))
		}
		totalLogProb += logProb

		// Compute min(Q1, Q2) for the sampled action.
		cInput := append(copySlice(exp.State), action...)
		q1 := s.critic1.forward(cInput)[0]
		q2 := s.critic2.forward(cInput)[0]
		minQ := q1
		if q2 < q1 {
			minQ = q2
		}

		// Compute dQ/da via finite differences for each action dimension.
		dQda := make([]float64, actionDim)
		for i := range actionDim {
			cPlus := append(copySlice(exp.State), copySlice(action)...)
			cPlus[cfg.StateDim+i] += eps
			cMinus := append(copySlice(exp.State), copySlice(action)...)
			cMinus[cfg.StateDim+i] -= eps

			qp1 := s.critic1.forward(cPlus)[0]
			qp2 := s.critic2.forward(cPlus)[0]
			qpMin := qp1
			if qp2 < qp1 {
				qpMin = qp2
			}
			qm1 := s.critic1.forward(cMinus)[0]
			qm2 := s.critic2.forward(cMinus)[0]
			qmMin := qm1
			if qm2 < qm1 {
				qmMin = qm2
			}
			dQda[i] = (qpMin - qmMin) / (2 * eps)
		}

		// Gradient of actor loss w.r.t. actor output (mean, logStd).
		// Loss = alpha * logProb - Q(s, a)
		// d(Loss)/d(actorOut) via chain rule through reparameterisation.
		dActorOut := make([]float64, 2*actionDim)
		for i := range actionDim {
			a := action[i]
			dtanh := math.Max(1-a*a, 1e-6)
			ls := clampLogStd(logStd[i])
			std := math.Exp(ls)

			// d(logProb)/da (tanh correction): 2a/(1-a^2)
			dLogProb_da := 2 * a / math.Max(1-a*a, 1e-6)

			// d(Loss)/da = alpha * d(logProb)/da - dQ/da
			dLoss_da := (s.alpha*dLogProb_da - dQda[i]) * invBatch

			// da/du = 1 - a^2; du/dmean = 1; du/dlogStd = noise*std
			dLoss_du := dLoss_da * dtanh

			dActorOut[i] += dLoss_du                           // d/dmean
			dActorOut[actionDim+i] += dLoss_du * noises[i] * std // d/dlogStd via u

			// d(logProb)/dlogStd = -1 (from -ls term)
			dActorOut[actionDim+i] += s.alpha * (-1) * invBatch
		}

		s.actor.backward(actorCache, dActorOut,
			s.actorDw1, s.actorDb1, s.actorDw2, s.actorDb2, s.actorDw3, s.actorDb3)

		_ = minQ
	}

	sgdUpdate(s.actor.w1, s.actorDw1, cfg.LearningRate)
	sgdUpdate(s.actor.b1, s.actorDb1, cfg.LearningRate)
	sgdUpdate(s.actor.w2, s.actorDw2, cfg.LearningRate)
	sgdUpdate(s.actor.b2, s.actorDb2, cfg.LearningRate)
	sgdUpdate(s.actor.w3, s.actorDw3, cfg.LearningRate)
	sgdUpdate(s.actor.b3, s.actorDb3, cfg.LearningRate)

	// ---- Update entropy temperature ----
	avgLogProb := totalLogProb / float64(len(batch))
	alphaGrad := -(avgLogProb + cfg.TargetEntropy)
	s.logAlpha -= cfg.AlphaLR * alphaGrad
	s.alpha = math.Exp(s.logAlpha)

	// ---- Soft-update target critics ----
	softUpdate(s.targetCritic1.w1, s.critic1.w1, cfg.Tau)
	softUpdate(s.targetCritic1.b1, s.critic1.b1, cfg.Tau)
	softUpdate(s.targetCritic1.w2, s.critic1.w2, cfg.Tau)
	softUpdate(s.targetCritic1.b2, s.critic1.b2, cfg.Tau)
	softUpdate(s.targetCritic1.w3, s.critic1.w3, cfg.Tau)
	softUpdate(s.targetCritic1.b3, s.critic1.b3, cfg.Tau)

	softUpdate(s.targetCritic2.w1, s.critic2.w1, cfg.Tau)
	softUpdate(s.targetCritic2.b1, s.critic2.b1, cfg.Tau)
	softUpdate(s.targetCritic2.w2, s.critic2.w2, cfg.Tau)
	softUpdate(s.targetCritic2.b2, s.critic2.b2, cfg.Tau)
	softUpdate(s.targetCritic2.w3, s.critic2.w3, cfg.Tau)
	softUpdate(s.targetCritic2.b3, s.critic2.b3, cfg.Tau)

	return nil
}
