package rl

import (
	"context"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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

// mlpNet is a three-layer MLP (in -> hidden -> hidden -> out) using functional.Linear
// with ReLU activations on hidden layers.
type mlpNet struct {
	layer1 *linearLayer
	layer2 *linearLayer
	layer3 *linearLayer
	engine compute.Engine[float64]
	ops    numeric.Arithmetic[float64]
}

func newMLPNet(engine compute.Engine[float64], ops numeric.Arithmetic[float64],
	name string, inDim, hiddenDim, outDim int) (*mlpNet, error) {
	l1, err := newLinearLayer(name+".l1", inDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	l2, err := newLinearLayer(name+".l2", hiddenDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	l3, err := newLinearLayer(name+".l3", hiddenDim, outDim)
	if err != nil {
		return nil, err
	}
	return &mlpNet{layer1: l1, layer2: l2, layer3: l3, engine: engine, ops: ops}, nil
}

func (m *mlpNet) forward(ctx context.Context, input *tensor.TensorNumeric[float64]) (*tensor.TensorNumeric[float64], error) {
	h1, err := m.layer1.forward(ctx, m.engine, input)
	if err != nil {
		return nil, err
	}
	h1, err = functional.ReLU(ctx, m.engine, m.ops, h1)
	if err != nil {
		return nil, err
	}
	h2, err := m.layer2.forward(ctx, m.engine, h1)
	if err != nil {
		return nil, err
	}
	h2, err = functional.ReLU(ctx, m.engine, m.ops, h2)
	if err != nil {
		return nil, err
	}
	return m.layer3.forward(ctx, m.engine, h2)
}

func (m *mlpNet) forwardSlice(ctx context.Context, input []float64) ([]float64, error) {
	t, err := tensor.New[float64]([]int{1, len(input)}, input)
	if err != nil {
		return nil, err
	}
	out, err := m.forward(ctx, t)
	if err != nil {
		return nil, err
	}
	return out.Data(), nil
}

type mlpCache struct {
	input         *tensor.TensorNumeric[float64]
	h1Pre, h1Post *tensor.TensorNumeric[float64]
	h2Pre, h2Post *tensor.TensorNumeric[float64]
	output        *tensor.TensorNumeric[float64]
}

func (m *mlpNet) forwardCached(ctx context.Context, input *tensor.TensorNumeric[float64]) (*mlpCache, []float64, error) {
	c := &mlpCache{input: input}
	h1Pre, err := m.layer1.forward(ctx, m.engine, input)
	if err != nil {
		return nil, nil, err
	}
	c.h1Pre = h1Pre
	h1Post, err := functional.ReLU(ctx, m.engine, m.ops, h1Pre)
	if err != nil {
		return nil, nil, err
	}
	c.h1Post = h1Post
	h2Pre, err := m.layer2.forward(ctx, m.engine, h1Post)
	if err != nil {
		return nil, nil, err
	}
	c.h2Pre = h2Pre
	h2Post, err := functional.ReLU(ctx, m.engine, m.ops, h2Pre)
	if err != nil {
		return nil, nil, err
	}
	c.h2Post = h2Post
	output, err := m.layer3.forward(ctx, m.engine, h2Post)
	if err != nil {
		return nil, nil, err
	}
	c.output = output
	return c, output.Data(), nil
}

func (m *mlpNet) backward(ctx context.Context, c *mlpCache, dOutput *tensor.TensorNumeric[float64]) (*tensor.TensorNumeric[float64], error) {
	outDim := m.layer3.outDim
	hiddenDim := m.layer3.inDim
	dOutputCol, err := m.engine.Reshape(ctx, dOutput, []int{outDim, 1})
	if err != nil {
		return nil, err
	}
	dw3, err := m.engine.MatMul(ctx, dOutputCol, c.h2Post)
	if err != nil {
		return nil, err
	}
	if err := m.layer3.weight.AddGradient(dw3); err != nil {
		return nil, err
	}
	dOutputBias, err := m.engine.Reshape(ctx, dOutput, []int{outDim})
	if err != nil {
		return nil, err
	}
	if err := m.layer3.bias.AddGradient(dOutputBias); err != nil {
		return nil, err
	}
	dOutputFlat, err := m.engine.Reshape(ctx, dOutput, []int{1, outDim})
	if err != nil {
		return nil, err
	}
	dh2, err := m.engine.MatMul(ctx, dOutputFlat, m.layer3.weight.Value)
	if err != nil {
		return nil, err
	}
	dh2Pre, err := reluGrad(ctx, m.engine, c.h2Pre, dh2)
	if err != nil {
		return nil, err
	}
	dh2PreCol, err := m.engine.Reshape(ctx, dh2Pre, []int{hiddenDim, 1})
	if err != nil {
		return nil, err
	}
	dw2, err := m.engine.MatMul(ctx, dh2PreCol, c.h1Post)
	if err != nil {
		return nil, err
	}
	if err := m.layer2.weight.AddGradient(dw2); err != nil {
		return nil, err
	}
	dh2PreBias, err := m.engine.Reshape(ctx, dh2Pre, []int{hiddenDim})
	if err != nil {
		return nil, err
	}
	if err := m.layer2.bias.AddGradient(dh2PreBias); err != nil {
		return nil, err
	}
	dh2PreFlat, err := m.engine.Reshape(ctx, dh2Pre, []int{1, hiddenDim})
	if err != nil {
		return nil, err
	}
	dh1, err := m.engine.MatMul(ctx, dh2PreFlat, m.layer2.weight.Value)
	if err != nil {
		return nil, err
	}
	dh1Pre, err := reluGrad(ctx, m.engine, c.h1Pre, dh1)
	if err != nil {
		return nil, err
	}
	dh1PreCol, err := m.engine.Reshape(ctx, dh1Pre, []int{hiddenDim, 1})
	if err != nil {
		return nil, err
	}
	dw1, err := m.engine.MatMul(ctx, dh1PreCol, c.input)
	if err != nil {
		return nil, err
	}
	if err := m.layer1.weight.AddGradient(dw1); err != nil {
		return nil, err
	}
	dh1PreBias, err := m.engine.Reshape(ctx, dh1Pre, []int{hiddenDim})
	if err != nil {
		return nil, err
	}
	if err := m.layer1.bias.AddGradient(dh1PreBias); err != nil {
		return nil, err
	}
	dh1PreFlat, err := m.engine.Reshape(ctx, dh1Pre, []int{1, hiddenDim})
	if err != nil {
		return nil, err
	}
	dInput, err := m.engine.MatMul(ctx, dh1PreFlat, m.layer1.weight.Value)
	if err != nil {
		return nil, err
	}
	return dInput, nil
}

func (m *mlpNet) params() []*graph.Parameter[float64] {
	params := m.layer1.params()
	params = append(params, m.layer2.params()...)
	params = append(params, m.layer3.params()...)
	return params
}

func (m *mlpNet) clearGradients() {
	for _, p := range m.params() {
		p.ClearGradient()
	}
}

func reluGrad(ctx context.Context, engine compute.Engine[float64],
	pre, dOutput *tensor.TensorNumeric[float64]) (*tensor.TensorNumeric[float64], error) {
	preData := pre.Data()
	dData := dOutput.Data()
	result := make([]float64, len(preData))
	for i := range result {
		if preData[i] > 0 {
			result[i] = dData[i]
		}
	}
	return tensor.New[float64](pre.Shape(), result)
}

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

func copyMLPNet(src *mlpNet) (*mlpNet, error) {
	l1, err := copyLinearLayer(src.layer1)
	if err != nil {
		return nil, err
	}
	l2, err := copyLinearLayer(src.layer2)
	if err != nil {
		return nil, err
	}
	l3, err := copyLinearLayer(src.layer3)
	if err != nil {
		return nil, err
	}
	return &mlpNet{layer1: l1, layer2: l2, layer3: l3, engine: src.engine, ops: src.ops}, nil
}

func copyLinearLayer(src *linearLayer) (*linearLayer, error) {
	wData := copySlice(src.weight.Value.Data())
	wTensor, err := tensor.New[float64](src.weight.Value.Shape(), wData)
	if err != nil {
		return nil, err
	}
	wParam, err := graph.NewParameter[float64](src.weight.Name+".copy", wTensor, tensor.New[float64])
	if err != nil {
		return nil, err
	}
	bData := copySlice(src.bias.Value.Data())
	bTensor, err := tensor.New[float64](src.bias.Value.Shape(), bData)
	if err != nil {
		return nil, err
	}
	bParam, err := graph.NewParameter[float64](src.bias.Name+".copy", bTensor, tensor.New[float64])
	if err != nil {
		return nil, err
	}
	return &linearLayer{weight: wParam, bias: bParam, inDim: src.inDim, outDim: src.outDim}, nil
}

func softUpdateMLP(target, source *mlpNet, tau float64) {
	softUpdate(target.layer1.weight.Value.Data(), source.layer1.weight.Value.Data(), tau)
	softUpdate(target.layer1.bias.Value.Data(), source.layer1.bias.Value.Data(), tau)
	softUpdate(target.layer2.weight.Value.Data(), source.layer2.weight.Value.Data(), tau)
	softUpdate(target.layer2.bias.Value.Data(), source.layer2.bias.Value.Data(), tau)
	softUpdate(target.layer3.weight.Value.Data(), source.layer3.weight.Value.Data(), tau)
	softUpdate(target.layer3.bias.Value.Data(), source.layer3.bias.Value.Data(), tau)
}

// SAC implements the Soft Actor-Critic algorithm with twin Q-networks
// and automatic entropy temperature tuning.
type SAC struct {
	config                       SACConfig
	engine                       compute.Engine[float64]
	ops                          numeric.Arithmetic[float64]
	actor                        *mlpNet
	critic1, critic2             *mlpNet
	targetCritic1, targetCritic2 *mlpNet
	logAlpha                     float64
	alpha                        float64
	actorOpt                     *optimizer.SGD[float64]
	critic1Opt                   *optimizer.SGD[float64]
	critic2Opt                   *optimizer.SGD[float64]
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
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	ops := numeric.Float64Ops{}
	criticIn := cfg.StateDim + cfg.ActionDim
	actor, err := newMLPNet(engine, ops, "actor", cfg.StateDim, cfg.HiddenDim, 2*cfg.ActionDim)
	if err != nil {
		panic("rl: failed to create actor: " + err.Error())
	}
	critic1, err := newMLPNet(engine, ops, "critic1", criticIn, cfg.HiddenDim, 1)
	if err != nil {
		panic("rl: failed to create critic1: " + err.Error())
	}
	critic2, err := newMLPNet(engine, ops, "critic2", criticIn, cfg.HiddenDim, 1)
	if err != nil {
		panic("rl: failed to create critic2: " + err.Error())
	}
	targetCritic1, err := copyMLPNet(critic1)
	if err != nil {
		panic("rl: failed to copy critic1: " + err.Error())
	}
	targetCritic2, err := copyMLPNet(critic2)
	if err != nil {
		panic("rl: failed to copy critic2: " + err.Error())
	}
	lr := float32(cfg.LearningRate)
	return &SAC{
		config: cfg, engine: engine, ops: ops,
		actor: actor, critic1: critic1, critic2: critic2,
		targetCritic1: targetCritic1, targetCritic2: targetCritic2,
		logAlpha: math.Log(cfg.InitAlpha), alpha: cfg.InitAlpha,
		actorOpt:   optimizer.NewSGD[float64](engine, ops, lr),
		critic1Opt: optimizer.NewSGD[float64](engine, ops, lr),
		critic2Opt: optimizer.NewSGD[float64](engine, ops, lr),
	}
}

// Alpha returns the current entropy temperature.
func (s *SAC) Alpha() float64 { return s.alpha }

func (s *SAC) sampleAction(state State) (Action, float64) {
	ctx := context.Background()
	out, err := s.actor.forwardSlice(ctx, state)
	if err != nil {
		return make(Action, s.config.ActionDim), 0
	}
	actionDim := s.config.ActionDim
	mean := out[:actionDim]
	logStd := out[actionDim:]
	action := make(Action, actionDim)
	lp := 0.0
	for i := range actionDim {
		ls := clampLogStd(logStd[i])
		std := math.Exp(ls)
		noise := rand.NormFloat64()
		u := mean[i] + std*noise
		a := math.Tanh(u)
		action[i] = a
		lp += -0.5*(noise*noise) - ls - 0.5*math.Log(2*math.Pi)
		lp -= math.Log(math.Max(1-a*a, 1e-6))
	}
	return action, lp
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
	ctx := context.Background()
	cfg := s.config
	actionDim := cfg.ActionDim
	s.critic1.clearGradients()
	s.critic2.clearGradients()
	for _, exp := range batch {
		nextAction, nextLogProb := s.sampleAction(exp.NextState)
		nextInput := append(copySlice(exp.NextState), nextAction...)
		tq1, err := s.targetCritic1.forwardSlice(ctx, nextInput)
		if err != nil {
			return err
		}
		tq2, err := s.targetCritic2.forwardSlice(ctx, nextInput)
		if err != nil {
			return err
		}
		minTQ := tq1[0]
		if tq2[0] < tq1[0] {
			minTQ = tq2[0]
		}
		doneVal := 0.0
		if !exp.Done {
			doneVal = 1.0
		}
		target := exp.Reward + cfg.Gamma*doneVal*(minTQ-s.alpha*nextLogProb)
		curInput := append(copySlice(exp.State), exp.Action...)
		curInputT, err := tensor.New[float64]([]int{1, len(curInput)}, curInput)
		if err != nil {
			return err
		}
		cache1, q1, err := s.critic1.forwardCached(ctx, curInputT)
		if err != nil {
			return err
		}
		dq1Val := 2 * (q1[0] - target) / float64(len(batch))
		dq1, err := tensor.New[float64]([]int{1, 1}, []float64{dq1Val})
		if err != nil {
			return err
		}
		if _, err := s.critic1.backward(ctx, cache1, dq1); err != nil {
			return err
		}
		cache2, q2, err := s.critic2.forwardCached(ctx, curInputT)
		if err != nil {
			return err
		}
		dq2Val := 2 * (q2[0] - target) / float64(len(batch))
		dq2, err := tensor.New[float64]([]int{1, 1}, []float64{dq2Val})
		if err != nil {
			return err
		}
		if _, err := s.critic2.backward(ctx, cache2, dq2); err != nil {
			return err
		}
	}
	if err := s.critic1Opt.Step(ctx, s.critic1.params()); err != nil {
		return err
	}
	if err := s.critic2Opt.Step(ctx, s.critic2.params()); err != nil {
		return err
	}
	s.actor.clearGradients()
	totalLogProb := 0.0
	invBatch := 1.0 / float64(len(batch))
	for _, exp := range batch {
		stateT, err := tensor.New[float64]([]int{1, cfg.StateDim}, exp.State)
		if err != nil {
			return err
		}
		actorCache, actorOut, err := s.actor.forwardCached(ctx, stateT)
		if err != nil {
			return err
		}
		mean := actorOut[:actionDim]
		logStd := actorOut[actionDim:]
		action := make([]float64, actionDim)
		noises := make([]float64, actionDim)
		lp := 0.0
		for i := range actionDim {
			ls := clampLogStd(logStd[i])
			std := math.Exp(ls)
			noise := rand.NormFloat64()
			noises[i] = noise
			u := mean[i] + std*noise
			a := math.Tanh(u)
			action[i] = a
			lp += -0.5*(noise*noise) - ls - 0.5*math.Log(2*math.Pi)
			lp -= math.Log(math.Max(1-a*a, 1e-6))
		}
		totalLogProb += lp
		cInput := append(copySlice(exp.State), action...)
		cInputT, err := tensor.New[float64]([]int{1, len(cInput)}, cInput)
		if err != nil {
			return err
		}
		cCache, _, err := s.critic1.forwardCached(ctx, cInputT)
		if err != nil {
			return err
		}
		savedGrads := make([][]float64, len(s.critic1.params()))
		for i, p := range s.critic1.params() {
			savedGrads[i] = copySlice(p.Gradient.Data())
		}
		s.critic1.clearGradients()
		dOne, err := tensor.New[float64]([]int{1, 1}, []float64{1.0})
		if err != nil {
			return err
		}
		dInput, err := s.critic1.backward(ctx, cCache, dOne)
		if err != nil {
			return err
		}
		dQda := dInput.Data()[cfg.StateDim:]
		for i, p := range s.critic1.params() {
			copy(p.Gradient.Data(), savedGrads[i])
		}
		dActorOut := make([]float64, 2*actionDim)
		for i := range actionDim {
			a := action[i]
			dtanh := math.Max(1-a*a, 1e-6)
			ls := clampLogStd(logStd[i])
			std := math.Exp(ls)
			dLogProb_da := 2 * a / math.Max(1-a*a, 1e-6)
			dLoss_da := (s.alpha*dLogProb_da - dQda[i]) * invBatch
			dLoss_du := dLoss_da * dtanh
			dActorOut[i] = dLoss_du
			dActorOut[actionDim+i] = dLoss_du*noises[i]*std + s.alpha*(-1)*invBatch
		}
		dActorOutT, err := tensor.New[float64]([]int{1, 2 * actionDim}, dActorOut)
		if err != nil {
			return err
		}
		if _, err := s.actor.backward(ctx, actorCache, dActorOutT); err != nil {
			return err
		}
	}
	if err := s.actorOpt.Step(ctx, s.actor.params()); err != nil {
		return err
	}
	avgLogProb := totalLogProb / float64(len(batch))
	alphaGrad := -(avgLogProb + cfg.TargetEntropy)
	s.logAlpha -= cfg.AlphaLR * alphaGrad
	s.alpha = math.Exp(s.logAlpha)
	softUpdateMLP(s.targetCritic1, s.critic1, cfg.Tau)
	softUpdateMLP(s.targetCritic2, s.critic2, cfg.Tau)
	return nil
}
