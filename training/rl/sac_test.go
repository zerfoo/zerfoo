package rl

import (
	"context"
	"math"
	"testing"
)

type targetEnv struct {
	target float64
	state  []float64
	steps  int
}

func newTargetEnv(target float64) *targetEnv {
	return &targetEnv{target: target}
}

func (e *targetEnv) Reset() State {
	e.steps = 0
	e.state = []float64{1.0}
	return e.state
}

func (e *targetEnv) Step(action Action) (State, float64, bool, error) {
	e.steps++
	diff := action[0] - e.target
	reward := -diff * diff
	done := e.steps >= 20
	return e.state, reward, done, nil
}

func TestSAC_ImplementsAgent(t *testing.T) {
	var _ Agent = &SAC{}
}

func TestSAC_ContinuousAction(t *testing.T) {
	env := newTargetEnv(0.3)
	cfg := SACConfig{
		StateDim: 1, ActionDim: 1, HiddenDim: 8,
		Gamma: 0.99, Tau: 0.01, LearningRate: 1e-2,
		AlphaLR: 1e-2, BatchSize: 16, InitAlpha: 0.05,
	}
	agent := NewSAC(cfg)
	buf, err := NewReplayBuffer(2000)
	if err != nil {
		t.Fatalf("NewReplayBuffer error: %v", err)
	}
	for ep := range 150 {
		state := env.Reset()
		for {
			action := agent.Act(state)
			next, reward, done, _ := env.Step(action)
			buf.Add(Experience{State: state, Action: action, Reward: reward, NextState: next, Done: done})
			state = next
			if done {
				break
			}
		}
		if buf.Len() >= cfg.BatchSize {
			for range 5 {
				batch := buf.Sample(cfg.BatchSize)
				if err := agent.Learn(batch); err != nil {
					t.Fatalf("Learn failed: %v", err)
				}
			}
		}
		_ = ep
	}
	var evalReward float64
	evalEps := 5
	for range evalEps {
		state := env.Reset()
		for {
			action := agent.Act(state)
			next, reward, done, _ := env.Step(action)
			evalReward += reward
			state = next
			if done {
				break
			}
		}
	}
	avgReward := evalReward / float64(evalEps)
	if avgReward < -8.4 {
		t.Errorf("SAC did not learn: average eval reward %.4f, want > -8.4 (random baseline)", avgReward)
	}
}

func TestSAC_EntropyTuning(t *testing.T) {
	cfg := SACConfig{
		StateDim: 2, ActionDim: 1, HiddenDim: 8,
		Gamma: 0.99, Tau: 0.005, LearningRate: 1e-3,
		AlphaLR: 1e-2, BatchSize: 16, InitAlpha: 1.0, TargetEntropy: -1.0,
	}
	agent := NewSAC(cfg)
	initialAlpha := agent.Alpha()
	batch := make([]Experience, cfg.BatchSize)
	for i := range batch {
		batch[i] = Experience{
			State: []float64{0.5, 0.5}, Action: []float64{0.1},
			Reward: 1.0, NextState: []float64{0.5, 0.5}, Done: false,
		}
	}
	alphaChanged := false
	for range 20 {
		if err := agent.Learn(batch); err != nil {
			t.Fatalf("Learn failed: %v", err)
		}
		if math.Abs(agent.Alpha()-initialAlpha) > 1e-4 {
			alphaChanged = true
		}
	}
	if !alphaChanged {
		t.Error("entropy temperature alpha did not change during training")
	}
}

func TestSAC_TwinCritics(t *testing.T) {
	cfg := SACConfig{StateDim: 2, ActionDim: 1, HiddenDim: 16}
	agent := NewSAC(cfg)
	ctx := context.Background()
	input := append([]float64{0.5, -0.3}, 0.1)
	q1, err := agent.critic1.forwardSlice(ctx, input)
	if err != nil {
		t.Fatalf("critic1 forward failed: %v", err)
	}
	q2, err := agent.critic2.forwardSlice(ctx, input)
	if err != nil {
		t.Fatalf("critic2 forward failed: %v", err)
	}
	if q1[0] == q2[0] {
		t.Error("twin critics produced identical outputs; expected different random initialisations")
	}
}

func TestSAC_TargetNetworkSoftUpdate(t *testing.T) {
	cfg := SACConfig{StateDim: 2, ActionDim: 1, HiddenDim: 8, Tau: 0.1}
	agent := NewSAC(cfg)
	targetW := agent.targetCritic1.layer1.weight.Value.Data()
	initialTargetW := copySlice(targetW)
	batch := []Experience{{
		State: []float64{1.0, 0.0}, Action: []float64{0.5},
		Reward: 1.0, NextState: []float64{0.0, 1.0}, Done: false,
	}}
	if err := agent.Learn(batch); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}
	moved := false
	targetWAfter := agent.targetCritic1.layer1.weight.Value.Data()
	for i := range targetWAfter {
		if targetWAfter[i] != initialTargetW[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Error("target network weights did not update after Learn")
	}
	criticW := agent.critic1.layer1.weight.Value.Data()
	for i := range targetWAfter {
		tw := targetWAfter[i]
		cw := criticW[i]
		iw := initialTargetW[i]
		expected := cfg.Tau*cw + (1-cfg.Tau)*iw
		if math.Abs(tw-expected) > 1e-10 {
			t.Errorf("target weight[%d] = %.10f, want %.10f (Polyak average)", i, tw, expected)
			break
		}
	}
}

func TestSAC_LearnEmptyBatch(t *testing.T) {
	agent := NewSAC(SACConfig{StateDim: 2, ActionDim: 1, HiddenDim: 8})
	if err := agent.Learn(nil); err != nil {
		t.Fatalf("Learn(nil) returned error: %v", err)
	}
	if err := agent.Learn([]Experience{}); err != nil {
		t.Fatalf("Learn([]) returned error: %v", err)
	}
}
