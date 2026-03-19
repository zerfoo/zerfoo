package rl

import (
	"math"
	"testing"
)

// targetEnv is a simple continuous control environment where the agent must
// output an action close to a fixed target value. Reward = -|action - target|^2.
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
	e.state = []float64{1.0} // constant observation
	return e.state
}

func (e *targetEnv) Step(action Action) (State, float64, bool, error) {
	e.steps++
	diff := action[0] - e.target
	reward := -diff * diff
	done := e.steps >= 50
	return e.state, reward, done, nil
}

func TestSAC_ImplementsAgent(t *testing.T) {
	var _ Agent = &SAC{}
}

func TestSAC_ContinuousAction(t *testing.T) {
	env := newTargetEnv(0.3)
	cfg := SACConfig{
		StateDim:     1,
		ActionDim:    1,
		HiddenDim:    16,
		Gamma:        0.99,
		Tau:          0.01,
		LearningRate: 3e-3,
		AlphaLR:      3e-3,
		BatchSize:    32,
		InitAlpha:    0.1,
	}
	agent := NewSAC(cfg)
	buf := NewReplayBuffer(5000)

	// Collect experience and train.
	var bestReward float64 = -math.MaxFloat64
	for ep := range 200 {
		state := env.Reset()
		var totalReward float64
		for {
			action := agent.Act(state)
			next, reward, done, _ := env.Step(action)
			buf.Add(Experience{
				State:     state,
				Action:    action,
				Reward:    reward,
				NextState: next,
				Done:      done,
			})

			totalReward += reward
			state = next
			if done {
				break
			}
		}
		// Train several times per episode to keep test fast while learning.
		if buf.Len() >= cfg.BatchSize {
			for range 10 {
				batch := buf.Sample(cfg.BatchSize)
				if err := agent.Learn(batch); err != nil {
					t.Fatalf("Learn failed: %v", err)
				}
			}
		}
		if totalReward > bestReward {
			bestReward = totalReward
		}
		_ = ep
	}

	// After training, the average action should be closer to target than random.
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

	// Random actions in [-1,1] against target=0.3 give E[reward] per step ≈ -0.42,
	// so random total ≈ -21 over 50 steps. A trained agent should do much better.
	if avgReward < -10 {
		t.Errorf("SAC did not learn: average eval reward %.4f, want > -10 (random ≈ -21)", avgReward)
	}
}

func TestSAC_EntropyTuning(t *testing.T) {
	cfg := SACConfig{
		StateDim:      2,
		ActionDim:     1,
		HiddenDim:     16,
		Gamma:         0.99,
		Tau:           0.005,
		LearningRate:  1e-3,
		AlphaLR:       1e-2, // Larger LR so alpha moves noticeably.
		BatchSize:     32,
		InitAlpha:     1.0,
		TargetEntropy: -1.0,
	}
	agent := NewSAC(cfg)

	initialAlpha := agent.Alpha()

	// Create a batch of experiences with deterministic-ish actions (low entropy).
	// This should cause alpha to increase (entropy too low -> raise temperature).
	batch := make([]Experience, cfg.BatchSize)
	for i := range batch {
		batch[i] = Experience{
			State:     []float64{0.5, 0.5},
			Action:    []float64{0.1},
			Reward:    1.0,
			NextState: []float64{0.5, 0.5},
			Done:      false,
		}
	}

	// Train for several steps and track alpha changes.
	alphaChanged := false
	for range 50 {
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
	cfg := SACConfig{
		StateDim:  2,
		ActionDim: 1,
		HiddenDim: 16,
	}
	agent := NewSAC(cfg)

	// Verify that the two critics produce different outputs (different initialisations).
	input := append([]float64{0.5, -0.3}, 0.1)
	q1 := agent.critic1.forward(input)
	q2 := agent.critic2.forward(input)

	// With random init it's astronomically unlikely they'd be exactly equal.
	if q1[0] == q2[0] {
		t.Error("twin critics produced identical outputs; expected different random initialisations")
	}
}

func TestSAC_TargetNetworkSoftUpdate(t *testing.T) {
	cfg := SACConfig{
		StateDim:  2,
		ActionDim: 1,
		HiddenDim: 8,
		Tau:       0.1,
	}
	agent := NewSAC(cfg)

	// Record initial target weights.
	initialTargetW := copySlice(agent.targetCritic1.w1)

	// Create a minimal batch and learn.
	batch := []Experience{{
		State:     []float64{1.0, 0.0},
		Action:    []float64{0.5},
		Reward:    1.0,
		NextState: []float64{0.0, 1.0},
		Done:      false,
	}}
	if err := agent.Learn(batch); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	// Target weights should have moved towards critic weights.
	moved := false
	for i := range agent.targetCritic1.w1 {
		if agent.targetCritic1.w1[i] != initialTargetW[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Error("target network weights did not update after Learn")
	}

	// Target should be between initial and current critic (Polyak average).
	for i := range agent.targetCritic1.w1 {
		tw := agent.targetCritic1.w1[i]
		cw := agent.critic1.w1[i]
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
