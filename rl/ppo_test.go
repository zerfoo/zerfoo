package rl

import (
	"context"
	"math"
	"testing"
)

type cartPoleEnv struct {
	state [4]float64
	steps int
}

func (c *cartPoleEnv) Reset() State {
	c.state = [4]float64{0, 0, 0.05, 0}
	c.steps = 0
	return c.state[:]
}

func (c *cartPoleEnv) Step(action Action) (State, float64, bool, error) {
	const (
		gravity  = 9.8
		massCart = 1.0
		massPole = 0.1
		length   = 0.5
		dt       = 0.02
		maxSteps = 200
	)
	totalMass := massCart + massPole
	poleMassLength := massPole * length
	x, xDot, theta, thetaDot := c.state[0], c.state[1], c.state[2], c.state[3]
	force := 10.0
	if len(action) > 0 && action[0] < 0 {
		force = -10.0
	}
	cosTheta := math.Cos(theta)
	sinTheta := math.Sin(theta)
	temp := (force + poleMassLength*thetaDot*thetaDot*sinTheta) / totalMass
	thetaAcc := (gravity*sinTheta - cosTheta*temp) /
		(length * (4.0/3.0 - massPole*cosTheta*cosTheta/totalMass))
	xAcc := temp - poleMassLength*thetaAcc*cosTheta/totalMass
	x += dt * xDot
	xDot += dt * xAcc
	theta += dt * thetaDot
	thetaDot += dt * thetaAcc
	c.state = [4]float64{x, xDot, theta, thetaDot}
	c.steps++
	done := x < -2.4 || x > 2.4 || theta < -0.2095 || theta > 0.2095 || c.steps >= maxSteps
	reward := 1.0
	if done && c.steps < maxSteps {
		reward = 0.0
	}
	return c.state[:], reward, done, nil
}

type banditEnv struct {
	state float64
	done  bool
}

func (b *banditEnv) Reset() State {
	if b.state >= 0 {
		b.state = -1.0
	} else {
		b.state = 1.0
	}
	b.done = false
	return []float64{b.state}
}

func (b *banditEnv) Step(action Action) (State, float64, bool, error) {
	reward := -1.0
	if len(action) > 0 && action[0]*b.state > 0 {
		reward = 1.0
	}
	b.done = true
	return []float64{b.state}, reward, true, nil
}

func TestPPO_CartPole(t *testing.T) {
	cfg := DefaultPPOConfig(4, 1)
	cfg.HiddenDim = 32
	cfg.NEpochs = 4
	cfg.BatchSize = 64
	cfg.LearningRate = 3e-3
	cfg.Gamma = 0.99
	cfg.Lambda = 0.95
	agent := NewPPO(cfg)
	env := &cartPoleEnv{}
	evaluate := func(nEpisodes int) float64 {
		totalReward := 0.0
		for ep := 0; ep < nEpisodes; ep++ {
			state := env.Reset()
			epReward := 0.0
			for step := 0; step < 200; step++ {
				action := agent.Act(state)
				next, reward, done, _ := env.Step(action)
				epReward += reward
				state = next
				if done {
					break
				}
			}
			totalReward += epReward
		}
		return totalReward / float64(nEpisodes)
	}
	bestReward := 0.0
	for iter := 0; iter < 100; iter++ {
		var trajectory []Experience
		for ep := 0; ep < 4; ep++ {
			state := env.Reset()
			for step := 0; step < 200; step++ {
				action := agent.Act(state)
				next, reward, done, _ := env.Step(action)
				trajectory = append(trajectory, Experience{
					State:     append([]float64(nil), state...),
					Action:    append([]float64(nil), action...),
					Reward:    reward,
					NextState: append([]float64(nil), next...),
					Done:      done,
				})
				state = next
				if done {
					break
				}
			}
		}
		if err := agent.Learn(trajectory); err != nil {
			t.Fatalf("Learn failed at iteration %d: %v", iter, err)
		}
		if iter%20 == 19 {
			avg := evaluate(10)
			if avg > bestReward {
				bestReward = avg
			}
			t.Logf("iter %d: avg reward %.1f (best %.1f)", iter+1, avg, bestReward)
		}
	}
	t.Logf("best avg reward: %.1f", bestReward)
	if bestReward < 9 {
		t.Errorf("PPO CartPole degenerate: best avg reward %.1f < 9", bestReward)
	}
}

func TestPPO_Bandit(t *testing.T) {
	cfg := DefaultPPOConfig(1, 1)
	cfg.HiddenDim = 16
	cfg.NEpochs = 4
	cfg.BatchSize = 32
	cfg.LearningRate = 3e-3
	cfg.Gamma = 0.0
	cfg.Lambda = 0.0
	agent := NewPPO(cfg)
	env := &banditEnv{}
	for iter := 0; iter < 200; iter++ {
		var trajectory []Experience
		for ep := 0; ep < 32; ep++ {
			state := env.Reset()
			action := agent.Act(state)
			next, reward, done, _ := env.Step(action)
			trajectory = append(trajectory, Experience{
				State:     append([]float64(nil), state...),
				Action:    append([]float64(nil), action...),
				Reward:    reward,
				NextState: append([]float64(nil), next...),
				Done:      done,
			})
			_ = done
		}
		if err := agent.Learn(trajectory); err != nil {
			t.Fatalf("Learn failed at iteration %d: %v", iter, err)
		}
	}
	correct := 0
	total := 100
	for i := 0; i < total; i++ {
		state := env.Reset()
		action := agent.Act(state)
		_, reward, _, _ := env.Step(action)
		if reward > 0 {
			correct++
		}
	}
	accuracy := float64(correct) / float64(total)
	t.Logf("bandit accuracy: %.0f%%", accuracy*100)
	if accuracy < 0.6 {
		t.Errorf("PPO bandit accuracy %.0f%% < 60%%", accuracy*100)
	}
}

func TestPPO_ClipObjective(t *testing.T) {
	cfg := DefaultPPOConfig(2, 1)
	cfg.HiddenDim = 16
	cfg.ClipRatio = 0.2
	cfg.NEpochs = 1
	cfg.BatchSize = 32
	cfg.LearningRate = 1e-3
	agent := NewPPO(cfg)
	batch := make([]Experience, 32)
	for i := range batch {
		s := []float64{float64(i) * 0.1, float64(i) * -0.1}
		batch[i] = Experience{
			State: s, Action: []float64{float64(i%2)*2 - 1},
			Reward: 1.0, NextState: s, Done: i == len(batch)-1,
		}
	}
	ctx := context.Background()
	oldLogProbs := make([]float64, len(batch))
	for i, exp := range batch {
		mean, logStd, err := agent.policy.forward(ctx, exp.State)
		if err != nil {
			t.Fatalf("policy forward failed: %v", err)
		}
		oldLogProbs[i] = logProb(exp.Action, mean, logStd)
	}
	if err := agent.Learn(batch); err != nil {
		t.Fatalf("Learn failed: %v", err)
	}
	clip := cfg.ClipRatio
	for i, exp := range batch {
		mean, logStd, err := agent.policy.forward(ctx, exp.State)
		if err != nil {
			t.Fatalf("policy forward failed: %v", err)
		}
		newLP := logProb(exp.Action, mean, logStd)
		ratio := math.Exp(newLP - oldLogProbs[i])
		tolerance := 0.25
		if ratio < 1-clip-tolerance || ratio > 1+clip+tolerance {
			t.Errorf("sample %d: ratio %.4f outside tolerance [%.2f, %.2f]",
				i, ratio, 1-clip-tolerance, 1+clip+tolerance)
		}
	}
}

func TestPPO_LearnEmptyBatch(t *testing.T) {
	cfg := DefaultPPOConfig(2, 1)
	agent := NewPPO(cfg)
	err := agent.Learn(nil)
	if err == nil {
		t.Error("expected error for empty batch, got nil")
	}
}

func TestPPO_ActDimension(t *testing.T) {
	cfg := DefaultPPOConfig(3, 2)
	agent := NewPPO(cfg)
	state := []float64{1, 2, 3}
	action := agent.Act(state)
	if len(action) != 2 {
		t.Errorf("Act returned %d-dim action, want 2", len(action))
	}
}

func TestPPO_DefaultConfig(t *testing.T) {
	cfg := DefaultPPOConfig(4, 2)
	if cfg.StateDim != 4 {
		t.Errorf("StateDim = %d, want 4", cfg.StateDim)
	}
	if cfg.ActionDim != 2 {
		t.Errorf("ActionDim = %d, want 2", cfg.ActionDim)
	}
	if cfg.ClipRatio != 0.2 {
		t.Errorf("ClipRatio = %f, want 0.2", cfg.ClipRatio)
	}
	if cfg.HiddenDim != 64 {
		t.Errorf("HiddenDim = %d, want 64", cfg.HiddenDim)
	}
}
