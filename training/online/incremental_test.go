package online

import (
	"math"
	"testing"
)

// makeSamples generates n samples where the target is a simple linear function
// of the input: label[j] = sum(input) * (j+1) * 0.1. This gives the optimizer
// a learnable signal.
func makeSamples(n, dIn, dOut int) []Sample {
	samples := make([]Sample, n)
	for i := 0; i < n; i++ {
		input := make([]float32, dIn)
		for j := range input {
			input[j] = float32(i+1) * 0.1 * float32(j+1)
		}
		label := make([]float32, dOut)
		var sum float32
		for _, v := range input {
			sum += v
		}
		for j := range label {
			label[j] = sum * float32(j+1) * 0.1
		}
		samples[i] = Sample{Input: input, Label: label}
	}
	return samples
}

func TestIncrementalUpdate(t *testing.T) {
	cfg := LoRAUpdateConfig{
		Rank:          4,
		Alpha:         4,
		LR:            0.0001,
		MaxSteps:      5,
		TargetModules: []string{"q_proj", "v_proj"},
	}
	updater := NewIncrementalUpdater(cfg)
	samples := makeSamples(10, 8, 4)

	lossBefore := updater.CurrentLoss(samples)
	// Before any update, adapter is nil so loss is +Inf.
	if !math.IsInf(lossBefore, 1) {
		t.Fatalf("expected +Inf loss before init, got %f", lossBefore)
	}

	if err := updater.Update(samples); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	lossAfter := updater.CurrentLoss(samples)
	if math.IsInf(lossAfter, 1) || math.IsNaN(lossAfter) {
		t.Fatalf("loss after update is invalid: %f", lossAfter)
	}

	// Run a second update to further reduce loss.
	if err := updater.Update(samples); err != nil {
		t.Fatalf("second Update failed: %v", err)
	}
	lossFinal := updater.CurrentLoss(samples)

	if lossFinal >= lossAfter {
		t.Errorf("expected loss to decrease after second update: first=%f final=%f", lossAfter, lossFinal)
	}
	t.Logf("loss progression: before=+Inf -> after_first=%f -> after_second=%f", lossAfter, lossFinal)
}

func TestRollback(t *testing.T) {
	cfg := LoRAUpdateConfig{
		Rank:     4,
		Alpha:    4,
		LR:       0.0001,
		MaxSteps: 5,
	}
	updater := NewIncrementalUpdater(cfg)
	samples := makeSamples(10, 8, 4)

	// First update to initialize adapter.
	if err := updater.Update(samples); err != nil {
		t.Fatalf("initial Update failed: %v", err)
	}
	updater.CommitUpdate()

	// Capture weights before second update.
	weightsBefore := make([]float32, len(updater.adapter.A)+len(updater.adapter.B))
	copy(weightsBefore[:len(updater.adapter.A)], updater.adapter.A)
	copy(weightsBefore[len(updater.adapter.A):], updater.adapter.B)

	// Second update.
	if err := updater.Update(samples); err != nil {
		t.Fatalf("second Update failed: %v", err)
	}

	// Verify weights changed.
	weightsAfter := make([]float32, len(updater.adapter.A)+len(updater.adapter.B))
	copy(weightsAfter[:len(updater.adapter.A)], updater.adapter.A)
	copy(weightsAfter[len(updater.adapter.A):], updater.adapter.B)

	changed := false
	for i := range weightsBefore {
		if weightsBefore[i] != weightsAfter[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("weights should have changed after update")
	}

	// Rollback and verify weights match original.
	if err := updater.Rollback(); err != nil {
		t.Fatalf("Rollback failed: %v", err)
	}

	weightsRolledBack := make([]float32, len(updater.adapter.A)+len(updater.adapter.B))
	copy(weightsRolledBack[:len(updater.adapter.A)], updater.adapter.A)
	copy(weightsRolledBack[len(updater.adapter.A):], updater.adapter.B)

	for i := range weightsBefore {
		if weightsBefore[i] != weightsRolledBack[i] {
			t.Fatalf("weight[%d] mismatch after rollback: expected %f, got %f",
				i, weightsBefore[i], weightsRolledBack[i])
		}
	}
}

func TestCommitUpdate(t *testing.T) {
	cfg := LoRAUpdateConfig{
		Rank:     4,
		Alpha:    4,
		LR:       0.0001,
		MaxSteps: 5,
	}
	updater := NewIncrementalUpdater(cfg)
	samples := makeSamples(10, 8, 4)

	if err := updater.Update(samples); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Commit clears the snapshot.
	updater.CommitUpdate()

	// Rollback should fail after commit.
	err := updater.Rollback()
	if err == nil {
		t.Fatal("expected error on Rollback after CommitUpdate")
	}
}

func TestUpdateEmptySamples(t *testing.T) {
	cfg := LoRAUpdateConfig{Rank: 2, Alpha: 2, LR: 0.01, MaxSteps: 1}
	updater := NewIncrementalUpdater(cfg)

	err := updater.Update(nil)
	if err == nil {
		t.Fatal("expected error for nil samples")
	}

	err = updater.Update([]Sample{})
	if err == nil {
		t.Fatal("expected error for empty samples")
	}
}

func TestCurrentLossBeforeInit(t *testing.T) {
	cfg := LoRAUpdateConfig{Rank: 2, Alpha: 2, LR: 0.01, MaxSteps: 1}
	updater := NewIncrementalUpdater(cfg)

	loss := updater.CurrentLoss(nil)
	if !math.IsInf(loss, 1) {
		t.Fatalf("expected +Inf for nil samples, got %f", loss)
	}

	loss = updater.CurrentLoss([]Sample{})
	if !math.IsInf(loss, 1) {
		t.Fatalf("expected +Inf for empty samples, got %f", loss)
	}
}
