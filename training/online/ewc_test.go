package online

import (
	"math"
	"testing"
)

// linearModel evaluates a simple linear model: prediction = dot(weights, input).
func linearModel(weights, input []float64) float64 {
	var sum float64
	for i := range input {
		if i < len(weights) {
			sum += weights[i] * input[i]
		}
	}
	return sum
}

// mseLoss computes mean squared error between model predictions and targets.
func mseLoss(weights []float64, data [][]float64, targets []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	var total float64
	for i, d := range data {
		pred := linearModel(weights, d)
		diff := pred - targets[i]
		total += diff * diff
	}
	return total / float64(len(data))
}

// trainSGD runs simple SGD on a linear model with optional EWC penalty.
func trainSGD(weights []float64, data [][]float64, targets []float64, lr float64, steps int, ewc *EWC) {
	n := len(data)
	if n == 0 {
		return
	}
	const eps = 1e-5
	grad := make([]float64, len(weights))
	tmp := make([]float64, len(weights))

	for step := 0; step < steps; step++ {
		// Compute gradient via finite differences on combined loss.
		lossFn := func(w []float64) float64 {
			l := mseLoss(w, data, targets)
			if ewc != nil {
				l += ewc.Penalty(w)
			}
			return l
		}

		baseLoss := lossFn(weights)
		for p := range weights {
			copy(tmp, weights)
			tmp[p] += eps
			grad[p] = (lossFn(tmp) - baseLoss) / eps
		}

		for p := range weights {
			weights[p] -= lr * grad[p]
		}
	}
}

func TestEWC_BasicPenalty(t *testing.T) {
	baseline := []float64{1.0, 2.0, 3.0}
	ewc := NewEWC(baseline, 10)
	ewc.SetLambda(1.0)

	// Before computing Fisher, penalty should be 0.
	current := []float64{2.0, 3.0, 4.0}
	if p := ewc.Penalty(current); p != 0 {
		t.Fatalf("expected 0 penalty before Fisher computation, got %f", p)
	}

	// Set up a simple loss function.
	data := [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	lossFn := func(w []float64) float64 {
		var sum float64
		for _, d := range data {
			pred := linearModel(w, d)
			sum += pred * pred
		}
		return sum / float64(len(data))
	}

	if err := ewc.ComputeFisher(data, lossFn); err != nil {
		t.Fatalf("ComputeFisher failed: %v", err)
	}

	// Now penalty should be > 0 for different weights.
	penalty := ewc.Penalty(current)
	if penalty <= 0 {
		t.Fatalf("expected positive penalty, got %f", penalty)
	}

	// Penalty at baseline should be 0.
	penaltyAtBaseline := ewc.Penalty(baseline)
	if penaltyAtBaseline != 0 {
		t.Fatalf("expected 0 penalty at baseline, got %f", penaltyAtBaseline)
	}
}

func TestEWC_LambdaScaling(t *testing.T) {
	baseline := []float64{1.0, 2.0}
	data := [][]float64{{1, 0}, {0, 1}}
	lossFn := func(w []float64) float64 {
		return mseLoss(w, data, []float64{1, 2})
	}

	ewc1 := NewEWC(baseline, 10)
	ewc1.SetLambda(1.0)
	if err := ewc1.ComputeFisher(data, lossFn); err != nil {
		t.Fatal(err)
	}

	ewc2 := NewEWC(baseline, 10)
	ewc2.SetLambda(10.0)
	if err := ewc2.ComputeFisher(data, lossFn); err != nil {
		t.Fatal(err)
	}

	current := []float64{2.0, 3.0}
	p1 := ewc1.Penalty(current)
	p2 := ewc2.Penalty(current)

	// p2 should be 10x p1 since lambda differs by 10x.
	ratio := p2 / p1
	if math.Abs(ratio-10.0) > 0.01 {
		t.Fatalf("expected penalty ratio of 10, got %f (p1=%f, p2=%f)", ratio, p1, p2)
	}
}

func TestEWC_PreventsForgetting(t *testing.T) {
	// Task A: learn weights that map inputs to targets.
	// Using identity-like inputs so each weight maps to one output.
	nParams := 4
	taskAData := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
	taskATargets := []float64{2.0, -1.0, 3.0, 0.5}

	// Train on task A from scratch.
	weightsA := make([]float64, nParams)
	trainSGD(weightsA, taskAData, taskATargets, 0.1, 500, nil)

	lossAbeforeB := mseLoss(weightsA, taskAData, taskATargets)
	if lossAbeforeB > 0.001 {
		t.Fatalf("failed to learn task A: loss=%f", lossAbeforeB)
	}

	// Set up EWC to protect task A weights.
	ewc := NewEWC(weightsA, len(taskAData))
	ewc.SetLambda(50.0) // strong protection relative to Hessian-based Fisher

	lossFn := func(w []float64) float64 {
		return mseLoss(w, taskAData, taskATargets)
	}
	if err := ewc.ComputeFisher(taskAData, lossFn); err != nil {
		t.Fatalf("ComputeFisher failed: %v", err)
	}

	// Task B: a conflicting mapping that would destroy task A knowledge.
	taskBData := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
	taskBTargets := []float64{-2.0, 1.0, -3.0, -0.5} // opposite of task A

	// Train on task B WITH EWC protection.
	weightsWithEWC := make([]float64, nParams)
	copy(weightsWithEWC, weightsA)
	trainSGD(weightsWithEWC, taskBData, taskBTargets, 0.05, 200, ewc)

	// Train on task B WITHOUT EWC (control).
	weightsWithoutEWC := make([]float64, nParams)
	copy(weightsWithoutEWC, weightsA)
	trainSGD(weightsWithoutEWC, taskBData, taskBTargets, 0.05, 200, nil)

	// Measure task A performance after learning task B.
	lossAWithEWC := mseLoss(weightsWithEWC, taskAData, taskATargets)
	lossAWithoutEWC := mseLoss(weightsWithoutEWC, taskAData, taskATargets)

	t.Logf("Task A loss: before_B=%f, with_EWC=%f, without_EWC=%f",
		lossAbeforeB, lossAWithEWC, lossAWithoutEWC)

	// EWC should preserve task A performance better than no protection.
	if lossAWithEWC >= lossAWithoutEWC {
		t.Errorf("EWC did not help preserve task A: with_EWC=%f >= without_EWC=%f",
			lossAWithEWC, lossAWithoutEWC)
	}

	// Task A loss with EWC should remain low.
	if lossAWithEWC > 0.5 {
		t.Errorf("task A loss with EWC too high: %f", lossAWithEWC)
	}
}

func TestEWC_ContinuousUpdate(t *testing.T) {
	nParams := 3

	// Learn 3 sequential tasks, updating EWC baseline after each.
	tasks := []struct {
		data    [][]float64
		targets []float64
	}{
		{
			data:    [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			targets: []float64{1.0, 2.0, 3.0},
		},
		{
			data:    [][]float64{{1, 1, 0}, {0, 1, 1}, {1, 0, 1}},
			targets: []float64{3.0, 5.0, 4.0},
		},
		{
			data:    [][]float64{{1, 1, 1}, {2, 0, 0}, {0, 2, 0}},
			targets: []float64{6.0, 2.0, 4.0},
		},
	}

	weights := make([]float64, nParams)
	var ewc *EWC

	for i, task := range tasks {
		// Train on this task.
		trainSGD(weights, task.data, task.targets, 0.05, 200, ewc)

		loss := mseLoss(weights, task.data, task.targets)
		t.Logf("Task %d loss after training: %f", i, loss)

		if loss > 1.0 {
			t.Errorf("task %d loss too high: %f", i, loss)
		}

		// Set up or update EWC after learning this task.
		if ewc == nil {
			ewc = NewEWC(weights, len(task.data))
			ewc.SetLambda(10.0)
		} else {
			ewc.UpdateBaseline(weights)
		}

		lossFn := func(w []float64) float64 {
			return mseLoss(w, task.data, task.targets)
		}
		if err := ewc.ComputeFisher(task.data, lossFn); err != nil {
			t.Fatalf("ComputeFisher failed on task %d: %v", i, err)
		}
	}

	// After all tasks, verify the model hasn't completely forgotten task 0.
	loss0 := mseLoss(weights, tasks[0].data, tasks[0].targets)
	t.Logf("Task 0 loss after all tasks: %f", loss0)

	// With EWC the first task loss should be bounded (not catastrophically forgotten).
	// We use a generous threshold since we're learning 3 tasks sequentially.
	if loss0 > 10.0 {
		t.Errorf("catastrophic forgetting detected: task 0 loss=%f", loss0)
	}
}

func TestEWC_UpdateBaseline(t *testing.T) {
	baseline := []float64{1.0, 2.0, 3.0}
	ewc := NewEWC(baseline, 5)

	newBaseline := []float64{4.0, 5.0, 6.0}
	ewc.UpdateBaseline(newBaseline)

	got := ewc.Baseline()
	for i, v := range got {
		if v != newBaseline[i] {
			t.Fatalf("baseline[%d] = %f, want %f", i, v, newBaseline[i])
		}
	}

	// Modifying newBaseline should not affect internal state.
	newBaseline[0] = 999
	if ewc.Baseline()[0] == 999 {
		t.Fatal("UpdateBaseline did not copy weights")
	}
}

func TestEWC_ComputeFisherErrors(t *testing.T) {
	ewc := NewEWC([]float64{1, 2}, 5)

	if err := ewc.ComputeFisher(nil, func(w []float64) float64 { return 0 }); err == nil {
		t.Fatal("expected error for nil data")
	}

	if err := ewc.ComputeFisher([][]float64{{1}}, nil); err == nil {
		t.Fatal("expected error for nil lossFn")
	}
}

func TestEWC_Loss(t *testing.T) {
	baseline := []float64{1.0, 2.0}
	ewc := NewEWC(baseline, 5)
	ewc.SetLambda(2.0)

	data := [][]float64{{1, 0}, {0, 1}}
	lossFn := func(w []float64) float64 {
		return mseLoss(w, data, []float64{1, 2})
	}
	if err := ewc.ComputeFisher(data, lossFn); err != nil {
		t.Fatal(err)
	}

	current := []float64{2.0, 3.0}
	taskLoss := 0.5

	totalLoss := ewc.Loss(taskLoss, current)
	expectedPenalty := ewc.Penalty(current)

	if math.Abs(totalLoss-(taskLoss+expectedPenalty)) > 1e-10 {
		t.Fatalf("Loss = %f, want %f + %f = %f", totalLoss, taskLoss, expectedPenalty, taskLoss+expectedPenalty)
	}

	// NaN and Inf pass through.
	if !math.IsNaN(ewc.Loss(math.NaN(), current)) {
		t.Fatal("expected NaN passthrough")
	}
	if !math.IsInf(ewc.Loss(math.Inf(1), current), 1) {
		t.Fatal("expected +Inf passthrough")
	}
}
