// Command automl demonstrates using the AutoML coordinator to search
// over hyperparameter configurations with Bayesian optimization and
// early stopping.
//
// This example uses a synthetic scoring function. In production, replace
// the worker with one that trains and evaluates a real model.
//
// Usage:
//
//	go build -o automl ./examples/automl/
//	./automl
package main

import (
	"fmt"
	"math"
	"os"

	"github.com/zerfoo/zerfoo/training/automl"
)

// syntheticWorker implements automl.Worker using a synthetic objective
// function. It simulates model training by computing a score based on
// the hyperparameters -- no GPU or real data required.
type syntheticWorker struct {
	evalCount int
}

// RunTrial evaluates a hyperparameter configuration. The synthetic
// objective has a known optimum near lr=0.003, hidden_dim=128,
// num_layers=2, dropout=0.1.
func (w *syntheticWorker) RunTrial(config automl.Config) (automl.Metric, error) {
	w.evalCount++
	p := config.Params

	lr := p["lr"]
	hiddenDim := p["hidden_dim"]
	numLayers := p["num_layers"]
	dropout := p["dropout"]

	// Synthetic objective: peaks near the known optimum.
	// Uses negative squared distance in a transformed space.
	score := 0.95 -
		0.5*math.Pow(math.Log10(lr)-math.Log10(0.003), 2) -
		0.001*math.Pow(hiddenDim-128, 2)/1000 -
		0.1*math.Pow(numLayers-2, 2) -
		0.3*math.Pow(dropout-0.1, 2)

	// Clamp to [0, 1].
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	fmt.Printf("  Trial %d: lr=%.4f hidden=%.0f layers=%.0f dropout=%.2f -> score=%.4f\n",
		w.evalCount, lr, hiddenDim, numLayers, dropout, score)

	return automl.Metric{Score: score}, nil
}

func main() {
	fmt.Println("=== AutoML Hyperparameter Search Example ===")

	// --- Step 1: Define the search space ---
	hparams := []automl.HParam{
		{Name: "lr", Min: 1e-4, Max: 1e-1, IsLog: true},
		{Name: "hidden_dim", Min: 32, Max: 256},
		{Name: "num_layers", Min: 1, Max: 4},
		{Name: "dropout", Min: 0.0, Max: 0.5},
	}

	fmt.Println("Search space:")
	for _, hp := range hparams {
		logStr := ""
		if hp.IsLog {
			logStr = " (log scale)"
		}
		fmt.Printf("  %s: [%.4f, %.4f]%s\n", hp.Name, hp.Min, hp.Max, logStr)
	}

	// --- Step 2: Create a Bayesian optimizer ---
	strategy := automl.NewBayesianOptimizer(hparams, 42)

	// --- Step 3: Configure and create the coordinator ---
	coordConfig := automl.CoordinatorConfig{
		Space: automl.SearchSpace{Params: hparams},
		Strategy:          strategy,
		MaxTrials:         20,
		EarlyStopPatience: 5,
	}
	fmt.Printf("\nMax trials: %d, Early stop patience: %d\n\n", coordConfig.MaxTrials, coordConfig.EarlyStopPatience)

	worker := &syntheticWorker{}

	coordinator, err := automl.NewCoordinator(coordConfig, worker)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating coordinator: %v\n", err)
		os.Exit(1)
	}

	// --- Step 4: Run the search ---
	fmt.Println("--- Running trials ---")
	best, err := coordinator.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running search: %v\n", err)
		os.Exit(1)
	}

	// --- Step 5: Report results ---
	fmt.Println("\n=== Search Complete ===")
	fmt.Printf("Best trial: #%d\n", best.TrialID)
	fmt.Printf("Best score: %.4f\n", best.Metric.Score)
	fmt.Println("Best hyperparameters:")
	for name, val := range best.Config.Params {
		fmt.Printf("  %s = %.4f\n", name, val)
	}

	fmt.Printf("\nTotal evaluations: %d\n", worker.evalCount)
}
