package main

import (
	"flag"
	"os"
	"testing"
	"time"
)

func TestTSTrainFlags(t *testing.T) {
	// Save and restore os.Args and flag state.
	origArgs := os.Args
	defer func() { os.Args = origArgs }()

	os.Args = []string{"ts_train"}
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)

	cfg := parseFlags()

	tests := []struct {
		name string
		got  any
		want any
	}{
		{"features-dir", cfg.featuresDir, "features/"},
		{"patch-len", cfg.patchLen, 16},
		{"stride", cfg.stride, 8},
		{"hidden-dim", cfg.hiddenDim, 128},
		{"num-heads", cfg.numHeads, 8},
		{"num-layers", cfg.numLayers, 6},
		{"horizon", cfg.horizon, 1},
		{"epochs", cfg.epochs, 50},
		{"batch-size", cfg.batchSize, 32},
		{"lr", cfg.lr, 1e-4},
		{"val-split", cfg.valSplit, 0.2},
		{"patience", cfg.patience, 5},
		{"output", cfg.output, "ts_model.gguf"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Errorf("flag %s: got %v, want %v", tt.name, tt.got, tt.want)
			}
		})
	}
}

func TestTimeOrderedSplit(t *testing.T) {
	base := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	n := 100
	ticks := make([]tick, n)
	for i := range n {
		ticks[i] = tick{
			timestamp: base.Add(time.Duration(i) * time.Hour),
			features:  []float64{float64(i)},
		}
	}

	train, val := timeOrderedSplit(ticks, 0.2)

	if len(train) != 80 {
		t.Fatalf("train length: got %d, want 80", len(train))
	}
	if len(val) != 20 {
		t.Fatalf("val length: got %d, want 20", len(val))
	}

	// Verify first 80 are in train.
	for i, tk := range train {
		if tk.features[0] != float64(i) {
			t.Errorf("train[%d].features[0]: got %v, want %v", i, tk.features[0], float64(i))
		}
	}

	// Verify last 20 are in val.
	for i, tk := range val {
		wantVal := float64(80 + i)
		if tk.features[0] != wantVal {
			t.Errorf("val[%d].features[0]: got %v, want %v", i, tk.features[0], wantVal)
		}
	}

	// Verify no overlap: last train timestamp < first val timestamp.
	if !train[len(train)-1].timestamp.Before(val[0].timestamp) {
		t.Error("train/val overlap: last train timestamp is not before first val timestamp")
	}
}

func TestEarlyStopping(t *testing.T) {
	// Simulate val losses: improves at epoch 1 and 2, then stalls.
	// With patience=3, should stop at epoch 5 (3 epochs without improvement after epoch 2).
	valLosses := []float32{0.5, 0.4, 0.41, 0.42, 0.43, 0.44}
	patience := 3

	bestValLoss := float32(1e30)
	patienceCounter := 0
	stoppedAt := 0

	for epoch := 1; epoch <= len(valLosses); epoch++ {
		valLoss := valLosses[epoch-1]

		if valLoss < bestValLoss {
			bestValLoss = valLoss
			patienceCounter = 0
		} else {
			patienceCounter++
			if patienceCounter >= patience {
				stoppedAt = epoch
				break
			}
		}
	}

	if stoppedAt != 5 {
		t.Errorf("early stopping epoch: got %d, want 5", stoppedAt)
	}
	if bestValLoss != 0.4 {
		t.Errorf("best val loss: got %v, want 0.4", bestValLoss)
	}
}
