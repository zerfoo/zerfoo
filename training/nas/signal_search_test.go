package nas

import (
	"bytes"
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
)

// syntheticSignalData implements SignalDataProvider with deterministic data
// for simulation testing. It generates sine-wave-like patterns suitable for
// testing the NAS search pipeline without GPU.
type syntheticSignalData struct {
	size   int
	offset float64
}

func (s *syntheticSignalData) TrainBatch() (input, target []float32, shape []int, err error) {
	input = make([]float32, s.size)
	target = make([]float32, s.size)
	for i := range s.size {
		x := float64(i) * 0.1
		input[i] = float32(math.Sin(x + s.offset))
		target[i] = float32(math.Sin(x+s.offset) * 2.0)
	}
	return input, target, []int{s.size}, nil
}

func (s *syntheticSignalData) ValBatch() (input, target []float32, shape []int, err error) {
	input = make([]float32, s.size)
	target = make([]float32, s.size)
	for i := range s.size {
		x := float64(i)*0.1 + 0.5
		input[i] = float32(math.Sin(x + s.offset))
		target[i] = float32(math.Sin(x+s.offset) * 2.0)
	}
	return input, target, []int{s.size}, nil
}

func TestRunSignalNAS(t *testing.T) {
	tests := []struct {
		name      string
		cfg       SignalSearchConfig
		wantErr   bool
		errSubstr string
	}{
		{
			name: "valid single trial",
			cfg: SignalSearchConfig{
				NumTrials:     1,
				SearchSteps:   10,
				WeightLR:      0.01,
				AlphaLR:       0.1,
				InputFeatures: 4,
				PatchLen:      8,
				HorizonLen:    4,
				HiddenDim:     32,
				NumLayers:     2,
				Seed:          42,
			},
		},
		{
			name: "multiple trials",
			cfg: SignalSearchConfig{
				NumTrials:     3,
				SearchSteps:   5,
				WeightLR:      0.01,
				AlphaLR:       0.1,
				InputFeatures: 4,
				PatchLen:      8,
				HorizonLen:    4,
				HiddenDim:     32,
				NumLayers:     2,
				Seed:          123,
			},
		},
		{
			name: "zero trials",
			cfg: SignalSearchConfig{
				NumTrials:     0,
				SearchSteps:   10,
				WeightLR:      0.01,
				AlphaLR:       0.1,
				InputFeatures: 4,
				PatchLen:      8,
				HorizonLen:    4,
				HiddenDim:     32,
				NumLayers:     2,
			},
			wantErr:   true,
			errSubstr: "NumTrials",
		},
		{
			name: "zero search steps",
			cfg: SignalSearchConfig{
				NumTrials:     1,
				SearchSteps:   0,
				WeightLR:      0.01,
				AlphaLR:       0.1,
				InputFeatures: 4,
				PatchLen:      8,
				HorizonLen:    4,
				HiddenDim:     32,
				NumLayers:     2,
			},
			wantErr:   true,
			errSubstr: "SearchSteps",
		},
		{
			name: "zero weight lr",
			cfg: SignalSearchConfig{
				NumTrials:     1,
				SearchSteps:   10,
				WeightLR:      0,
				AlphaLR:       0.1,
				InputFeatures: 4,
				PatchLen:      8,
				HorizonLen:    4,
				HiddenDim:     32,
				NumLayers:     2,
			},
			wantErr:   true,
			errSubstr: "WeightLR",
		},
		{
			name: "missing input features",
			cfg: SignalSearchConfig{
				NumTrials:   1,
				SearchSteps: 10,
				WeightLR:    0.01,
				AlphaLR:     0.1,
				PatchLen:    8,
				HorizonLen:  4,
				HiddenDim:   32,
				NumLayers:   2,
			},
			wantErr:   true,
			errSubstr: "InputFeatures",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := &syntheticSignalData{size: 8, offset: 0}
			out, err := RunSignalNAS(context.Background(), tt.cfg, data)

			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.errSubstr)
				}
				if tt.errSubstr != "" {
					if got := err.Error(); !contains(got, tt.errSubstr) {
						t.Errorf("error %q does not contain %q", got, tt.errSubstr)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if out == nil {
				t.Fatal("output is nil")
			}
			if out.Best.Arch == nil {
				t.Fatal("best architecture is nil")
			}
			if !out.Best.Arch.Cell.Valid() {
				t.Error("best architecture cell is not valid")
			}
			if len(out.AllResults) != tt.cfg.NumTrials {
				t.Errorf("got %d results, want %d", len(out.AllResults), tt.cfg.NumTrials)
			}
		})
	}
}

func TestNASSignalSearch(t *testing.T) {
	// Full pipeline test: search -> discretize -> export -> round-trip load.
	ctx := context.Background()
	data := &syntheticSignalData{size: 16, offset: 0.3}

	cfg := SignalSearchConfig{
		NumTrials:     2,
		SearchSteps:   50,
		WeightLR:      0.01,
		AlphaLR:       0.1,
		InputFeatures: 4,
		PatchLen:      8,
		HorizonLen:    4,
		HiddenDim:     64,
		NumLayers:     2,
		Seed:          42,
	}

	// Step 1: Run NAS search.
	out, err := RunSignalNAS(ctx, cfg, data)
	if err != nil {
		t.Fatalf("RunSignalNAS: %v", err)
	}

	if out.Best.Arch == nil {
		t.Fatal("best architecture is nil")
	}
	if !out.Best.Arch.Cell.Valid() {
		t.Fatal("best architecture cell is not valid")
	}

	// Verify the discovered architecture achieves a measurable Sharpe ratio.
	// We use the metric values from all trials as a proxy for returns.
	returns := make([]float64, len(out.AllResults))
	for i, r := range out.AllResults {
		returns[i] = r.Metric
	}
	sharpe := SharpeRatio(returns)
	// With deterministic seed and 2 trials, we expect a non-zero Sharpe.
	// The exact value depends on the search, but it should be finite.
	if math.IsNaN(sharpe) || math.IsInf(sharpe, 0) {
		t.Errorf("Sharpe ratio is not finite: %f", sharpe)
	}
	t.Logf("Sharpe ratio across trials: %.4f", sharpe)
	t.Logf("Best trial: %d, loss: %.6f, metric: %.6f", out.Best.Trial, out.Best.FinalLoss, out.Best.Metric)

	// Step 2: Export to GGUF.
	weights := map[string][]float32{
		"blk.0.attn_q.weight": {0.1, 0.2, 0.3, 0.4},
		"blk.0.attn_k.weight": {0.5, 0.6, 0.7, 0.8},
	}
	shapes := map[string][]int{
		"blk.0.attn_q.weight": {2, 2},
		"blk.0.attn_k.weight": {2, 2},
	}

	var buf bytes.Buffer
	if err := ExportGGUF(&buf, out.Best.Arch, out.ExportConfig, weights, shapes); err != nil {
		t.Fatalf("ExportGGUF: %v", err)
	}

	// Step 3: Round-trip load.
	r := bytes.NewReader(buf.Bytes())
	gf, err := gguf.Parse(r)
	if err != nil {
		t.Fatalf("parse GGUF: %v", err)
	}

	loaded, loadedCfg, err := LoadNASArchFromGGUF(gf)
	if err != nil {
		t.Fatalf("LoadNASArchFromGGUF: %v", err)
	}

	// Verify architecture round-trip.
	if loaded.Cell.NumNodes != out.Best.Arch.Cell.NumNodes {
		t.Errorf("NumNodes: got %d, want %d", loaded.Cell.NumNodes, out.Best.Arch.Cell.NumNodes)
	}
	if len(loaded.Cell.Edges) != len(out.Best.Arch.Cell.Edges) {
		t.Errorf("num edges: got %d, want %d", len(loaded.Cell.Edges), len(out.Best.Arch.Cell.Edges))
	}
	for i, e := range loaded.Cell.Edges {
		orig := out.Best.Arch.Cell.Edges[i]
		if e.From != orig.From || e.To != orig.To || e.Op != orig.Op {
			t.Errorf("edge %d: got {%d,%d,%s}, want {%d,%d,%s}",
				i, e.From, e.To, e.Op, orig.From, orig.To, orig.Op)
		}
	}

	// Verify config round-trip.
	if loadedCfg.HiddenDim != cfg.HiddenDim {
		t.Errorf("HiddenDim: got %d, want %d", loadedCfg.HiddenDim, cfg.HiddenDim)
	}
	if loadedCfg.NumLayers != cfg.NumLayers {
		t.Errorf("NumLayers: got %d, want %d", loadedCfg.NumLayers, cfg.NumLayers)
	}
	if loadedCfg.InputFeatures != cfg.InputFeatures {
		t.Errorf("InputFeatures: got %d, want %d", loadedCfg.InputFeatures, cfg.InputFeatures)
	}
	if loadedCfg.PatchLen != cfg.PatchLen {
		t.Errorf("PatchLen: got %d, want %d", loadedCfg.PatchLen, cfg.PatchLen)
	}
	if loadedCfg.HorizonLen != cfg.HorizonLen {
		t.Errorf("HorizonLen: got %d, want %d", loadedCfg.HorizonLen, cfg.HorizonLen)
	}

	// Verify tensors are loadable.
	tensors, err := gguf.LoadTensors(gf, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	if len(tensors) != len(weights) {
		t.Errorf("tensor count: got %d, want %d", len(tensors), len(weights))
	}

	t.Logf("Full pipeline passed: search -> discretize -> export -> load")
	t.Logf("Architecture: %d nodes, %d edges", loaded.Cell.NumNodes, len(loaded.Cell.Edges))
	for i, e := range loaded.Cell.Edges {
		t.Logf("  edge %d: %d->%d op=%s", i, e.From, e.To, e.Op)
	}
}

func TestNASSignalSearchContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	cfg := SignalSearchConfig{
		NumTrials:     10,
		SearchSteps:   100,
		WeightLR:      0.01,
		AlphaLR:       0.1,
		InputFeatures: 4,
		PatchLen:      8,
		HorizonLen:    4,
		HiddenDim:     32,
		NumLayers:     2,
	}
	data := &syntheticSignalData{size: 8, offset: 0}

	_, err := RunSignalNAS(ctx, cfg, data)
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestNASSignalSearchCustomSearchSpace(t *testing.T) {
	space := NewSearchSpaceWithOps(3, []OpType{OpSkipConnect, OpZero})
	cfg := SignalSearchConfig{
		NumTrials:     1,
		SearchSteps:   5,
		WeightLR:      0.01,
		AlphaLR:       0.1,
		InputFeatures: 4,
		PatchLen:      8,
		HorizonLen:    4,
		HiddenDim:     32,
		NumLayers:     2,
		SearchSpace:   space,
		Seed:          99,
	}
	data := &syntheticSignalData{size: 8, offset: 0}

	out, err := RunSignalNAS(context.Background(), cfg, data)
	if err != nil {
		t.Fatalf("RunSignalNAS with custom space: %v", err)
	}

	// All edges should use ops from the custom space.
	validOps := map[OpType]bool{OpSkipConnect: true, OpZero: true}
	for i, e := range out.Best.Arch.Cell.Edges {
		if !validOps[e.Op] {
			t.Errorf("edge %d has op %s, not in custom search space", i, e.Op)
		}
	}
}

func TestSharpeRatio(t *testing.T) {
	tests := []struct {
		name    string
		returns []float64
		want    float64
		tol     float64
	}{
		{
			name:    "empty",
			returns: nil,
			want:    0,
		},
		{
			name:    "single value",
			returns: []float64{1.0},
			want:    0,
		},
		{
			name:    "constant returns",
			returns: []float64{1.0, 1.0, 1.0},
			want:    0,
		},
		{
			name:    "positive mean positive std",
			returns: []float64{0.1, 0.2, 0.3},
			want:    2.0, // mean=0.2, std=0.1 -> 0.2/0.1 = 2.0
			tol:     0.01,
		},
		{
			name:    "negative mean",
			returns: []float64{-0.1, -0.2, -0.3},
			want:    -2.0,
			tol:     0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SharpeRatio(tt.returns)
			if tt.tol > 0 {
				if math.Abs(got-tt.want) > tt.tol {
					t.Errorf("SharpeRatio(%v) = %f, want %f (tol %f)", tt.returns, got, tt.want, tt.tol)
				}
			} else {
				if got != tt.want {
					t.Errorf("SharpeRatio(%v) = %f, want %f", tt.returns, got, tt.want)
				}
			}
		})
	}
}

func TestDefaultSignalSearchSpace(t *testing.T) {
	space := DefaultSignalSearchSpace()
	if space.NumNodes != 4 {
		t.Errorf("NumNodes: got %d, want 4", space.NumNodes)
	}
	if len(space.Ops) != 4 {
		t.Errorf("num ops: got %d, want 4", len(space.Ops))
	}

	wantOps := map[OpType]bool{
		OpAvgPool3x3:  true,
		OpMaxPool3x3:  true,
		OpSkipConnect: true,
		OpZero:        true,
	}
	for _, op := range space.Ops {
		if !wantOps[op] {
			t.Errorf("unexpected op %s in default signal search space", op)
		}
	}
}

// contains reports whether s contains substr.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
