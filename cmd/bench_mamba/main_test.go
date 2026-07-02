package main

import (
	"testing"
)

func TestParseFlags(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    ModelConfig
		wantErr bool
	}{
		{
			name: "defaults",
			args: []string{},
			want: ModelConfig{
				Layers:   24,
				DModel:   2048,
				DState:   16,
				DInner:   4096,
				DConv:    4,
				Heads:    16,
				HeadDim:  128,
				GPUTFlop: 150,
			},
		},
		{
			name: "custom",
			args: []string{
				"--layers", "32",
				"--d-model", "4096",
				"--d-state", "64",
				"--d-inner", "8192",
				"--heads", "32",
				"--head-dim", "128",
				"--gpu-tflops", "300",
			},
			want: ModelConfig{
				Layers:   32,
				DModel:   4096,
				DState:   64,
				DInner:   8192,
				DConv:    4,
				Heads:    32,
				HeadDim:  128,
				GPUTFlop: 300,
			},
		},
		{
			name:    "invalid flag",
			args:    []string{"--nonexistent", "value"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _, err := parseFlags(tt.args)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseFlags() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if got != tt.want {
				t.Errorf("parseFlags() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestMambaDecodeFLOPs(t *testing.T) {
	cfg := ModelConfig{
		Layers: 24,
		DModel: 2048,
		DState: 16,
		DInner: 4096,
		DConv:  4,
	}
	flops := mambaDecodeFLOPs(cfg)
	if flops <= 0 {
		t.Fatalf("mambaDecodeFLOPs = %v, want > 0", flops)
	}
	// FLOPs should be independent of sequence length — call twice with same config.
	flops2 := mambaDecodeFLOPs(cfg)
	if flops != flops2 {
		t.Errorf("mambaDecodeFLOPs not deterministic: %v != %v", flops, flops2)
	}
}

func TestTransformerDecodeFLOPs(t *testing.T) {
	cfg := ModelConfig{
		Layers:  24,
		DModel:  2048,
		Heads:   16,
		HeadDim: 128,
	}
	f512 := transformerDecodeFLOPs(cfg, 512)
	f2048 := transformerDecodeFLOPs(cfg, 2048)
	f8192 := transformerDecodeFLOPs(cfg, 8192)

	if f512 <= 0 {
		t.Fatalf("transformerDecodeFLOPs(512) = %v, want > 0", f512)
	}
	// FLOPs should increase with sequence length (attention is O(n)).
	if f2048 <= f512 {
		t.Errorf("transformerDecodeFLOPs(2048) = %v should be > (512) = %v", f2048, f512)
	}
	if f8192 <= f2048 {
		t.Errorf("transformerDecodeFLOPs(8192) = %v should be > (2048) = %v", f8192, f2048)
	}
}

func TestMambaBenchmark(t *testing.T) {
	cfg := ModelConfig{
		Layers:   24,
		DModel:   2048,
		DState:   16,
		DInner:   4096,
		DConv:    4,
		Heads:    16,
		HeadDim:  128,
		GPUTFlop: 150,
	}

	report := runBenchmark(cfg)

	if len(report.Results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(report.Results))
	}

	// Verify sequence lengths.
	expectedSeqs := []int{512, 2048, 8192}
	for i, want := range expectedSeqs {
		if report.Results[i].SeqLen != want {
			t.Errorf("result[%d].SeqLen = %d, want %d", i, report.Results[i].SeqLen, want)
		}
	}

	// All results should have positive throughput.
	for i, res := range report.Results {
		if res.MambaTokPerSec <= 0 {
			t.Errorf("result[%d] MambaTokPerSec = %v, want > 0", i, res.MambaTokPerSec)
		}
		if res.TransTokPerSec <= 0 {
			t.Errorf("result[%d] TransTokPerSec = %v, want > 0", i, res.TransTokPerSec)
		}
		if res.Speedup <= 0 {
			t.Errorf("result[%d] Speedup = %v, want > 0", i, res.Speedup)
		}
	}

	// Mamba speedup should increase with sequence length (O(1) vs O(n)).
	for i := 1; i < len(report.Results); i++ {
		if report.Results[i].Speedup <= report.Results[i-1].Speedup {
			t.Errorf("speedup should increase: seq=%d (%.2fx) <= seq=%d (%.2fx)",
				report.Results[i].SeqLen, report.Results[i].Speedup,
				report.Results[i-1].SeqLen, report.Results[i-1].Speedup)
		}
	}

	// At seq=8192, Mamba should show >= 1.0x speedup (acceptance criteria: > 1.0x).
	last := report.Results[len(report.Results)-1]
	if last.Speedup <= 1.0 {
		t.Errorf("Mamba speedup at seq=8192 = %.2fx, want > 1.0x", last.Speedup)
	}

	// Verify printReport does not panic.
	printReport(report)
}

func TestTokPerSec(t *testing.T) {
	tps := tokPerSec(1e9, 150)
	if tps <= 0 {
		t.Errorf("tokPerSec = %v, want > 0", tps)
	}
	// Higher FLOPs per token should give lower tok/s.
	tps2 := tokPerSec(2e9, 150)
	if tps2 >= tps {
		t.Errorf("doubling FLOPs should halve tok/s: %v >= %v", tps2, tps)
	}
}
