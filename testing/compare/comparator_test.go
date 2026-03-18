package compare

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// mockInfer returns a deterministic inference function for testing.
// Each model returns metrics scaled by the given factors.
func mockInfer(modelMetrics map[string]map[string]float64) InferFunc {
	return func(ctx context.Context, model ModelSpec, prompt string) (string, map[string]float64, error) {
		metrics, ok := modelMetrics[model.Name]
		if !ok {
			return "", nil, fmt.Errorf("unknown model: %s", model.Name)
		}
		// Copy to avoid mutation.
		m := make(map[string]float64, len(metrics))
		for k, v := range metrics {
			m[k] = v
		}
		return "output for " + prompt, m, nil
	}
}

func TestModelComparator_Compare(t *testing.T) {
	cfg := Config{
		Models: []ModelSpec{
			{Path: "/models/a.gguf", Name: "model-a", Architecture: "llama", Quantization: "Q4_K_M"},
			{Path: "/models/b.gguf", Name: "model-b", Architecture: "gemma", Quantization: "Q8_0"},
		},
		Prompts: []string{"Hello", "World"},
	}

	infer := mockInfer(map[string]map[string]float64{
		"model-a": {"throughput_tok_s": 200, "memory_mb": 512, "perplexity": 5.5},
		"model-b": {"throughput_tok_s": 150, "memory_mb": 768, "perplexity": 4.2},
	})

	mc := NewModelComparator(cfg, infer)
	result, err := mc.Compare(context.Background())
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}

	if len(result.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(result.Results))
	}

	// Verify model-a metrics.
	a := result.Results[0]
	if a.Model.Name != "model-a" {
		t.Errorf("first result model = %q, want model-a", a.Model.Name)
	}
	if got := a.Metrics["throughput_tok_s"]; got != 200 {
		t.Errorf("model-a throughput = %.2f, want 200", got)
	}
	if got := a.Metrics["memory_mb"]; got != 512 {
		t.Errorf("model-a memory = %.2f, want 512", got)
	}

	// Verify per-prompt data.
	if len(a.PerPrompt) != 2 {
		t.Errorf("model-a per-prompt count = %d, want 2", len(a.PerPrompt))
	}

	// Verify rankings exist.
	if len(result.Rankings) == 0 {
		t.Error("expected rankings, got none")
	}

	// Check throughput ranking: model-a (200) > model-b (150).
	for _, r := range result.Rankings {
		if r.Metric == "throughput_tok_s" {
			if len(r.Order) != 2 || r.Order[0] != "model-a" {
				t.Errorf("throughput ranking = %v, want [model-a, model-b]", r.Order)
			}
		}
		// Check perplexity ranking: model-b (4.2) < model-a (5.5) — lower is better.
		if r.Metric == "perplexity" {
			if len(r.Order) != 2 || r.Order[0] != "model-b" {
				t.Errorf("perplexity ranking = %v, want [model-b, model-a]", r.Order)
			}
		}
	}
}

func TestModelComparator_Compare_NoModels(t *testing.T) {
	cfg := Config{
		Prompts: []string{"test"},
	}
	mc := NewModelComparator(cfg, nil)
	_, err := mc.Compare(context.Background())
	if err == nil {
		t.Fatal("expected error for empty models")
	}
	if !strings.Contains(err.Error(), "no models") {
		t.Errorf("error = %q, want 'no models'", err)
	}
}

func TestModelComparator_Compare_NoPrompts(t *testing.T) {
	cfg := Config{
		Models: []ModelSpec{{Name: "test"}},
	}
	mc := NewModelComparator(cfg, nil)
	_, err := mc.Compare(context.Background())
	if err == nil {
		t.Fatal("expected error for empty prompts")
	}
	if !strings.Contains(err.Error(), "no prompts") {
		t.Errorf("error = %q, want 'no prompts'", err)
	}
}

func TestModelComparator_Compare_ContextCanceled(t *testing.T) {
	cfg := Config{
		Models:  []ModelSpec{{Name: "test"}},
		Prompts: []string{"hello"},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mc := NewModelComparator(cfg, func(ctx context.Context, m ModelSpec, p string) (string, map[string]float64, error) {
		return "", nil, nil
	})
	_, err := mc.Compare(ctx)
	if err == nil {
		t.Fatal("expected error for canceled context")
	}
}

func TestModelComparator_Compare_InferError(t *testing.T) {
	cfg := Config{
		Models:  []ModelSpec{{Name: "bad-model"}},
		Prompts: []string{"test"},
	}
	mc := NewModelComparator(cfg, func(ctx context.Context, m ModelSpec, p string) (string, map[string]float64, error) {
		return "", nil, fmt.Errorf("inference failed")
	})
	_, err := mc.Compare(context.Background())
	if err == nil {
		t.Fatal("expected error from failing inference")
	}
	if !strings.Contains(err.Error(), "inference failed") {
		t.Errorf("error = %q, want to contain 'inference failed'", err)
	}
}

func TestModelComparator_Compare_MetricFilter(t *testing.T) {
	cfg := Config{
		Models:  []ModelSpec{{Name: "test-model"}},
		Prompts: []string{"prompt1"},
		Metrics: []string{"throughput_tok_s"},
	}
	infer := mockInfer(map[string]map[string]float64{
		"test-model": {"throughput_tok_s": 100, "memory_mb": 256, "perplexity": 3.0},
	})

	mc := NewModelComparator(cfg, infer)
	result, err := mc.Compare(context.Background())
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}

	r := result.Results[0]
	if _, ok := r.Metrics["throughput_tok_s"]; !ok {
		t.Error("expected throughput_tok_s in filtered metrics")
	}
	// latency_ms is auto-added when not provided by infer, so it should also
	// be filtered out when Metrics is specified (but latency_ms is not listed).
	if _, ok := r.Metrics["memory_mb"]; ok {
		t.Error("memory_mb should be filtered out")
	}
	if _, ok := r.Metrics["perplexity"]; ok {
		t.Error("perplexity should be filtered out")
	}
}

func TestModelComparator_Table(t *testing.T) {
	result := &ComparisonResult{
		Results: []ModelResult{
			{
				Model:   ModelSpec{Name: "model-a", Architecture: "llama", Quantization: "Q4_K_M"},
				Metrics: map[string]float64{"throughput_tok_s": 200, "latency_ms": 50},
				StdDev:  map[string]float64{"throughput_tok_s": 10, "latency_ms": 5},
			},
			{
				Model:   ModelSpec{Name: "model-b", Architecture: "gemma", Quantization: "Q8_0"},
				Metrics: map[string]float64{"throughput_tok_s": 150, "latency_ms": 70},
				StdDev:  map[string]float64{"throughput_tok_s": 8, "latency_ms": 3},
			},
		},
		Rankings: []Ranking{
			{Metric: "throughput_tok_s", Order: []string{"model-a", "model-b"}},
			{Metric: "latency_ms", Order: []string{"model-a", "model-b"}},
		},
	}

	var buf bytes.Buffer
	if err := FormatTable(&buf, result); err != nil {
		t.Fatalf("FormatTable: %v", err)
	}

	out := buf.String()

	// Check header columns.
	if !strings.Contains(out, "Model") {
		t.Error("table missing Model header")
	}
	if !strings.Contains(out, "Arch") {
		t.Error("table missing Arch header")
	}
	if !strings.Contains(out, "Quant") {
		t.Error("table missing Quant header")
	}
	if !strings.Contains(out, "throughput_tok_s") {
		t.Error("table missing throughput_tok_s column")
	}
	if !strings.Contains(out, "latency_ms") {
		t.Error("table missing latency_ms column")
	}

	// Check data rows.
	if !strings.Contains(out, "model-a") {
		t.Error("table missing model-a row")
	}
	if !strings.Contains(out, "model-b") {
		t.Error("table missing model-b row")
	}

	// Check standard deviation display.
	if !strings.Contains(out, "±") {
		t.Error("table missing standard deviation display (±)")
	}

	// Check rankings section.
	if !strings.Contains(out, "Rankings:") {
		t.Error("table missing Rankings section")
	}
	if !strings.Contains(out, "model-a > model-b") {
		t.Errorf("table missing ranking order, got:\n%s", out)
	}
}

func TestModelComparator_Table_Empty(t *testing.T) {
	var buf bytes.Buffer
	if err := FormatTable(&buf, &ComparisonResult{}); err != nil {
		t.Fatalf("FormatTable: %v", err)
	}
	if !strings.Contains(buf.String(), "No results") {
		t.Error("expected 'No results' message for empty result")
	}
}

func TestModelComparator_Table_Nil(t *testing.T) {
	var buf bytes.Buffer
	if err := FormatTable(&buf, nil); err != nil {
		t.Fatalf("FormatTable: %v", err)
	}
	if !strings.Contains(buf.String(), "No results") {
		t.Error("expected 'No results' message for nil result")
	}
}

func TestModelComparator_Metrics(t *testing.T) {
	tests := []struct {
		name           string
		metric         string
		higherIsBetter bool
	}{
		{"throughput tok/s", "throughput_tok_s", true},
		{"latency ms", "latency_ms", false},
		{"memory mb", "memory_mb", false},
		{"perplexity", "perplexity", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := higherIsBetter(tt.metric)
			if got != tt.higherIsBetter {
				t.Errorf("higherIsBetter(%q) = %v, want %v", tt.metric, got, tt.higherIsBetter)
			}
		})
	}
}

func TestModelComparator_JSON(t *testing.T) {
	result := &ComparisonResult{
		Results: []ModelResult{
			{
				Model:     ModelSpec{Name: "model-a", Architecture: "llama", Quantization: "Q4_K_M"},
				Metrics:   map[string]float64{"throughput_tok_s": 200},
				StdDev:    map[string]float64{"throughput_tok_s": 10},
				PerPrompt: []map[string]float64{{"throughput_tok_s": 200}},
			},
		},
		Rankings: []Ranking{
			{Metric: "throughput_tok_s", Order: []string{"model-a"}},
		},
	}

	var buf bytes.Buffer
	if err := FormatJSON(&buf, result); err != nil {
		t.Fatalf("FormatJSON: %v", err)
	}

	// Verify it's valid JSON by unmarshaling back.
	var decoded ComparisonResult
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, buf.String())
	}

	if len(decoded.Results) != 1 {
		t.Errorf("decoded %d results, want 1", len(decoded.Results))
	}
	if decoded.Results[0].Model.Name != "model-a" {
		t.Errorf("decoded model name = %q, want model-a", decoded.Results[0].Model.Name)
	}
	if len(decoded.Rankings) != 1 {
		t.Errorf("decoded %d rankings, want 1", len(decoded.Rankings))
	}
}

func TestModelComparator_StdDev(t *testing.T) {
	cfg := Config{
		Models:  []ModelSpec{{Name: "test"}},
		Prompts: []string{"a", "b", "c"},
	}

	callNum := 0
	values := []float64{10, 20, 30}
	infer := func(ctx context.Context, m ModelSpec, p string) (string, map[string]float64, error) {
		v := values[callNum]
		callNum++
		return "", map[string]float64{"metric": v}, nil
	}

	mc := NewModelComparator(cfg, infer)
	result, err := mc.Compare(context.Background())
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}

	r := result.Results[0]
	// Mean should be 20.
	if got := r.Metrics["metric"]; got != 20 {
		t.Errorf("mean = %.2f, want 20", got)
	}
	// StdDev of [10, 20, 30] with mean 20: sqrt(((100+0+100)/3)) = sqrt(200/3) ≈ 8.16
	sd := r.StdDev["metric"]
	if sd < 8.1 || sd > 8.2 {
		t.Errorf("stddev = %.4f, want ~8.165", sd)
	}
}
