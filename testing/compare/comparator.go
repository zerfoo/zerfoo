// Package compare provides a model comparison tool that runs the same prompts
// through multiple models and compares their performance metrics.
package compare

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"text/tabwriter"
	"time"
)

// ModelSpec identifies a model to compare.
type ModelSpec struct {
	Path         string `json:"path"`
	Name         string `json:"name"`
	Architecture string `json:"architecture"`
	Quantization string `json:"quantization"`
}

// Config controls a model comparison run.
type Config struct {
	Models  []ModelSpec `json:"models"`
	Prompts []string    `json:"prompts"`
	Metrics []string    `json:"metrics"`
}

// InferFunc is the function signature for running inference on a model.
// It takes a model spec and prompt, and returns the generated text along with
// collected metrics (throughput_tok_s, latency_ms, memory_mb, perplexity).
type InferFunc func(ctx context.Context, model ModelSpec, prompt string) (text string, metrics map[string]float64, err error)

// ModelResult holds the aggregated results for a single model.
type ModelResult struct {
	Model   ModelSpec          `json:"model"`
	Metrics map[string]float64 `json:"metrics"`
	StdDev  map[string]float64 `json:"std_dev"`
	// PerPrompt holds the raw metrics for each prompt.
	PerPrompt []map[string]float64 `json:"per_prompt"`
}

// Ranking holds a metric name and the ranked list of model names from best to worst.
type Ranking struct {
	Metric string   `json:"metric"`
	Order  []string `json:"order"`
}

// ComparisonResult holds the full comparison output.
type ComparisonResult struct {
	Results  []ModelResult `json:"results"`
	Rankings []Ranking     `json:"rankings"`
}

// ModelComparator compares multiple models on the same set of prompts.
type ModelComparator struct {
	config Config
	infer  InferFunc
}

// NewModelComparator creates a new comparator with the given config and
// inference function.
func NewModelComparator(cfg Config, infer InferFunc) *ModelComparator {
	return &ModelComparator{
		config: cfg,
		infer:  infer,
	}
}

// Compare runs all prompts through all models and returns the comparison result.
func (mc *ModelComparator) Compare(ctx context.Context) (*ComparisonResult, error) {
	if len(mc.config.Models) == 0 {
		return nil, fmt.Errorf("compare: no models specified")
	}
	if len(mc.config.Prompts) == 0 {
		return nil, fmt.Errorf("compare: no prompts specified")
	}

	metricsSet := make(map[string]bool, len(mc.config.Metrics))
	for _, m := range mc.config.Metrics {
		metricsSet[m] = true
	}
	filterMetrics := len(metricsSet) > 0

	var results []ModelResult
	for _, model := range mc.config.Models {
		mr := ModelResult{
			Model:     model,
			Metrics:   make(map[string]float64),
			StdDev:    make(map[string]float64),
			PerPrompt: make([]map[string]float64, 0, len(mc.config.Prompts)),
		}

		// Collect raw metrics per prompt.
		allMetrics := make(map[string][]float64)
		for _, prompt := range mc.config.Prompts {
			if err := ctx.Err(); err != nil {
				return nil, err
			}

			start := time.Now()
			_, metrics, err := mc.infer(ctx, model, prompt)
			elapsed := time.Since(start)
			if err != nil {
				return nil, fmt.Errorf("compare: model %q prompt %q: %w", model.Name, prompt, err)
			}

			// If latency_ms was not provided by the infer function, use wall-clock time.
			if _, ok := metrics["latency_ms"]; !ok {
				metrics["latency_ms"] = float64(elapsed.Milliseconds())
			}

			promptMetrics := make(map[string]float64)
			for k, v := range metrics {
				if filterMetrics && !metricsSet[k] {
					continue
				}
				promptMetrics[k] = v
				allMetrics[k] = append(allMetrics[k], v)
			}
			mr.PerPrompt = append(mr.PerPrompt, promptMetrics)
		}

		// Compute mean and standard deviation per metric.
		for metric, values := range allMetrics {
			mean := computeMean(values)
			mr.Metrics[metric] = mean
			mr.StdDev[metric] = computeStdDev(values, mean)
		}

		results = append(results, mr)
	}

	// Determine all metrics present across results.
	metricNames := collectMetricNames(results)

	// Rank models per metric.
	rankings := rankModels(results, metricNames)

	return &ComparisonResult{
		Results:  results,
		Rankings: rankings,
	}, nil
}

// FormatTable writes a comparison table to w using text/tabwriter.
func FormatTable(w io.Writer, cr *ComparisonResult) error {
	if cr == nil || len(cr.Results) == 0 {
		_, err := fmt.Fprintln(w, "No results to display.")
		return err
	}

	metricNames := collectMetricNames(cr.Results)

	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)

	// Header: Model | Arch | Quant | metric1 | metric2 | ...
	header := []string{"Model", "Arch", "Quant"}
	for _, m := range metricNames {
		header = append(header, m)
	}
	fmt.Fprintln(tw, strings.Join(header, "\t"))

	// Separator.
	sep := make([]string, len(header))
	for i := range sep {
		sep[i] = strings.Repeat("-", len(header[i]))
	}
	fmt.Fprintln(tw, strings.Join(sep, "\t"))

	// Data rows.
	for _, r := range cr.Results {
		row := []string{r.Model.Name, r.Model.Architecture, r.Model.Quantization}
		for _, m := range metricNames {
			val, ok := r.Metrics[m]
			sd := r.StdDev[m]
			if ok {
				row = append(row, fmt.Sprintf("%.2f (±%.2f)", val, sd))
			} else {
				row = append(row, "---")
			}
		}
		fmt.Fprintln(tw, strings.Join(row, "\t"))
	}

	if err := tw.Flush(); err != nil {
		return err
	}

	// Rankings section.
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Rankings:")
	for _, rank := range cr.Rankings {
		fmt.Fprintf(w, "  %s: %s\n", rank.Metric, strings.Join(rank.Order, " > "))
	}

	return nil
}

// FormatJSON writes the comparison result as indented JSON to w.
func FormatJSON(w io.Writer, cr *ComparisonResult) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(cr)
}

// collectMetricNames returns a sorted, deduplicated list of metric names
// present across all results.
func collectMetricNames(results []ModelResult) []string {
	seen := make(map[string]bool)
	for _, r := range results {
		for k := range r.Metrics {
			seen[k] = true
		}
	}
	names := make([]string, 0, len(seen))
	for k := range seen {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// higherIsBetter returns true for metrics where a higher value indicates
// better performance.
func higherIsBetter(metric string) bool {
	m := strings.ToLower(metric)
	if strings.Contains(m, "tok") && strings.Contains(m, "s") {
		return true
	}
	if strings.Contains(m, "throughput") || strings.Contains(m, "gflops") || strings.Contains(m, "mb/s") {
		return true
	}
	// latency, memory, perplexity: lower is better.
	return false
}

// rankModels ranks models for each metric. For throughput-style metrics,
// higher is better. For latency/memory/perplexity, lower is better.
func rankModels(results []ModelResult, metricNames []string) []Ranking {
	rankings := make([]Ranking, 0, len(metricNames))
	for _, metric := range metricNames {
		type modelVal struct {
			name string
			val  float64
		}
		var mvs []modelVal
		for _, r := range results {
			if v, ok := r.Metrics[metric]; ok {
				mvs = append(mvs, modelVal{name: r.Model.Name, val: v})
			}
		}
		hib := higherIsBetter(metric)
		sort.Slice(mvs, func(i, j int) bool {
			if hib {
				return mvs[i].val > mvs[j].val
			}
			return mvs[i].val < mvs[j].val
		})
		order := make([]string, len(mvs))
		for i, mv := range mvs {
			order[i] = mv.name
		}
		rankings = append(rankings, Ranking{
			Metric: metric,
			Order:  order,
		})
	}
	return rankings
}

func computeMean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func computeStdDev(vals []float64, mean float64) float64 {
	if len(vals) < 2 {
		return 0
	}
	var sumSq float64
	for _, v := range vals {
		d := v - mean
		sumSq += d * d
	}
	return math.Sqrt(sumSq / float64(len(vals)))
}
