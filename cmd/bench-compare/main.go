// Command bench-compare compares two NDJSON benchmark result files and outputs
// a markdown regression report. It exits with code 1 if any metric regresses
// by more than 5%.
//
// Usage:
//
//	bench-compare <previous.json> <current.json>
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// BenchResult represents a single benchmark metric line in NDJSON.
type BenchResult struct {
	Metric string  `json:"metric"`
	Value  float64 `json:"value"`
	Unit   string  `json:"unit,omitempty"`
}

const threshold = 5.0

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "usage: bench-compare <previous.json> <current.json>\n")
		os.Exit(2)
	}

	prev, err := readResults(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading previous: %v\n", err)
		os.Exit(2)
	}

	curr, err := readResults(os.Args[2])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading current: %v\n", err)
		os.Exit(2)
	}

	report, regressed := compare(prev, curr)
	fmt.Print(report)

	if regressed {
		os.Exit(1)
	}
}

// readResults parses an NDJSON file into a map of metric name -> BenchResult.
// Returns an empty map (no error) if the file is empty or does not exist.
func readResults(path string) (map[string]BenchResult, error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer func() { _ = f.Close() }()

	results := make(map[string]BenchResult)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var r BenchResult
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			continue
		}
		if r.Metric != "" {
			results[r.Metric] = r
		}
	}
	return results, scanner.Err()
}

// comparison holds the result of comparing a single metric.
type comparison struct {
	Metric   string
	Previous float64
	Current  float64
	PrevUnit string
	CurrUnit string
	Change   float64
	Status   string
}

// compare generates a markdown report and returns whether any regression was detected.
func compare(prev, curr map[string]BenchResult) (string, bool) {
	if len(prev) == 0 {
		return "No previous baseline -- skipping comparison.\n", false
	}
	if len(curr) == 0 {
		return "No current results to compare.\n", false
	}

	var comparisons []comparison
	regressed := false

	// Gather all metric names from current results.
	metrics := make([]string, 0, len(curr))
	for m := range curr {
		metrics = append(metrics, m)
	}
	sort.Strings(metrics)

	for _, metric := range metrics {
		cr := curr[metric]
		pr, hasPrev := prev[metric]

		c := comparison{
			Metric:   metric,
			Current:  cr.Value,
			CurrUnit: cr.Unit,
		}

		if !hasPrev {
			c.Status = "NEW"
			c.Change = 0
			comparisons = append(comparisons, c)
			continue
		}

		c.Previous = pr.Value
		c.PrevUnit = pr.Unit

		if pr.Value == 0 {
			c.Status = "OK"
			comparisons = append(comparisons, c)
			continue
		}

		change := (cr.Value - pr.Value) / math.Abs(pr.Value) * 100
		c.Change = change

		if math.Abs(change) <= threshold {
			c.Status = "OK"
		} else {
			// Determine if this change is a regression based on metric direction.
			// For timing metrics (ns/op), higher is worse (regression).
			// For throughput metrics (tok/s, GFLOPS, MB/s), lower is worse.
			isRegression := isRegressionChange(metric, cr.Unit, change)
			if isRegression {
				c.Status = "REGRESSION"
				regressed = true
			} else {
				c.Status = "IMPROVED"
			}
		}

		comparisons = append(comparisons, c)
	}

	return formatMarkdown(comparisons, regressed), regressed
}

// isRegressionChange determines if a change percentage represents a regression.
func isRegressionChange(metric, unit string, change float64) bool {
	u := strings.ToLower(unit)
	m := strings.ToLower(metric)

	// For timing metrics, higher is worse.
	if strings.Contains(u, "ns") || strings.Contains(u, "ms") ||
		strings.Contains(u, "alloc") || strings.Contains(u, "b/op") ||
		strings.Contains(m, "latency") || strings.Contains(m, "ns_op") {
		return change > 0
	}

	// For throughput metrics, lower is worse.
	if strings.Contains(u, "tok/s") || strings.Contains(u, "gflops") ||
		strings.Contains(u, "mb/s") || strings.Contains(u, "ops/s") ||
		strings.Contains(m, "throughput") || strings.Contains(m, "tok_s") ||
		strings.Contains(m, "gflops") {
		return change < 0
	}

	// Default: treat increase as regression (conservative for ns/op style).
	return change > 0
}

// formatMarkdown produces a markdown table from comparisons.
func formatMarkdown(comparisons []comparison, regressed bool) string {
	var b strings.Builder

	if regressed {
		b.WriteString("## Benchmark Regression Detected\n\n")
	} else {
		b.WriteString("## Benchmark Comparison\n\n")
	}

	b.WriteString("| Metric | Previous | Current | Change | Status |\n")
	b.WriteString("|--------|----------|---------|--------|--------|\n")

	for _, c := range comparisons {
		unit := c.CurrUnit
		if unit == "" {
			unit = c.PrevUnit
		}

		var prevStr, changeStr string
		if c.Status == "NEW" {
			prevStr = "---"
			changeStr = "---"
		} else {
			prevStr = formatValue(c.Previous, unit)
			changeStr = fmt.Sprintf("%+.1f%%", c.Change)
		}

		status := c.Status
		switch status {
		case "REGRESSION":
			status = "REGRESSION"
		case "IMPROVED":
			status = "IMPROVED"
		}

		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s |\n",
			c.Metric, prevStr, formatValue(c.Current, unit), changeStr, status)
	}

	b.WriteString("\n")
	return b.String()
}

func formatValue(v float64, unit string) string {
	if unit != "" {
		return fmt.Sprintf("%.2f %s", v, unit)
	}
	return fmt.Sprintf("%.2f", v)
}
