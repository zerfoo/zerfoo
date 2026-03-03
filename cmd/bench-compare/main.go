// Command bench-compare parses Go benchmark output and compares against a
// baseline file. It fails if any benchmark regresses by more than the
// configured threshold percentage.
//
// Usage:
//
//	go test -bench=. -benchmem -count=3 ./... > new.txt
//	go run ./cmd/bench-compare -baseline benchmarks/baseline.txt -current new.txt -threshold 10
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

func main() {
	baselinePath := flag.String("baseline", "benchmarks/baseline.txt", "path to baseline benchmark file")
	currentPath := flag.String("current", "", "path to current benchmark file")
	threshold := flag.Float64("threshold", 10.0, "regression threshold percentage")
	flag.Parse()

	if *currentPath == "" {
		fmt.Fprintln(os.Stderr, "error: -current flag is required")
		os.Exit(1)
	}

	baseline, err := parseBenchmarks(*baselinePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading baseline: %v\n", err)
		os.Exit(1)
	}

	current, err := parseBenchmarks(*currentPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading current: %v\n", err)
		os.Exit(1)
	}

	failed := false
	for name, baseNS := range baseline {
		curNS, ok := current[name]
		if !ok {
			fmt.Printf("%-50s  SKIP (not in current)\n", name)
			continue
		}

		if baseNS == 0 {
			continue
		}

		pctChange := (curNS - baseNS) / baseNS * 100
		status := "OK"
		if pctChange > *threshold {
			status = fmt.Sprintf("REGRESSION +%.1f%%", pctChange)
			failed = true
		} else if pctChange < -*threshold {
			status = fmt.Sprintf("IMPROVED %.1f%%", pctChange)
		}

		fmt.Printf("%-50s  base=%10.0f ns  cur=%10.0f ns  %s\n", name, baseNS, curNS, status)
	}

	if failed {
		fmt.Fprintf(os.Stderr, "\nbenchmark regression detected: one or more benchmarks regressed by more than %.1f%%\n", *threshold)
		os.Exit(1)
	}

	fmt.Println("\nNo regressions detected.")
}

// parseBenchmarks reads Go benchmark output and returns a map of benchmark name -> ns/op.
// When multiple runs exist for the same benchmark, the median is used.
func parseBenchmarks(path string) (map[string]float64, error) {
	f, err := os.Open(path) //nolint:gosec // caller-provided file path
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer func() { _ = f.Close() }()

	// Collect all ns/op values per benchmark name.
	allValues := make(map[string][]float64)
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "Benchmark") {
			continue
		}

		// Format: BenchmarkName-N  iterations  ns/op  ...
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}

		name := fields[0]
		// Find ns/op value
		for i := 2; i < len(fields)-1; i++ {
			if fields[i+1] == "ns/op" {
				val, err := strconv.ParseFloat(fields[i], 64)
				if err == nil {
					allValues[name] = append(allValues[name], val)
				}
				break
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan: %w", err)
	}

	// Compute median for each benchmark.
	results := make(map[string]float64, len(allValues))
	for name, values := range allValues {
		results[name] = median(values)
	}

	return results, nil
}

// median returns the median of a sorted slice of float64 values.
func median(values []float64) float64 {
	n := len(values)
	if n == 0 {
		return 0
	}

	// Simple insertion sort for small slices.
	sorted := make([]float64, n)
	copy(sorted, values)
	for i := 1; i < n; i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] > key {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}

	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

// approxEqual checks if two floats are approximately equal.
func approxEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}
