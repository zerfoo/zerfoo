// Command coverage-gate reads a Go coverage profile and fails if any testable
// package drops below the configured coverage threshold.
//
// Usage:
//
//	go test -coverprofile=coverage.out ./...
//	go run ./cmd/coverage-gate -profile coverage.out -threshold 93
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func main() {
	profile := flag.String("profile", "coverage.out", "path to coverage profile")
	threshold := flag.Float64("threshold", 93.0, "minimum coverage percentage")
	exclude := flag.String("exclude", "", "comma-separated package path prefixes to exclude")
	flag.Parse()

	var excludePrefixes []string
	if *exclude != "" {
		excludePrefixes = strings.Split(*exclude, ",")
	}

	if err := run(*profile, *threshold, excludePrefixes, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// run contains the core logic: parse coverage profile, check each package
// against the threshold, and report results. Returns an error if any package
// is below the threshold.
func run(profilePath string, threshold float64, excludePrefixes []string, w io.Writer) error {
	results, err := parseCoverageProfile(profilePath)
	if err != nil {
		return fmt.Errorf("error: %w", err)
	}

	failed := false
	for pkg, cov := range results {
		if isExcluded(pkg, excludePrefixes) {
			_, _ = fmt.Fprintf(w, "%-60s %5.1f%%  SKIP\n", pkg, cov)
			continue
		}
		status := "PASS"
		if cov < threshold {
			status = "FAIL"
			failed = true
		}
		_, _ = fmt.Fprintf(w, "%-60s %5.1f%%  %s\n", pkg, cov, status)
	}

	if failed {
		return fmt.Errorf("coverage gate failed: one or more packages below %.1f%%", threshold)
	}

	_, _ = fmt.Fprintf(w, "\nAll packages at or above %.1f%% coverage.\n", threshold)
	return nil
}

// packageCoverage tracks statement counts for a single package.
type packageCoverage struct {
	stmts   int
	covered int
}

// parseCoverageProfile reads a Go coverage profile and returns per-package coverage percentages.
func parseCoverageProfile(path string) (map[string]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open profile: %w", err)
	}
	defer func() { _ = f.Close() }()

	pkgs := make(map[string]*packageCoverage)
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := scanner.Text()
		// Skip the mode line
		if strings.HasPrefix(line, "mode:") {
			continue
		}

		// Format: file:startLine.startCol,endLine.endCol numStatements count
		parts := strings.Fields(line)
		if len(parts) != 3 {
			continue
		}

		colonIdx := strings.LastIndex(parts[0], ":")
		if colonIdx < 0 {
			continue
		}
		filePath := parts[0][:colonIdx]

		// Extract package from file path (everything up to the last /)
		slashIdx := strings.LastIndex(filePath, "/")
		pkg := filePath
		if slashIdx >= 0 {
			pkg = filePath[:slashIdx]
		}

		numStmts, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		count, err := strconv.Atoi(parts[2])
		if err != nil {
			continue
		}

		pc, ok := pkgs[pkg]
		if !ok {
			pc = &packageCoverage{}
			pkgs[pkg] = pc
		}
		pc.stmts += numStmts
		if count > 0 {
			pc.covered += numStmts
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan profile: %w", err)
	}

	results := make(map[string]float64, len(pkgs))
	for pkg, pc := range pkgs {
		if pc.stmts > 0 {
			results[pkg] = float64(pc.covered) / float64(pc.stmts) * 100
		}
	}

	return results, nil
}

// isExcluded returns true if pkg matches any of the exclude prefixes.
func isExcluded(pkg string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.Contains(pkg, strings.TrimSpace(p)) {
			return true
		}
	}
	return false
}
