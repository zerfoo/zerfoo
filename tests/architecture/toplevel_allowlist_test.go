package architecture

import (
	"os"
	"sort"
	"strings"
	"testing"
)

// TestTopLevelAllowlist enforces the post-E124 package layout from
// docs/design.md section 2.1 ("Package Layout"). Adding a new top-level
// directory at the repo root requires either:
//
//  1. An ADR slug referenced in the new package's doc.go AND an entry
//     added to allowedTopLevel below, OR
//  2. Moving the new package under an existing approved subtree
//     (e.g. layers/<family>/, training/<area>/, serve/<area>/,
//     inference/<area>/, model/<area>/, internal/<area>/, tests/<area>/).
//
// Do not silently expand the allowlist. New entries must be justified
// in the commit message and ideally backed by an ADR.
//
// See T124.1.3 in docs/plan.md and docs/design.md section 2.1.
func TestTopLevelAllowlist(t *testing.T) {
	// allowedTopLevel is the union of:
	//   - sanctioned packages from docs/design.md section 2.1
	//     (cmd, model, layers, training, distributed, config, metrics,
	//      log, inference, generate, serve, data, internal, tests, sdk)
	//   - infra/tooling dirs explicitly excluded from the package count
	//     (docs, examples, scripts, benchmarks, bin, deploy, infra)
	//   - pending-migration dirs still tracked under E124 sub-tasks;
	//     entries marked PENDING must be removed once the migration
	//     task lands. Each PENDING entry names its blocking task.
	allowedTopLevel := []string{
		// design.md section 2.1 sanctioned packages
		"cmd",
		"config",
		"data",
		"distributed",
		"generate",
		"inference",
		"internal",
		"layers",
		"log",
		"metrics",
		"model",
		"sdk",
		"serve",
		"tests",
		"training",
		// infra/tooling explicitly excluded from the count
		"benchmarks",
		"bin",
		"deploy",
		"docs",
		"examples",
		"infra",
		"scripts",
		// PENDING migrations (E124) -- remove once each task lands
		"results",      // PENDING -- benchmark output, candidate for benchmarks/results
		"tabular",      // PENDING -- candidate for layers/tabular per E62 work
		"timeseries",   // PENDING E76 -- candidate for inference/timeseries or layers/timeseries
	}

	// Required allowed entries (sanctioned by design.md). These must
	// always be valid even if currently empty on disk; this guards
	// against silent typos in the allowlist itself.
	required := []string{
		"cmd", "internal", "layers", "model", "tests",
	}
	for _, r := range required {
		if !contains(allowedTopLevel, r) {
			t.Fatalf("allowedTopLevel is missing required entry %q (typo?)", r)
		}
	}

	root := rootDir(t)
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("read repo root %s: %v", root, err)
	}

	allowed := map[string]struct{}{}
	for _, name := range allowedTopLevel {
		allowed[name] = struct{}{}
	}

	var violations []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			// hidden dirs (.git, .github, .claude, ...) are ignored
			continue
		}
		if _, ok := allowed[name]; !ok {
			violations = append(violations, name)
		}
	}

	if len(violations) > 0 {
		sort.Strings(violations)
		t.Fatalf(
			"unsanctioned top-level package(s) detected: %s\n"+
				"New top-level Go packages require an ADR slug in the "+
				"package's doc.go and an entry in allowedTopLevel "+
				"(see docs/design.md section 2.1 and T124.1.3). "+
				"Prefer relocating under an existing approved subtree "+
				"(layers/, training/, serve/, inference/, model/, internal/, tests/).",
			strings.Join(violations, ", "),
		)
	}
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}
