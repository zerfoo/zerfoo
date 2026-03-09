package codegen

import (
	"testing"
)

// tracedOps lists every op name that makeTracedForward in graph/compile.go
// can produce. Keep this in sync with the switch statement there.
var tracedOps = []string{
	// Binary elementwise
	"Add", "Sub", "Mul", "Div", "Pow",
	// Unary elementwise
	"Exp", "Log", "Tanh", "Sqrt", "Rsqrt",
	// Scalar ops
	"MulScalar", "AddScalar", "DivScalar",
	// Reductions
	"Softmax", "ReduceSum", "ReduceMean", "Sum",
	// Memory / GEMM
	"MatMul",
	// Shape ops
	"Reshape", "Transpose", "Concat", "Repeat",
}

func TestEmitterCoverage(t *testing.T) {
	var missing []string
	for _, op := range tracedOps {
		if !Supported(op) {
			missing = append(missing, op)
		}
	}

	// Print a coverage report regardless of pass/fail.
	t.Logf("Traced ops: %d", len(tracedOps))
	t.Logf("Emitter table size: %d", len(emitters))

	covered := len(tracedOps) - len(missing)
	t.Logf("Coverage: %d/%d traced ops have emitters (%.0f%%)",
		covered, len(tracedOps),
		100*float64(covered)/float64(len(tracedOps)))

	if len(missing) > 0 {
		for _, op := range missing {
			t.Errorf("traced op %q has no emitter in optable.go", op)
		}
	}

	// Also report emitter-only ops (in table but not traced) as info.
	tracedSet := make(map[string]bool, len(tracedOps))
	for _, op := range tracedOps {
		tracedSet[op] = true
	}
	var extra []string
	for op := range emitters {
		if !tracedSet[op] {
			extra = append(extra, op)
		}
	}
	if len(extra) > 0 {
		t.Logf("Emitter-only ops (in table but not in traced switch): %v", extra)
	}

	// Summary table
	t.Log("")
	t.Log("--- Emitter Coverage Report ---")
	t.Logf("%-16s %s", "Op", "Status")
	t.Logf("%-16s %s", "---", "------")
	for _, op := range tracedOps {
		status := "OK"
		if !Supported(op) {
			status = "MISSING"
		}
		t.Logf("%-16s %s", op, status)
	}
}
