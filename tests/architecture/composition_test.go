package architecture

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// rootDir returns the repository root (two levels up from tests/architecture/).
func rootDir(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	return filepath.Join(wd, "..", "..")
}

// relPath returns a repo-relative path for display.
func relPath(root, abs string) string {
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return abs
	}
	return rel
}

// isAllowedPackage returns true for packages that legitimately operate at a
// lower abstraction level and are exempt from composition rules.
func isAllowedPackage(rel string) bool {
	prefixes := []string{
		"internal/",
		"tests/",
		".claude/",
		"vendor/",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(rel, p) {
			return true
		}
	}
	return false
}

// isTestFile returns true if the file is a Go test file.
func isTestFile(path string) bool {
	return strings.HasSuffix(path, "_test.go")
}

// collectGoFiles walks the repo and returns all .go files that are not in
// allowed packages and are not test files.
func collectGoFiles(t *testing.T, root string) []string {
	t.Helper()
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			base := info.Name()
			if base == ".git" || base == "vendor" || base == ".claude" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel := relPath(root, path)
		if isAllowedPackage(rel) || isTestFile(path) {
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	return files
}

// violation records a single architecture violation.
type violation struct {
	File    string
	Line    int
	Message string
}

func (v violation) String() string {
	return fmt.Sprintf("%s:%d: %s", v.File, v.Line, v.Message)
}

// ---------------------------------------------------------------------------
// Check 1: .Data() followed by element-wise arithmetic in for loops
// ---------------------------------------------------------------------------

// dataAbuseAllowlist contains files that have known .Data() + loop arithmetic
// patterns tracked for migration in E70. Format: repo-relative path.
var dataAbuseAllowlist = map[string]bool{
	// training/ — optimizers operate on raw parameter slices by design
	"training/optimizer/adamw.go":          true,
	"training/optimizer/adamw8bit.go":      true,
	"training/adapter.go":                  true,
	"training/strategy_common.go":          true,
	"training/nas/darts_optimizer.go":      true,
	"training/nas/darts_layer.go":          true,
	"training/nas/export.go":               true,
	"training/nas/signal_search.go":        true,
	"training/loss/bce.go":                 true,
	"training/loss/corr.go":                true,
	"training/loss/cross_entropy_loss.go":  true,
	"training/loss/mse.go":                 true,
	"training/loss/quantile.go":            true,
	"training/loss/routing_contrastive.go": true,
	"training/lora/qlora.go":               true,
	"training/lora/checkpoint.go":          true,
	"training/fp8/master_weights.go":       true,
	"training/fp8/linear.go":               true,
	"training/automl/search.go":            true,
	// timeseries/ — legacy direct-slice code tracked for Engine migration
	"timeseries/patchtst.go":                    true,
	"timeseries/patchtst_encoder.go":            true,
	"timeseries/patchtst_engine.go":             true,
	"timeseries/patchtst_gpu_train.go":          true,
	"timeseries/patchtst_backward.go":           true,
	"timeseries/ttm.go":                         true,
	"timeseries/ttm_engine.go":                  true,
	"timeseries/tft.go":                         true,
	"timeseries/nhits.go":                       true,
	"timeseries/nhits_engine.go":                true,
	"timeseries/timemixer_engine.go":            true,
	"timeseries/frets_engine.go":                true,
	"timeseries/itransformer_engine.go":         true,
	"timeseries/itransformer_backward_batch.go": true,
	"timeseries/trainable.go":                   true,
	"timeseries/mamba.go":                       true,
	"timeseries/foundation.go":                  true,
	"timeseries/cfc_engine.go":                  true,
	"timeseries/layernorm_ops.go":               true, // backward pass .Data() — tracked in E67
	// layers/ssm — SSM package uses raw .Data() for stateful recurrences (E70)
	"layers/ssm/mamba_block.go":   true,
	"layers/ssm/s4.go":            true,
	"layers/ssm/bc_norm.go":       true,
	"layers/ssm/complex_state.go": true,
	"layers/ssm/mimo_ssm.go":      true,
	// layers/attention — NSA uses .Data() for fine-grained indexing
	"layers/attention/nsa_fine.go": true,
	// generate/ — sampling operates on logit slices
	"generate/sampling.go": true,
	// inference/ — GGUF loading and architecture builders access raw data
	"inference/load_gguf.go":                 true,
	"inference/sentiment/pipeline.go":        true,
	"inference/arch_bert.go":                 true,
	"inference/arch_gpt2.go":                 true,
	"inference/arch_vision_helpers.go":       true,
	"inference/timeseries/arch_timemixer.go": true,
	"inference/timeseries/arch_ttm.go":       true,
	// model/ — GGUF parsing accesses raw tensor data
	"model/gguf.go": true,
	// cmd/ — CLI entry points may access raw data for debugging
	"cmd/debug-infer/main.go": true,
	"cmd/cli/framework.go":    true,
	"cmd/cli/eagle_train.go":  true,
	"cmd/ts_train/main.go":    true,
	// distributed/ — gradient exchange needs raw buffers
	"distributed/worker.go": true,
	// gnn/ — graph neural nets access adjacency data directly
	"gnn/gcn.go":  true,
	"gnn/sage.go": true,
	// tabular/ — legacy models
	"tabular/model.go":          true,
	"tabular/ft_transformer.go": true,
	"tabular/saint.go":          true,
	// rl/ — PPO accesses raw reward/advantage buffers
	"rl/ppo.go": true,
	// examples/ — demonstration code
	"examples/inference/main.go": true,
}

// checkDataAbuse parses a Go file and detects .Data() calls whose result is
// used in a for-loop body with element-wise arithmetic (+, -, *, /).
func checkDataAbuse(root, path string, fset *token.FileSet, file *ast.File) []violation {
	rel := relPath(root, path)
	if dataAbuseAllowlist[rel] {
		return nil
	}

	var violations []violation

	// Walk all function bodies looking for assignment of .Data() to a variable,
	// then check if that variable appears in a range/for loop with arithmetic.
	ast.Inspect(file, func(n ast.Node) bool {
		fn, ok := n.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			return true
		}

		// Collect variable names assigned from .Data() calls.
		dataVars := map[string]token.Pos{}
		for _, stmt := range fn.Body.List {
			collectDataVars(stmt, dataVars)
		}
		if len(dataVars) == 0 {
			return true
		}

		// Look for range/for loops that use those variables with arithmetic.
		for _, stmt := range fn.Body.List {
			checkLoopArithmetic(fset, rel, stmt, dataVars, &violations)
		}

		return true
	})

	return violations
}

// collectDataVars finds assignments like `d := t.Data()` or `d = t.Data()`.
func collectDataVars(node ast.Node, vars map[string]token.Pos) {
	switch s := node.(type) {
	case *ast.AssignStmt:
		for i, rhs := range s.Rhs {
			if isDataCall(rhs) && i < len(s.Lhs) {
				if ident, ok := s.Lhs[i].(*ast.Ident); ok {
					vars[ident.Name] = ident.Pos()
				}
			}
		}
	case *ast.BlockStmt:
		if s != nil {
			for _, stmt := range s.List {
				collectDataVars(stmt, vars)
			}
		}
	case *ast.IfStmt:
		collectDataVars(s.Body, vars)
		if s.Else != nil {
			collectDataVars(s.Else, vars)
		}
	}
}

// isDataCall checks if an expression is a method call to .Data().
func isDataCall(expr ast.Expr) bool {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	return sel.Sel.Name == "Data"
}

// checkLoopArithmetic checks if a for/range loop body uses a data variable
// with element-wise arithmetic.
func checkLoopArithmetic(fset *token.FileSet, rel string, node ast.Node, dataVars map[string]token.Pos, violations *[]violation) {
	ast.Inspect(node, func(n ast.Node) bool {
		var body *ast.BlockStmt
		switch s := n.(type) {
		case *ast.ForStmt:
			body = s.Body
		case *ast.RangeStmt:
			body = s.Body
		default:
			return true
		}
		if body == nil {
			return true
		}

		// Check if the loop body contains arithmetic on data variables.
		hasArith := false
		ast.Inspect(body, func(inner ast.Node) bool {
			if hasArith {
				return false
			}
			bin, ok := inner.(*ast.BinaryExpr)
			if !ok {
				return true
			}
			switch bin.Op {
			case token.ADD, token.SUB, token.MUL, token.QUO,
				token.ADD_ASSIGN, token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN:
				if usesDataVar(bin, dataVars) {
					hasArith = true
					pos := fset.Position(bin.Pos())
					*violations = append(*violations, violation{
						File:    rel,
						Line:    pos.Line,
						Message: ".Data() result used in element-wise loop arithmetic — use Engine ops instead",
					})
				}
			}
			return true
		})
		return !hasArith
	})
}

// usesDataVar checks if a binary expression references a known .Data() variable
// through indexing (e.g., data[i]).
func usesDataVar(bin *ast.BinaryExpr, dataVars map[string]token.Pos) bool {
	return exprUsesDataVar(bin.X, dataVars) || exprUsesDataVar(bin.Y, dataVars)
}

func exprUsesDataVar(expr ast.Expr, dataVars map[string]token.Pos) bool {
	switch e := expr.(type) {
	case *ast.Ident:
		_, ok := dataVars[e.Name]
		return ok
	case *ast.IndexExpr:
		return exprUsesDataVar(e.X, dataVars)
	case *ast.ParenExpr:
		return exprUsesDataVar(e.X, dataVars)
	}
	return false
}

// ---------------------------------------------------------------------------
// Check 2: import "math" without layers/ import in neural-net packages
// ---------------------------------------------------------------------------

// mathImportAllowlist contains packages that legitimately need math without
// layers/ — e.g., non-neural-net utility packages.
var mathImportAllowlist = map[string]bool{
	// compute/numeric utility code
	"generate/":          true,
	"model/":             true,
	"serve/":             true,
	"cmd/":               true,
	"health/":            true,
	"config/":            true,
	"examples/":          true,
	"benchmarks/":        true,
	"data/":              true,
	"debug-infer/":       true,
	"deploy/":            true,
	"tests/integration/": true,
	"sdk/integrations/":  true,
	"infra/":             true,
	"mobile/":            true,
	"monitor/":           true,
	"provenance/":        true,
	"recover/":           true,
	"registry/":          true,
	"results/":           true,
	"scripts/":           true,
	"security/":          true,
	"shared/":            true,
	"shutdown/":          true,
	"support/":           true,
	"synth/":             true,
	"zerfoo/":            true,
	"model/dsl/":         true,
	"meta/":              true,
	"features/":          true,
	"modelcache/":        true,
	"autoopt/":           true,
	"gguf-info/":         true,
	"testing/":           true,
	// inference/ uses math for architecture builders (sin/cos for RoPE, etc.)
	"inference/": true,
	// training/ uses math for optimizer computations (lr schedules, etc.)
	"training/": true,
	// distributed/ uses math for gradient aggregation
	"distributed/": true,
	// timeseries/ legacy — tracked in E70
	"timeseries/": true,
	// tabular/ legacy
	"tabular/": true,
	// rl/ — reward computations
	"rl/": true,
	// gnn/ — graph computations
	"gnn/": true,
	// gp/ — gaussian processes
	"gp/": true,
	// layers/ sub-packages legitimately need math
	"layers/": true,
	// federated/ uses math for aggregation
	"federated/": true,
	// regime/ uses math for regime detection
	"regime/": true,
	// causal/ uses math for causal inference
	"causal/": true,
	// ts_train/
	"ts_train/": true,
}

// mathImportAllowFiles lists specific root-level files allowed to import math.
var mathImportAllowFiles = map[string]bool{
	// api.go — uses math.MaxInt for pagination limits
	"api.go": true,
	// zerfoo.go — top-level package file
	"zerfoo.go": true,
}

func isMathImportAllowed(rel string) bool {
	if mathImportAllowFiles[rel] {
		return true
	}
	for prefix := range mathImportAllowlist {
		if strings.HasPrefix(rel, prefix) {
			return true
		}
	}
	return false
}

// checkMathWithoutLayers reports files that import "math" but no layers/ package
// in packages doing neural network computation.
func checkMathWithoutLayers(root, path string, file *ast.File) []violation {
	rel := relPath(root, path)
	if isMathImportAllowed(rel) {
		return nil
	}

	hasMath := false
	hasLayers := false
	var mathPos token.Pos

	for _, imp := range file.Imports {
		importPath := strings.Trim(imp.Path.Value, `"`)
		if importPath == "math" {
			hasMath = true
			mathPos = imp.Pos()
		}
		if strings.Contains(importPath, "/layers/") || strings.HasSuffix(importPath, "/layers") {
			hasLayers = true
		}
	}

	if hasMath && !hasLayers {
		return []violation{{
			File:    rel,
			Line:    int(mathPos),
			Message: `imports "math" without layers/ import — may be reimplementing layer operations`,
		}}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Check 3: Private layer reimplementations outside layers/
// ---------------------------------------------------------------------------

// privateLayerAllowlist contains files outside layers/ that legitimately define
// layer-like types or functions. These are tracked for consolidation in E70.
var privateLayerAllowlist = map[string]bool{
	// timeseries/ — legacy layer implementations tracked in E70
	"timeseries/nbeats.go":                true,
	"timeseries/patchtst.go":              true,
	"timeseries/patchtst_encoder.go":      true,
	"timeseries/patchtst_gpu_train.go":    true,
	"timeseries/itransformer.go":          true,
	"timeseries/itransformer_backward.go": true,
	"timeseries/layernorm_ops.go":         true,
	"timeseries/cfc.go":                   true,
	"timeseries/doc.go":                   true,
	"timeseries/foundation.go":            true,
	"timeseries/math_ops.go":              true,
	"timeseries/patchtst_backward.go":     true,
	"timeseries/tft.go":                   true,
	"timeseries/ttm.go":                   true,
	// tabular/ — legacy models
	"tabular/model.go":          true,
	"tabular/ft_transformer.go": true,
	"tabular/saint.go":          true,
	"tabular/ensemble.go":       true,
	"tabular/resnet.go":         true,
	"tabular/save.go":           true,
	// model/dsl/ — model definition DSL
	"model/dsl/model.go": true,
	"model/dsl/train.go": true,
	// rl/ — PPO actor/critic layers
	"rl/ppo.go": true,
	// gnn/ — graph neural network layers
	"gnn/gat.go": true,
	"gnn/gcn.go": true,
	// inference/ — LoRA adapter layers and architecture graph builders
	"inference/lora/adapter.go":              true,
	"inference/arch_bert.go":                 true,
	"inference/arch_falcon.go":               true,
	"inference/arch_vision_helpers.go":       true,
	"inference/fused_add_rmsnorm_node.go":    true,
	"inference/guardian/verdict.go":          true,
	"inference/timeseries/arch_chronos.go":   true,
	"inference/timeseries/arch_timemixer.go": true,
	// generate/ — sampling has its own softmax
	"generate/sampling.go": true,
	// inference/ — sentiment pipeline has layerNorm
	"inference/sentiment/pipeline.go": true,
	// training/ — automl/nas have layer-like helpers
	"training/automl/search.go":   true,
	"training/nas/darts_layer.go": true,
	// cmd/ — debug tool
	"cmd/debug-infer/main.go": true,
}

// layerLikeNames are function/type name patterns that suggest layer reimplementation.
var layerLikeNames = []string{
	"layerNorm",
	"rmsNorm",
	"softmax",
	"mlpLayer",
	"feedForward",
	"multiHeadAttention",
	"groupNorm",
	"batchNorm",
	"instanceNorm",
}

// checkPrivateLayerReimpl detects private layer-like type or function
// definitions outside the layers/ package.
func checkPrivateLayerReimpl(root, path string, fset *token.FileSet, file *ast.File) []violation {
	rel := relPath(root, path)
	if strings.HasPrefix(rel, "layers/") {
		return nil
	}
	if privateLayerAllowlist[rel] {
		return nil
	}

	var violations []violation

	ast.Inspect(file, func(n ast.Node) bool {
		switch decl := n.(type) {
		case *ast.FuncDecl:
			name := decl.Name.Name
			if !decl.Name.IsExported() {
				for _, pattern := range layerLikeNames {
					if strings.Contains(strings.ToLower(name), strings.ToLower(pattern)) {
						pos := fset.Position(decl.Pos())
						violations = append(violations, violation{
							File:    rel,
							Line:    pos.Line,
							Message: fmt.Sprintf("private function %q looks like a layer reimplementation — use layers/ package", name),
						})
						break
					}
				}
			}
		case *ast.GenDecl:
			if decl.Tok != token.TYPE {
				return true
			}
			for _, spec := range decl.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				name := ts.Name.Name
				if !ts.Name.IsExported() {
					for _, pattern := range layerLikeNames {
						if strings.Contains(strings.ToLower(name), strings.ToLower(pattern)) {
							pos := fset.Position(ts.Pos())
							violations = append(violations, violation{
								File:    rel,
								Line:    pos.Line,
								Message: fmt.Sprintf("private type %q looks like a layer reimplementation — use layers/ package", name),
							})
							break
						}
					}
				}
			}
		}
		return true
	})

	return violations
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

func TestDataAbuse(t *testing.T) {
	root := rootDir(t)
	files := collectGoFiles(t, root)
	fset := token.NewFileSet()

	var all []violation
	for _, path := range files {
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			// Skip files with parse errors (e.g. build-tagged files).
			continue
		}
		all = append(all, checkDataAbuse(root, path, fset, f)...)
	}

	for _, v := range all {
		t.Errorf("%s", v)
	}
}

func TestMathWithoutLayers(t *testing.T) {
	root := rootDir(t)
	files := collectGoFiles(t, root)
	fset := token.NewFileSet()

	var all []violation
	for _, path := range files {
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			continue
		}
		all = append(all, checkMathWithoutLayers(root, path, f)...)
	}

	for _, v := range all {
		t.Errorf("%s", v)
	}
}

func TestPrivateLayerReimpl(t *testing.T) {
	root := rootDir(t)
	files := collectGoFiles(t, root)
	fset := token.NewFileSet()

	var all []violation
	for _, path := range files {
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			continue
		}
		all = append(all, checkPrivateLayerReimpl(root, path, fset, f)...)
	}

	for _, v := range all {
		t.Errorf("%s", v)
	}
}
