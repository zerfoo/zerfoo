// Harness-honesty tests for the verified-model matrix (T136.6 / S136.6.1).
//
// These tests are host-side path resolution only: no GPU, no CUDA build tag,
// no multi-gigabyte load. They exist because the flagship GGUFs are staged
// FLAT in one directory, which made every parity row resolve to the same file.
package parity_test

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/model/registry"
	"github.com/zerfoo/zerfoo/tests/parity/modelset"
	"github.com/zerfoo/zerfoo/tests/parity/testutil"
)

// defaultStagedModelsDir is where the GB10 stages the flagship GGUFs, mounted
// read-only into the validation pod by docs/bench/manifests/validate-arm64.yaml.
const defaultStagedModelsDir = "/var/lib/zerfoo/models"

// allParityConfigs is every model-family parity suite in this package. The
// matrix-honesty tests below iterate it so a new suite cannot be added without
// pinning a matrix row.
var allParityConfigs = []testutil.ModelParityConfig{
	gemma3Config,
	gemma3nConfig,
	gemma3Q4Config,
	llama3Config,
	llama4Config,
	mistralConfig,
	mixtralConfig,
	deepseekV3Config,
	qwenConfig,
	phi4Config,
	commandRConfig,
	falconConfig,
	rwkvConfig,
}

// writeStubGGUF writes a file with a recognizable, deliberately invalid magic
// number. The magic comes back verbatim in the loader's error message, which
// makes it possible to observe WHICH file a code path actually opened.
func writeStubGGUF(t *testing.T, dir, name string, magic uint32) string {
	t.Helper()

	buf := make([]byte, 64)
	binary.LittleEndian.PutUint32(buf[0:4], magic)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, buf, 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}

	return path
}

// TestDirectoryScanCollapsesDistinctRowsOntoOneFile is the red-proof for
// T136.6, kept as a permanent witness of why parity must never resolve a model
// from a directory.
//
// It drives the exact pre-T136.6 production path -- inference.Load with a
// directory-backed registry, which funnels into findGGUF (inference.go:317,
// "first *.gguf wins") -- with two DIFFERENT model IDs pointed at one flat
// directory holding two DIFFERENT models. Both loads open the same bytes.
//
// The observation is the GGUF magic echoed in the error: the DeepSeek-named
// stub carries 0x41414141 and the Llama-named stub carries 0x42424242, so an
// error quoting 0x41414141 for the Llama row proves the Llama file was never
// opened.
//
// If this test ever fails because inference.Load stopped scanning directories,
// the hazard is gone and the test should be deleted along with this comment.
func TestDirectoryScanCollapsesDistinctRowsOntoOneFile(t *testing.T) {
	dir := t.TempDir()
	const (
		firstMagic  = 0x41414141
		secondMagic = 0x42424242
	)
	// Names chosen to match the real staging: alphabetically, the DeepSeek
	// distill sorts first in /var/lib/zerfoo/models.
	writeStubGGUF(t, dir, "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", firstMagic)
	writeStubGGUF(t, dir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf", secondMagic)

	sawSecond := false
	for _, modelID := range []string{"deepseek-r1-distill", "llama-3"} {
		reg := &testutil.DirRegistry{Models: map[string]*registry.ModelInfo{
			modelID: {ID: modelID, Path: dir},
		}}

		_, err := inference.Load(modelID, inference.WithRegistry(reg))
		if err == nil {
			t.Fatalf("model %q: stub GGUF loaded successfully; the fixture is wrong", modelID)
		}
		t.Logf("model %q -> %v", modelID, err)

		switch {
		case strings.Contains(err.Error(), "0x41414141"):
			// Opened the alphabetically-first (DeepSeek-named) file.
		case strings.Contains(err.Error(), "0x42424242"):
			sawSecond = true
		default:
			t.Fatalf("model %q: unexpected error, cannot tell which file was read: %v", modelID, err)
		}
	}

	if sawSecond {
		t.Fatal("directory-scan resolution distinguished the two models; " +
			"the vacuous-parity hazard appears fixed in inference.Load -- " +
			"delete this red-proof and its comment")
	}
	t.Log("RED-PROOF: two distinct model rows pointed at one flat directory both " +
		"read the alphabetically-first GGUF; a parity matrix built on directory " +
		"scanning proves nothing")
}

// TestMatrixResolverKeepsFlatDirRowsDistinct is the green counterpart: the same
// flat directory, resolved through the checked-in row -> filename table, yields
// a different file for every staged row.
func TestMatrixResolverKeepsFlatDirRowsDistinct(t *testing.T) {
	m, err := modelset.Default()
	if err != nil {
		t.Fatalf("load matrix: %v", err)
	}

	dir := t.TempDir()
	staged := 0
	for i, row := range m.Rows {
		if !row.Staged() {
			continue
		}
		// Distinct contents per file so a collapse would be observable.
		writeStubGGUF(t, dir, row.File, uint32(0x41414141+i))
		staged++
	}
	if staged < 2 {
		t.Fatalf("matrix pins %d staged files; need at least 2 to prove distinctness", staged)
	}

	resolved, missing, err := m.ResolveAll(dir)
	if err != nil {
		t.Fatalf("ResolveAll: %v", err)
	}
	if len(resolved) != staged {
		t.Fatalf("resolved %d rows, want %d (missing: %v)", len(resolved), staged, missing)
	}

	seen := make(map[string]string, len(resolved))
	for key, path := range resolved {
		row, err := m.Row(key)
		if err != nil {
			t.Fatalf("Row(%s): %v", key, err)
		}
		if filepath.Base(path) != row.File {
			t.Errorf("row %q resolved to %q, want file %q", key, path, row.File)
		}
		if other, dup := seen[path]; dup {
			t.Errorf("rows %q and %q both resolved to %s", other, key, path)
		}
		seen[path] = key
	}
}

// TestParityConfigsPinKnownDistinctMatrixRows makes it impossible to add a
// parity suite that resolves its model by scanning, or to point two suites at
// one matrix row.
func TestParityConfigsPinKnownDistinctMatrixRows(t *testing.T) {
	m, err := modelset.Default()
	if err != nil {
		t.Fatalf("load matrix: %v", err)
	}

	byRow := make(map[string]string, len(allParityConfigs))
	for _, cfg := range allParityConfigs {
		if cfg.MatrixRow == "" {
			t.Errorf("parity suite %q declares no MatrixRow", cfg.Name)

			continue
		}
		row, err := m.Row(cfg.MatrixRow)
		if err != nil {
			t.Errorf("parity suite %q: %v", cfg.Name, err)

			continue
		}
		if other, dup := byRow[cfg.MatrixRow]; dup {
			t.Errorf("parity suites %q and %q both claim matrix row %q",
				other, cfg.Name, cfg.MatrixRow)
		}
		byRow[cfg.MatrixRow] = cfg.Name

		if cfg.ModelDirEnvVar != "" && cfg.ModelDirEnvVar != row.ModelDirEnv {
			t.Errorf("parity suite %q uses %s but matrix row %q declares %s",
				cfg.Name, cfg.ModelDirEnvVar, row.Key, row.ModelDirEnv)
		}
	}
}

// TestUnknownMatrixRowIsFatal proves the runner refuses a row it does not know
// instead of falling back to a directory scan.
func TestUnknownMatrixRowIsFatal(t *testing.T) {
	m, err := modelset.Default()
	if err != nil {
		t.Fatalf("load matrix: %v", err)
	}
	if _, err := m.Row("definitely-not-a-row"); !errors.Is(err, modelset.ErrUnknownRow) {
		t.Fatalf("Row(unknown) = %v, want ErrUnknownRow", err)
	}
}

// TestStagedModelIdentity is the per-row identity assertion against the real
// staged models. It runs wherever the GGUFs are mounted (the GB10 validation
// pod, or any host with ZERFOO_MODELS_DIR set) and skips per row otherwise.
// It reads GGUF headers only -- no tensor data, no GPU.
func TestStagedModelIdentity(t *testing.T) {
	m, err := modelset.Default()
	if err != nil {
		t.Fatalf("load matrix: %v", err)
	}

	dir := os.Getenv(modelset.EnvModelsDir)
	if dir == "" {
		dir = defaultStagedModelsDir
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("staged models directory %s not available: %v", dir, err)
	}

	rec := modelset.NewRecorder()
	verified := 0
	for _, row := range m.Rows {
		t.Run(row.Key, func(t *testing.T) {
			if !row.Staged() {
				t.Skipf("matrix row %q pins no GGUF file (model not staged)", row.Key)
			}
			path, err := m.Resolve(row.Key, dir)
			if err != nil {
				if errors.Is(err, modelset.ErrFileMissing) {
					t.Skipf("%v", err)
				}
				t.Fatalf("resolve %q: %v", row.Key, err)
			}
			id, err := modelset.Inspect(path)
			if err != nil {
				t.Fatalf("inspect %s: %v", path, err)
			}
			if err := row.VerifyIdentity(id); err != nil {
				t.Fatalf("%v", err)
			}
			if err := rec.Record(row.Key, path); err != nil {
				t.Fatalf("%v", err)
			}
			verified++
			t.Logf("row %q -> %s (general.architecture=%q general.name=%q size=%d)",
				row.Key, id.Path, id.Architecture, id.Name, id.SizeBytes)
		})
	}

	if verified == 0 {
		t.Skipf("no matrix models present under %s", dir)
	}
	t.Logf("verified identity of %d staged matrix models under %s", verified, dir)
}
