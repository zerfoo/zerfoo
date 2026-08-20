package modelset_test

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/modelset"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
)

// writeGGUF writes a minimal but structurally valid GGUF file declaring the
// given architecture and name, so identity checks can be exercised hermetically.
func writeGGUF(t *testing.T, path, arch, name string) {
	t.Helper()

	w := ztensorgguf.NewWriter()
	w.AddMetadataString("general.architecture", arch)
	w.AddMetadataString("general.name", name)
	w.AddTensorF32("token_embd.weight", []int{2, 2}, []float32{1, 2, 3, 4})

	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("write GGUF: %v", err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func testMatrix(t *testing.T) *modelset.Matrix {
	t.Helper()

	return &modelset.Matrix{Rows: []modelset.Row{
		{
			Key: "alpha", Label: "Alpha", ModelDirEnv: "ALPHA_MODEL_DIR",
			File: "alpha-Q4_K_M.gguf", Architecture: "llama", GGUFName: "Alpha",
		},
		{
			Key: "beta", Label: "Beta", ModelDirEnv: "BETA_MODEL_DIR",
			File: "beta-Q4_K_M.gguf", Architecture: "llama", GGUFName: "Beta",
		},
		{
			Key: "unstaged", Label: "Unstaged", ModelDirEnv: "UNSTAGED_MODEL_DIR",
			File: "", Architecture: "rwkv",
		},
	}}
}

// TestEmbeddedMatrixIsValid guards the checked-in table itself: unique row
// keys, unique filenames, and every row naming a directory env var.
func TestEmbeddedMatrixIsValid(t *testing.T) {
	m, err := modelset.Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	if len(m.Rows) == 0 {
		t.Fatal("embedded matrix has no rows")
	}
	staged := 0
	for _, r := range m.Rows {
		if r.Staged() {
			staged++
		}
	}
	if staged == 0 {
		t.Fatal("embedded matrix pins no GGUF filenames; the table would be inert")
	}
}

// TestUnknownRowIsErrorNotFallback is the core anti-vacuity property: asking
// for a row that does not exist must fail, never silently yield some file.
func TestUnknownRowIsErrorNotFallback(t *testing.T) {
	m := testMatrix(t)

	if _, err := m.Row("no-such-row"); !errors.Is(err, modelset.ErrUnknownRow) {
		t.Fatalf("Row(unknown) error = %v, want ErrUnknownRow", err)
	}

	dir := t.TempDir()
	writeGGUF(t, filepath.Join(dir, "alpha-Q4_K_M.gguf"), "llama", "Alpha")

	path, err := m.Resolve("no-such-row", dir)
	if !errors.Is(err, modelset.ErrUnknownRow) {
		t.Fatalf("Resolve(unknown) error = %v, want ErrUnknownRow", err)
	}
	if path != "" {
		t.Fatalf("Resolve(unknown) returned %q; an unknown row must never resolve to a file", path)
	}
}

// TestUnstagedRowDoesNotFallBackToDirectoryScan proves a row with no pinned
// file fails instead of picking up whatever GGUF happens to be in the dir.
func TestUnstagedRowDoesNotFallBackToDirectoryScan(t *testing.T) {
	m := testMatrix(t)
	dir := t.TempDir()
	writeGGUF(t, filepath.Join(dir, "alpha-Q4_K_M.gguf"), "llama", "Alpha")

	path, err := m.Resolve("unstaged", dir)
	if !errors.Is(err, modelset.ErrNoPinnedFile) {
		t.Fatalf("Resolve(unstaged) error = %v, want ErrNoPinnedFile", err)
	}
	if path != "" {
		t.Fatalf("Resolve(unstaged) returned %q; want no file", path)
	}
}

// TestMissingPinnedFileIsErrorNotSubstitution proves that when the declared
// file is absent the resolver errors rather than handing back a sibling.
func TestMissingPinnedFileIsErrorNotSubstitution(t *testing.T) {
	m := testMatrix(t)
	dir := t.TempDir()
	// Only alpha is staged; beta must not resolve to alpha.
	writeGGUF(t, filepath.Join(dir, "alpha-Q4_K_M.gguf"), "llama", "Alpha")

	path, err := m.Resolve("beta", dir)
	if !errors.Is(err, modelset.ErrFileMissing) {
		t.Fatalf("Resolve(beta) error = %v, want ErrFileMissing", err)
	}
	if path != "" {
		t.Fatalf("Resolve(beta) returned %q; want no file", path)
	}
	if !strings.Contains(err.Error(), "beta-Q4_K_M.gguf") {
		t.Errorf("error should name the file it wanted, got %v", err)
	}
}

// TestResolveReturnsExactDeclaredFile proves distinct rows against one flat
// directory resolve to distinct, exactly-named files.
func TestResolveReturnsExactDeclaredFile(t *testing.T) {
	m := testMatrix(t)
	dir := t.TempDir()
	writeGGUF(t, filepath.Join(dir, "alpha-Q4_K_M.gguf"), "llama", "Alpha")
	writeGGUF(t, filepath.Join(dir, "beta-Q4_K_M.gguf"), "llama", "Beta")

	got := map[string]string{}
	for _, key := range []string{"alpha", "beta"} {
		path, err := m.Resolve(key, dir)
		if err != nil {
			t.Fatalf("Resolve(%s): %v", key, err)
		}
		if !filepath.IsAbs(path) {
			t.Errorf("Resolve(%s) = %q, want an absolute path", key, path)
		}
		row, err := m.Row(key)
		if err != nil {
			t.Fatalf("Row(%s): %v", key, err)
		}
		if filepath.Base(path) != row.File {
			t.Errorf("Resolve(%s) = %q, want basename %q", key, path, row.File)
		}
		got[key] = path
	}
	if got["alpha"] == got["beta"] {
		t.Fatalf("rows alpha and beta both resolved to %q", got["alpha"])
	}
}

// TestDuplicateFilesInTableRejected proves the table cannot declare two rows
// backed by the same GGUF filename.
func TestDuplicateFilesInTableRejected(t *testing.T) {
	m := &modelset.Matrix{Rows: []modelset.Row{
		{Key: "a", ModelDirEnv: "A_DIR", File: "same.gguf", Architecture: "llama"},
		{Key: "b", ModelDirEnv: "B_DIR", File: "same.gguf", Architecture: "llama"},
	}}
	if err := m.Validate(); !errors.Is(err, modelset.ErrDuplicateResolution) {
		t.Fatalf("Validate error = %v, want ErrDuplicateResolution", err)
	}
}

// TestDuplicateRowKeysRejected guards against a copy-pasted row key silently
// shadowing another row.
func TestDuplicateRowKeysRejected(t *testing.T) {
	m := &modelset.Matrix{Rows: []modelset.Row{
		{Key: "a", ModelDirEnv: "A_DIR", File: "one.gguf", Architecture: "llama"},
		{Key: "a", ModelDirEnv: "A_DIR", File: "two.gguf", Architecture: "llama"},
	}}
	if err := m.Validate(); !errors.Is(err, modelset.ErrInvalidMatrix) {
		t.Fatalf("Validate error = %v, want ErrInvalidMatrix", err)
	}
}

// TestResolveAllRejectsAliasedRows proves two rows whose declared filenames are
// different aliases of one underlying GGUF are an error, not two data points.
func TestResolveAllRejectsAliasedRows(t *testing.T) {
	m := testMatrix(t)
	dir := t.TempDir()
	real := filepath.Join(dir, "alpha-Q4_K_M.gguf")
	writeGGUF(t, real, "llama", "Alpha")
	if err := os.Symlink(real, filepath.Join(dir, "beta-Q4_K_M.gguf")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	_, _, err := m.ResolveAll(dir)
	if !errors.Is(err, modelset.ErrDuplicateResolution) {
		t.Fatalf("ResolveAll error = %v, want ErrDuplicateResolution", err)
	}
}

// TestResolveAllReportsMissingWithoutSubstituting proves unstaged and absent
// rows are reported, never filled in from a neighbor.
func TestResolveAllReportsMissingWithoutSubstituting(t *testing.T) {
	m := testMatrix(t)
	dir := t.TempDir()
	writeGGUF(t, filepath.Join(dir, "alpha-Q4_K_M.gguf"), "llama", "Alpha")

	resolved, missing, err := m.ResolveAll(dir)
	if err != nil {
		t.Fatalf("ResolveAll: %v", err)
	}
	if len(resolved) != 1 || resolved["alpha"] == "" {
		t.Fatalf("resolved = %v, want only alpha", resolved)
	}
	if !errors.Is(missing["beta"], modelset.ErrFileMissing) {
		t.Errorf("missing[beta] = %v, want ErrFileMissing", missing["beta"])
	}
	if !errors.Is(missing["unstaged"], modelset.ErrNoPinnedFile) {
		t.Errorf("missing[unstaged] = %v, want ErrNoPinnedFile", missing["unstaged"])
	}
}

// TestRecorderRejectsTwoRowsOnOneFile is the runtime backstop: even if the
// table were wrong, two rows loading one file must fail the run.
func TestRecorderRejectsTwoRowsOnOneFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "alpha-Q4_K_M.gguf")
	writeGGUF(t, path, "llama", "Alpha")

	rec := modelset.NewRecorder()
	if err := rec.Record("alpha", path); err != nil {
		t.Fatalf("first Record: %v", err)
	}
	if err := rec.Record("alpha", path); err != nil {
		t.Fatalf("re-recording the same row/path must be allowed: %v", err)
	}
	if err := rec.Record("beta", path); !errors.Is(err, modelset.ErrDuplicateResolution) {
		t.Fatalf("Record(beta, same path) = %v, want ErrDuplicateResolution", err)
	}
}

// TestRecorderFollowsSymlinks proves aliasing cannot smuggle one file in as two.
func TestRecorderFollowsSymlinks(t *testing.T) {
	dir := t.TempDir()
	real := filepath.Join(dir, "alpha-Q4_K_M.gguf")
	writeGGUF(t, real, "llama", "Alpha")
	alias := filepath.Join(dir, "beta-Q4_K_M.gguf")
	if err := os.Symlink(real, alias); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	rec := modelset.NewRecorder()
	if err := rec.Record("alpha", real); err != nil {
		t.Fatalf("Record(alpha): %v", err)
	}
	if err := rec.Record("beta", alias); !errors.Is(err, modelset.ErrDuplicateResolution) {
		t.Fatalf("Record(beta, alias) = %v, want ErrDuplicateResolution", err)
	}
}

// TestVerifyIdentityCatchesWrongFile proves the per-row identity assertion
// rejects a file that is not the one the row declared, even when the
// architecture matches -- which is exactly the flat-directory failure mode.
func TestVerifyIdentityCatchesWrongFile(t *testing.T) {
	m := testMatrix(t)
	dir := t.TempDir()
	alpha := filepath.Join(dir, "alpha-Q4_K_M.gguf")
	writeGGUF(t, alpha, "llama", "Alpha")

	beta, err := m.Row("beta")
	if err != nil {
		t.Fatalf("Row(beta): %v", err)
	}
	id, err := modelset.Inspect(alpha)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	// Same architecture ("llama") as beta declares, different file: the path
	// check must still reject it.
	if id.Architecture != beta.Architecture {
		t.Fatalf("fixture setup: architectures differ (%q vs %q)", id.Architecture, beta.Architecture)
	}
	if err := beta.VerifyIdentity(id); !errors.Is(err, modelset.ErrIdentityMismatch) {
		t.Fatalf("VerifyIdentity = %v, want ErrIdentityMismatch", err)
	}
}

// TestVerifyIdentityCatchesWrongArchitecture proves the header assertion fires
// when a correctly-named file contains the wrong model.
func TestVerifyIdentityCatchesWrongArchitecture(t *testing.T) {
	m := testMatrix(t)
	dir := t.TempDir()
	alpha := filepath.Join(dir, "alpha-Q4_K_M.gguf")
	writeGGUF(t, alpha, "qwen2", "Alpha")

	row, err := m.Row("alpha")
	if err != nil {
		t.Fatalf("Row(alpha): %v", err)
	}
	id, err := modelset.Inspect(alpha)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if err := row.VerifyIdentity(id); !errors.Is(err, modelset.ErrIdentityMismatch) {
		t.Fatalf("VerifyIdentity = %v, want ErrIdentityMismatch", err)
	}
}

// TestVerifyIdentityAcceptsDeclaredFile is the two-sided half of the check.
func TestVerifyIdentityAcceptsDeclaredFile(t *testing.T) {
	m := testMatrix(t)
	dir := t.TempDir()
	alpha := filepath.Join(dir, "alpha-Q4_K_M.gguf")
	writeGGUF(t, alpha, "llama", "Alpha")

	path, err := m.Resolve("alpha", dir)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	row, err := m.Row("alpha")
	if err != nil {
		t.Fatalf("Row: %v", err)
	}
	id, err := modelset.Inspect(path)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if err := row.VerifyIdentity(id); err != nil {
		t.Fatalf("VerifyIdentity on the declared file: %v", err)
	}
}

// TestBaseDirPrefersRowEnvThenSuiteEnv documents the directory lookup order.
func TestBaseDirPrefersRowEnvThenSuiteEnv(t *testing.T) {
	m := testMatrix(t)

	if _, err := m.BaseDir("alpha"); !errors.Is(err, modelset.ErrNoBaseDir) {
		t.Fatalf("BaseDir with no env = %v, want ErrNoBaseDir", err)
	}

	t.Setenv(modelset.EnvModelsDir, "/suite/dir")
	if got, err := m.BaseDir("alpha"); err != nil || got != "/suite/dir" {
		t.Fatalf("BaseDir = (%q, %v), want /suite/dir", got, err)
	}

	t.Setenv("ALPHA_MODEL_DIR", "/row/dir")
	if got, err := m.BaseDir("alpha"); err != nil || got != "/row/dir" {
		t.Fatalf("BaseDir = (%q, %v), want /row/dir", got, err)
	}
}
