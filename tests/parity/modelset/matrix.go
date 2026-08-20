// Package modelset resolves verified-model matrix rows to exact GGUF files.
//
// The flagship models are staged FLAT in /var/lib/zerfoo/models (no per-model
// subdirectories). Any resolution strategy that scans a directory for "the
// first *.gguf" -- the findGGUF pattern at inference/inference.go:317 -- maps
// every matrix row onto the same alphabetically-first file, so a parity matrix
// built on it goes green while proving nothing. This package replaces that
// scan with an explicit, checked-in row -> filename table plus identity
// assertions, and refuses to hand back a file it was not asked for.
//
// Rules enforced here:
//   - An unknown row is an error, never a fallback to some other file.
//   - A row with no pinned file is an error, never a directory scan.
//   - A pinned file that is not present is an error, never a substitute.
//   - Two different rows resolving to the same absolute path is an error.
package modelset

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/zerfoo/zerfoo/model/gguf"
)

// Sentinel errors returned by the resolver. Callers distinguish "this model is
// not staged on this host" (skip) from "the harness is lying to you" (fail).
var (
	// ErrUnknownRow is returned when a matrix row key is not in the table.
	ErrUnknownRow = errors.New("modelset: unknown matrix row")
	// ErrNoPinnedFile is returned when a row declares no GGUF filename.
	ErrNoPinnedFile = errors.New("modelset: matrix row has no pinned GGUF file")
	// ErrFileMissing is returned when the pinned file is absent from the base directory.
	ErrFileMissing = errors.New("modelset: pinned GGUF file not present")
	// ErrDuplicateResolution is returned when two rows resolve to the same file.
	ErrDuplicateResolution = errors.New("modelset: two matrix rows resolved to the same file")
	// ErrIdentityMismatch is returned when a resolved file's GGUF header does
	// not match what the row declares.
	ErrIdentityMismatch = errors.New("modelset: resolved GGUF does not match the matrix row")
	// ErrNoBaseDir is returned when no directory was supplied for a row.
	ErrNoBaseDir = errors.New("modelset: no base directory for matrix row")
	// ErrInvalidMatrix is returned when the checked-in table is malformed.
	ErrInvalidMatrix = errors.New("modelset: invalid matrix table")
)

// Row is one verified-model matrix row bound to an exact GGUF filename.
type Row struct {
	// Key is the stable row identifier used by ModelParityConfig.MatrixRow.
	Key string `json:"row"`
	// Label is the human-readable name used in docs/verified-models.md.
	Label string `json:"label"`
	// ModelDirEnv names the environment variable holding the DIRECTORY the
	// pinned file lives in. The filename is never taken from the environment.
	ModelDirEnv string `json:"model_dir_env"`
	// File is the exact GGUF filename joined onto the directory. Empty means
	// the row has no staged file; resolution fails rather than scanning.
	File string `json:"file"`
	// Architecture is the expected GGUF general.architecture value.
	Architecture string `json:"architecture"`
	// GGUFName is the expected general.name value, when the file declares one.
	GGUFName string `json:"gguf_name,omitempty"`
	// SizeBytes is the expected file size; 0 disables the size check.
	SizeBytes int64 `json:"size_bytes,omitempty"`
}

// Staged reports whether the row pins a concrete GGUF filename.
func (r Row) Staged() bool { return r.File != "" }

// Matrix is the checked-in set of verified-model matrix rows.
type Matrix struct {
	Rows []Row `json:"rows"`
}

//go:embed model-matrix.json
var matrixJSON []byte

var (
	defaultOnce    sync.Once
	defaultMatrix  *Matrix
	errDefaultLoad error
)

// Load parses and validates the checked-in matrix table.
func Load() (*Matrix, error) {
	return Parse(matrixJSON)
}

// Parse parses and validates a matrix table from JSON.
func Parse(data []byte) (*Matrix, error) {
	var m Matrix
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidMatrix, err)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}

	return &m, nil
}

// Default returns the process-wide matrix parsed from the embedded table.
func Default() (*Matrix, error) {
	defaultOnce.Do(func() {
		defaultMatrix, errDefaultLoad = Load()
	})

	return defaultMatrix, errDefaultLoad
}

// Validate checks the table's internal invariants: rows are keyed uniquely,
// every row names a directory env var and an architecture, and no two rows
// pin the same filename (which would make two matrix rows the same model).
func (m *Matrix) Validate() error {
	if len(m.Rows) == 0 {
		return fmt.Errorf("%w: no rows", ErrInvalidMatrix)
	}
	seenKey := make(map[string]bool, len(m.Rows))
	seenFile := make(map[string]string, len(m.Rows))
	for i, r := range m.Rows {
		switch {
		case r.Key == "":
			return fmt.Errorf("%w: row %d has no key", ErrInvalidMatrix, i)
		case r.ModelDirEnv == "":
			return fmt.Errorf("%w: row %q has no model_dir_env", ErrInvalidMatrix, r.Key)
		case r.Architecture == "":
			return fmt.Errorf("%w: row %q has no architecture", ErrInvalidMatrix, r.Key)
		case seenKey[r.Key]:
			return fmt.Errorf("%w: duplicate row key %q", ErrInvalidMatrix, r.Key)
		}
		seenKey[r.Key] = true

		if r.File == "" {
			continue
		}
		if r.File != filepath.Base(r.File) {
			return fmt.Errorf("%w: row %q file %q must be a bare filename", ErrInvalidMatrix, r.Key, r.File)
		}
		if other, dup := seenFile[r.File]; dup {
			return fmt.Errorf("%w: rows %q and %q both pin %q",
				ErrDuplicateResolution, other, r.Key, r.File)
		}
		seenFile[r.File] = r.Key
	}

	return nil
}

// Keys returns the row keys in table order.
func (m *Matrix) Keys() []string {
	keys := make([]string, 0, len(m.Rows))
	for _, r := range m.Rows {
		keys = append(keys, r.Key)
	}

	return keys
}

// Row returns the row with the given key. An unknown key is an error; the
// resolver never falls back to a "close enough" row.
func (m *Matrix) Row(key string) (Row, error) {
	for _, r := range m.Rows {
		if r.Key == key {
			return r, nil
		}
	}
	known := m.Keys()
	sort.Strings(known)

	return Row{}, fmt.Errorf("%w: %q (known rows: %v)", ErrUnknownRow, key, known)
}

// BaseDir returns the directory configured for a row, consulting the row's own
// environment variable first and then the suite-wide ZERFOO_MODELS_DIR.
// Callers that already know the directory should use Resolve directly.
func (m *Matrix) BaseDir(key string) (string, error) {
	r, err := m.Row(key)
	if err != nil {
		return "", err
	}
	if dir := os.Getenv(r.ModelDirEnv); dir != "" {
		return dir, nil
	}
	if dir := os.Getenv(EnvModelsDir); dir != "" {
		return dir, nil
	}

	return "", fmt.Errorf("%w: set %s or %s", ErrNoBaseDir, r.ModelDirEnv, EnvModelsDir)
}

// EnvModelsDir is the suite-wide fallback directory environment variable.
const EnvModelsDir = "ZERFOO_MODELS_DIR"

// Resolve returns the absolute path of the row's pinned GGUF inside baseDir.
// It never scans the directory: the filename comes from the table, and a
// missing file is an error rather than a substitution.
func (m *Matrix) Resolve(key, baseDir string) (string, error) {
	r, err := m.Row(key)
	if err != nil {
		return "", err
	}
	if !r.Staged() {
		return "", fmt.Errorf("%w: row %q (%s)", ErrNoPinnedFile, r.Key, r.Label)
	}
	if baseDir == "" {
		return "", fmt.Errorf("%w: row %q; set %s or %s", ErrNoBaseDir, r.Key, r.ModelDirEnv, EnvModelsDir)
	}

	abs, err := filepath.Abs(filepath.Join(baseDir, r.File))
	if err != nil {
		return "", fmt.Errorf("resolve %q: %w", r.Key, err)
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("%w: row %q wants %s: %w", ErrFileMissing, r.Key, abs, err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("%w: row %q resolved to a directory: %s", ErrFileMissing, r.Key, abs)
	}

	return abs, nil
}

// ResolveAll resolves every staged row against a single flat base directory
// and fails if two rows land on the same file. Rows whose pinned file is not
// present are reported in the returned missing map, never substituted.
//
// Collisions are detected with os.SameFile, so two differently-named symlinks
// or hard links pointing at one GGUF are caught, not just identical names.
func (m *Matrix) ResolveAll(baseDir string) (resolved map[string]string, missing map[string]error, err error) {
	resolved = make(map[string]string, len(m.Rows))
	missing = make(map[string]error, len(m.Rows))

	type claim struct {
		key  string
		path string
		info os.FileInfo
	}
	claims := make([]claim, 0, len(m.Rows))

	for _, r := range m.Rows {
		path, rerr := m.Resolve(r.Key, baseDir)
		if rerr != nil {
			missing[r.Key] = rerr

			continue
		}
		info, serr := os.Stat(path)
		if serr != nil {
			missing[r.Key] = fmt.Errorf("%w: row %q: %w", ErrFileMissing, r.Key, serr)

			continue
		}
		for _, c := range claims {
			if os.SameFile(c.info, info) {
				return nil, nil, fmt.Errorf("%w: rows %q (%s) and %q (%s) are the same file",
					ErrDuplicateResolution, c.key, c.path, r.Key, path)
			}
		}
		claims = append(claims, claim{key: r.Key, path: path, info: info})
		resolved[r.Key] = path
	}

	return resolved, missing, nil
}

// Identity is what a GGUF file says about itself, read from its header.
type Identity struct {
	Path         string
	Architecture string
	Name         string
	SizeBytes    int64
}

// Inspect reads a GGUF file's header and reports its self-declared identity.
// It parses metadata only; tensor data is never read.
func Inspect(path string) (Identity, error) {
	// #nosec G304 -- path comes from the checked-in matrix table joined onto an
	// operator-supplied models directory; reading it is the point of this call.
	f, err := os.Open(path)
	if err != nil {
		return Identity{}, fmt.Errorf("open %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	info, err := f.Stat()
	if err != nil {
		return Identity{}, fmt.Errorf("stat %s: %w", path, err)
	}

	gf, err := gguf.Parse(f)
	if err != nil {
		return Identity{}, fmt.Errorf("parse GGUF header %s: %w", path, err)
	}
	arch, _ := gf.GetString("general.architecture")
	name, _ := gf.GetString("general.name")

	return Identity{
		Path:         path,
		Architecture: arch,
		Name:         name,
		SizeBytes:    info.Size(),
	}, nil
}

// VerifyIdentity checks a resolved file against what the row declares: the
// filename must be the pinned one, and the GGUF header's architecture (plus
// name and size when the row declares them) must agree.
//
// Architecture is corroboration, not identity: "llama" covers both Llama 3.2
// and Mistral, and "qwen2" covers both Qwen2-7B and the DeepSeek-R1 distill,
// so the path check is the load-bearing assertion.
func (r Row) VerifyIdentity(id Identity) error {
	if got := filepath.Base(id.Path); got != r.File {
		return fmt.Errorf("%w: row %q resolved to %q, want file %q",
			ErrIdentityMismatch, r.Key, got, r.File)
	}
	if !filepath.IsAbs(id.Path) {
		return fmt.Errorf("%w: row %q resolved to a relative path %q",
			ErrIdentityMismatch, r.Key, id.Path)
	}
	if id.Architecture != r.Architecture {
		return fmt.Errorf("%w: row %q (%s) has general.architecture=%q, want %q",
			ErrIdentityMismatch, r.Key, id.Path, id.Architecture, r.Architecture)
	}
	if r.GGUFName != "" && id.Name != r.GGUFName {
		return fmt.Errorf("%w: row %q (%s) has general.name=%q, want %q",
			ErrIdentityMismatch, r.Key, id.Path, id.Name, r.GGUFName)
	}
	if r.SizeBytes != 0 && id.SizeBytes != r.SizeBytes {
		return fmt.Errorf("%w: row %q (%s) is %d bytes, want %d",
			ErrIdentityMismatch, r.Key, id.Path, id.SizeBytes, r.SizeBytes)
	}

	return nil
}

// Recorder tracks which matrix row claimed which file within a run, so that
// two rows silently loading the same GGUF is a hard error even if the table
// itself were wrong.
type Recorder struct {
	mu    sync.Mutex
	owner map[string]string
}

// NewRecorder returns an empty Recorder.
func NewRecorder() *Recorder {
	return &Recorder{owner: make(map[string]string)}
}

// Record claims path for rowKey. Re-recording the same row/path pair is fine;
// a different row claiming an already-claimed path is ErrDuplicateResolution.
// Symlinks are followed so that two aliases of one GGUF still collide.
func (rec *Recorder) Record(rowKey, path string) error {
	key := path
	if real, err := filepath.EvalSymlinks(path); err == nil {
		key = real
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()

	if rec.owner == nil {
		rec.owner = make(map[string]string)
	}
	if other, ok := rec.owner[key]; ok && other != rowKey {
		return fmt.Errorf("%w: rows %q and %q both loaded %s",
			ErrDuplicateResolution, other, rowKey, key)
	}
	rec.owner[key] = rowKey

	return nil
}

var sharedRecorder = NewRecorder()

// RecordResolution claims a path for a row in the process-wide recorder.
func RecordResolution(rowKey, path string) error {
	return sharedRecorder.Record(rowKey, path)
}
