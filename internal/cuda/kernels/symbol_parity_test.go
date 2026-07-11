package kernels

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// TestForkParitySymbols guards against the T135.3 class of drift (T135.6,
// docs/plan-gpu-training-hardening.md).
//
// internal/cuda/kernels is a hand-maintained FORK of a subset of
// github.com/zerfoo/ztensor's kernel sources. Both this package's own
// purego.go AND ztensor's own internal/cuda/kernels/purego.go dlopen the
// SAME libkernels.so deployed on the DGX host (/opt/zerfoo/lib); every
// ztensor compute.GPUEngine op resolves its kernel symbols through
// ztensor's loader, not this repo's. On 2026-07-03, rebuilding
// libkernels.so from this repo's Makefile SRCS list for the first time
// silently dropped five symbols ztensor's loader dlsym's unconditionally
// (launch_transpose_2d_bf16, launch_transpose_nd_bf16, dropout_f32,
// fused_adamw_f32, tiny_batched_gemm_f32): one missing REQUIRED symbol
// fails ztensor's dlsym loop wholesale, so every GPUEngine op reported
// "kernels not available" -- while this repo's own purego loader (which
// never referenced those symbols) stayed green, masking the break. See
// docs/devlog.md 2026-07-03 "T135.3 .so rebuild dropped flash_decode_splitkv_f32"
// and the entry above it for the full incident.
//
// This test is the STATIC, CI-runnable half of the fix: it extracts (a)
// every symbol name this package's own purego.go dlsym-looks-up, (b) every
// symbol name ztensor's purego.go (resolved via `go list -m` against the
// version actually pinned in go.mod/go.sum) dlsym-looks-up, and (c) every
// symbol name defined across the .cu files listed in this package's
// Makefile SRCS -- then asserts every symbol from (a) and (b) that its own
// loader treats as REQUIRED (i.e. not listed in that loader's optionalSyms
// map) is present in (c). It cannot see true dynamic symbol EXPORT (that
// needs `nm -D` against a built .so, which needs nvcc/the DGX) -- but it
// does catch "the .cu source that defines this symbol was never added to
// SRCS", which is exactly the T135.3 failure mode, entirely offline.
//
// Dynamic (DGX) mode: set ZERFOO_KERNELS_SO=/path/to/libkernels.so to
// switch the "provided" set from the static SRCS-text scan to a real
// `nm -D` dump of the built shared object -- the check the standing gate
// should run after every libkernels.so rebuild, e.g.:
//
//	scripts/dgx-validate.sh \
//	  -env "ZERFOO_KERNELS_SO=/opt/zerfoo/lib/libkernels.so" \
//	  -pkgs "-run TestForkParitySymbols ./internal/cuda/kernels/"
//
// which requires `nm` on the pod image (present in the nvcc/build-pod
// image used for the .so rebuild itself) and fails loudly (not skips) on
// any nm error, since the caller opted in explicitly via the env var.
func TestForkParitySymbols(t *testing.T) {
	ownSrc, err := os.ReadFile("purego.go")
	if err != nil {
		t.Fatalf("read purego.go: %v", err)
	}
	ownRequired, err := parsePuregoRequiredSymbols(ownSrc)
	if err != nil {
		t.Fatalf("parse this package's purego.go: %v", err)
	}
	if len(ownRequired) < 10 {
		t.Fatalf("parsed only %d required symbols from purego.go -- parser likely out of sync with the source shape (expected several dozen); update parsePuregoRequiredSymbols", len(ownRequired))
	}

	required := map[string]bool{}
	for _, s := range ownRequired {
		required[s] = true
	}

	if ztensorRequired, ztensorPath, skip := ztensorRequiredSymbols(t); skip == "" {
		if len(ztensorRequired) < 10 {
			t.Fatalf("parsed only %d required symbols from %s -- parser likely out of sync with the source shape; update parsePuregoRequiredSymbols", len(ztensorRequired), ztensorPath)
		}
		for _, s := range ztensorRequired {
			required[s] = true
		}
	} else {
		t.Logf("skipping the ztensor half of the fork-parity check: %s", skip)
	}

	names := make([]string, 0, len(required))
	for s := range required {
		names = append(names, s)
	}
	sort.Strings(names)

	if soPath := os.Getenv("ZERFOO_KERNELS_SO"); soPath != "" {
		provided, err := exportedSymbolsFromSharedObject(soPath)
		if err != nil {
			t.Fatalf("nm -D %s: %v (ZERFOO_KERNELS_SO was set explicitly -- this dynamic check requires `nm` and a built .so, run it on the DGX build pod)", soPath, err)
		}
		assertAllProvided(t, names, provided, fmt.Sprintf("`nm -D %s`", soPath))
		return
	}

	srcs, err := makefileSRCS("Makefile")
	if err != nil {
		t.Fatalf("parse Makefile SRCS: %v", err)
	}
	if len(srcs) == 0 {
		t.Fatalf("Makefile SRCS parsed empty -- update makefileSRCS")
	}
	var content strings.Builder
	for _, f := range srcs {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("Makefile SRCS lists %q but it could not be read: %v (every source listed in SRCS must exist and define its symbols)", f, err)
		}
		content.Write(b)
		content.WriteByte('\n')
	}
	provided := sourceDefinedSymbols(content.String(), names)
	assertAllProvided(t, names, provided, "internal/cuda/kernels/Makefile SRCS (static source scan)")
}

func assertAllProvided(t *testing.T, required []string, provided map[string]bool, providedDesc string) {
	t.Helper()
	var missing []string
	for _, s := range required {
		if !provided[s] {
			missing = append(missing, s)
		}
	}
	if len(missing) > 0 {
		t.Fatalf(
			"fork-parity check FAILED: %d required kernel symbol(s) not found in %s: %s\n\n"+
				"These symbols are dlsym'd unconditionally by openKernelLib() in this repo's "+
				"internal/cuda/kernels/purego.go and/or github.com/zerfoo/ztensor's own "+
				"internal/cuda/kernels/purego.go. If a deployed libkernels.so is missing any of "+
				"them, ztensor's dlsym loop fails wholesale and EVERY compute.GPUEngine op reports "+
				"\"kernels not available\" (the T135.3 class of drift, docs/devlog.md 2026-07-03). "+
				"Add the .cu source that defines each missing symbol to this package's Makefile SRCS.",
			len(missing), providedDesc, strings.Join(missing, ", "),
		)
	}
}

// symsEntryRe matches one `{"symbol_name", &k.field}` entry in a purego.go
// `syms := []struct{ name string; dest *uintptr }{ ... }` literal.
var symsEntryRe = regexp.MustCompile(`\{"([A-Za-z0-9_]+)"\s*,\s*&k\.\w+\}`)

// optionalEntryRe matches one `"symbol_name": true` entry in a purego.go
// `optionalSyms := map[string]bool{ ... }` literal.
var optionalEntryRe = regexp.MustCompile(`"([A-Za-z0-9_]+)"\s*:\s*true`)

// parsePuregoRequiredSymbols parses the `syms := []struct{...}{...}` /
// `optionalSyms := map[string]bool{...}` shape shared by zerfoo's and
// ztensor's internal/cuda/kernels/purego.go (openKernelLib) and returns the
// symbol names dlsym MUST resolve for openKernelLib to succeed, i.e. every
// entry in syms that does not also appear in optionalSyms.
func parsePuregoRequiredSymbols(src []byte) ([]string, error) {
	text := string(src)
	symsStart := strings.Index(text, "syms := []struct")
	optStart := strings.Index(text, "optionalSyms := map[string]bool{")
	loopStart := strings.Index(text, "for _, s := range syms {")
	if symsStart < 0 || optStart < 0 || loopStart < 0 || !(symsStart < optStart && optStart < loopStart) {
		return nil, fmt.Errorf("purego.go structure not recognized (syms/optionalSyms/loop markers missing or out of order) -- update the parser in symbol_parity_test.go")
	}

	symsBlock := text[symsStart:optStart]
	optBlock := text[optStart:loopStart]

	all := map[string]bool{}
	for _, m := range symsEntryRe.FindAllStringSubmatch(symsBlock, -1) {
		all[m[1]] = true
	}
	if len(all) == 0 {
		return nil, fmt.Errorf("no symbol entries parsed from the syms block -- parser out of sync with source")
	}

	optional := map[string]bool{}
	for _, m := range optionalEntryRe.FindAllStringSubmatch(optBlock, -1) {
		optional[m[1]] = true
	}

	var required []string
	for s := range all {
		if !optional[s] {
			required = append(required, s)
		}
	}
	sort.Strings(required)
	return required, nil
}

// ztensorRequiredSymbols locates github.com/zerfoo/ztensor's own
// internal/cuda/kernels/purego.go via `go list -m` (the exact version this
// module currently depends on) and parses its required symbol set. If the
// module cannot be located (offline module cache, vendored build without
// the package, etc.) it returns a non-empty skip reason instead of failing
// -- this half of the check is best-effort; the in-repo half always runs.
func ztensorRequiredSymbols(t *testing.T) (syms []string, path string, skipReason string) {
	t.Helper()
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/zerfoo/ztensor").Output()
	if err != nil {
		return nil, "", fmt.Sprintf("`go list -m` for github.com/zerfoo/ztensor failed: %v", err)
	}
	dir := strings.TrimSpace(string(out))
	if dir == "" {
		return nil, "", "`go list -m` returned an empty module dir for github.com/zerfoo/ztensor"
	}
	path = filepath.Join(dir, "internal", "cuda", "kernels", "purego.go")
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, path, fmt.Sprintf("could not read %s: %v", path, err)
	}
	required, err := parsePuregoRequiredSymbols(src)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return required, path, ""
}

// makefileSRCS extracts the whitespace-separated filenames in a Makefile's
// `SRCS = a.cu b.cu ...` assignment line.
func makefileSRCS(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	srcsRe := regexp.MustCompile(`(?m)^SRCS\s*=\s*(.+)$`)
	m := srcsRe.FindSubmatch(b)
	if m == nil {
		return nil, fmt.Errorf("no top-level SRCS assignment found in %s", path)
	}
	return strings.Fields(string(m[1])), nil
}

// sourceDefinedSymbols reports, for each candidate symbol name, whether it
// appears in content at a word boundary immediately followed by optional
// whitespace and '(' -- i.e. the shape of a C function definition or
// declaration. This is a deliberately crude static scan (it cannot
// distinguish a definition from a comment or a call site), but it is
// exactly enough to catch "the .cu file defining this symbol was never
// added to SRCS", which is the T135.3 failure mode this test targets.
func sourceDefinedSymbols(content string, candidates []string) map[string]bool {
	found := map[string]bool{}
	for _, name := range candidates {
		re := regexp.MustCompile(`\b` + regexp.QuoteMeta(name) + `\s*\(`)
		if re.MatchString(content) {
			found[name] = true
		}
	}
	return found
}

// exportedSymbolsFromSharedObject runs `nm -D` against a built shared
// object and returns the set of symbol names it exports (dynamic symbol
// table). This is the true dynamic check -- it requires `nm` and an
// nvcc-built libkernels.so, so it only runs when ZERFOO_KERNELS_SO is set
// (the DGX build-pod / standing-gate invocation), never in ordinary CI.
func exportedSymbolsFromSharedObject(soPath string) (map[string]bool, error) {
	out, err := exec.Command("nm", "-D", soPath).Output()
	if err != nil {
		return nil, err
	}
	provided := map[string]bool{}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		// `nm -D` lines look like "0000000000001234 T symbol_name" for
		// defined symbols and "                 U symbol_name" for
		// undefined ones; the symbol name is always the last field.
		provided[fields[len(fields)-1]] = true
	}
	return provided, nil
}
