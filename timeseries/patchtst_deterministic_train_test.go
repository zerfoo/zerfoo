package timeseries

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// GPU training determinism proof for ZTENSOR_DETERMINISTIC=1 (zerfoo
// plan-gpu-training-hardening.md T4.1 / plan.md T135.5).
//
// The claim under test: two identically-seeded training runs, in two separate
// processes with ZTENSOR_DETERMINISTIC=1, produce bitwise-identical per-epoch
// losses. Bitwise means exact float64 bit patterns, not printed decimals.
//
// Process structure: ztensor latches ZTENSOR_DETERMINISTIC once at process
// init (a package-level var, like ZTENSOR_ARENA_POISON), so setting the env
// inside a test is too late. The parent test therefore re-execs its own test
// binary twice as child processes with the env set in each child's
// environment from birth, and compares the loss bits the children print.
// Separate processes are deliberately the strongest form of the claim: each
// child creates its own CUDA context, cuBLAS handle, and arena, so nothing
// is accidentally shared between the two runs.
//
// On CPU-only hosts the child skips (NewGPUEngine fails) and the parent
// skips with it. Run on the GB10 via scripts/dgx-validate.sh:
//
//	scripts/dgx-validate.sh -ref <branch> -pkgs "-v -run TestPatchTSTTrainGPUDeterministicDoubleRun ./timeseries/"

const deterministicChildEnv = "ZERFOO_TEST_DETERMINISTIC_CHILD"

// TestPatchTSTTrainGPUDeterministicChild is the child-process body. It is
// guarded by an env marker so it does nothing under a normal `go test ./...`
// sweep; only the double-run parent below spawns it.
func TestPatchTSTTrainGPUDeterministicChild(t *testing.T) {
	if os.Getenv(deterministicChildEnv) != "1" {
		t.Skip("child-process body; spawned by TestPatchTSTTrainGPUDeterministicDoubleRun")
	}

	ops := numeric.Float32Ops{}
	gpu, err := compute.NewGPUEngine[float32](ops)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer gpu.Close()

	losses, err := runSeededPatchTSTTraining(gpu)
	if err != nil {
		t.Fatalf("seeded training run: %v", err)
	}
	for i, l := range losses {
		if math.IsNaN(l) || math.IsInf(l, 0) {
			t.Fatalf("epoch %d loss not finite: %v", i+1, l)
		}
		// The parent greps for this exact prefix. Bits, not decimals.
		fmt.Printf("DETERMINISM_EPOCH %d 0x%016x\n", i+1, math.Float64bits(l))
	}
}

// runSeededPatchTSTTraining performs one fixed-seed PatchTST training run on
// the given engine and returns the per-epoch loss history. Everything that
// feeds the run is a pure function of the constant seed: weight init (via
// SeedWeightInit), synthetic data, and the (shuffle-free) batch order.
func runSeededPatchTSTTraining(engine compute.Engine[float32]) ([]float64, error) {
	const (
		seed      = 42
		nSamples  = 256
		nChannels = 4
		inputLen  = 24
		epochs    = 3
	)

	SeedWeightInit(seed, seed<<32|seed)

	config := PatchTSTConfig{
		InputLength: inputLen,
		PatchLength: 8,
		Stride:      4,
		DModel:      64,
		NHeads:      4,
		NLayers:     2,
		OutputDim:   1,
	}

	ops := numeric.Float32Ops{}
	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("NewPatchTST: %w", err)
	}

	// Synthetic random-walk data, same construction as cmd/bench_train.
	//nolint:gosec // deterministic test data, not cryptographic material
	rng := rand.New(rand.NewSource(seed))
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.OutputDim)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, nChannels)
		sum := 0.0
		for c := 0; c < nChannels; c++ {
			windows[s][c] = make([]float64, inputLen)
			val := rng.Float64()
			for i := 0; i < inputLen; i++ {
				val += rng.NormFloat64() * 0.1
				windows[s][c][i] = val
				sum += val
			}
		}
		labels[s] = sum / float64(nChannels*inputLen)
	}

	result, err := model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:    epochs,
		LR:        1e-3,
		GradClip:  1.0,
		BatchSize: 64,
	})
	if err != nil {
		return nil, fmt.Errorf("TrainWindowed: %w", err)
	}
	return result.LossHistory, nil
}

// spawnDeterministicChild runs the child test in a fresh process with
// deterministic=true|false controlling ZTENSOR_DETERMINISTIC in the child's
// environment, and returns the ordered per-epoch loss bit lines it printed.
func spawnDeterministicChild(t *testing.T, deterministic bool) []string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	//nolint:gosec // exe is this test binary itself; args are constant
	cmd := exec.Command(exe, "-test.run", "^TestPatchTSTTrainGPUDeterministicChild$", "-test.v")
	env := append(os.Environ(), deterministicChildEnv+"=1")
	if deterministic {
		env = append(env, "ZTENSOR_DETERMINISTIC=1")
	} else {
		env = append(env, "ZTENSOR_DETERMINISTIC=")
	}
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child process (deterministic=%v) failed: %v\noutput:\n%s", deterministic, err, out)
	}
	if strings.Contains(string(out), "SKIP") && !strings.Contains(string(out), "DETERMINISM_EPOCH") {
		t.Skipf("child skipped (no GPU): %s", out)
	}
	var lines []string
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "DETERMINISM_EPOCH ") {
			lines = append(lines, strings.TrimSpace(line))
		}
	}
	if len(lines) == 0 {
		t.Fatalf("child (deterministic=%v) printed no DETERMINISM_EPOCH lines:\n%s", deterministic, out)
	}
	return lines
}

// TestPatchTSTTrainGPUDeterministicDoubleRun is the T4.1 acceptance proof:
// two identically-seeded GPU training runs in separate processes under
// ZTENSOR_DETERMINISTIC=1 must produce bitwise-identical per-epoch losses.
//
// A control pair WITHOUT the flag is also run and its bits logged. The
// control's runs may or may not differ (cuBLAS's default heuristics are
// frequently stable for a fixed shape on a fixed GPU even without the mode),
// so the control asserts nothing -- it only records the comparison honestly.
func TestPatchTSTTrainGPUDeterministicDoubleRun(t *testing.T) {
	if os.Getenv(deterministicChildEnv) == "1" {
		t.Skip("running inside a child; only the parent orchestrates")
	}
	if testing.Short() {
		t.Skip("double training run; skipped in -short mode")
	}
	// Cheap local pre-check so CPU-only hosts skip without paying for two
	// child processes.
	ops := numeric.Float32Ops{}
	gpu, err := compute.NewGPUEngine[float32](ops)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	_ = gpu.Close()

	// The proof pair: ZTENSOR_DETERMINISTIC=1 in both children.
	run1 := spawnDeterministicChild(t, true)
	run2 := spawnDeterministicChild(t, true)

	if len(run1) != len(run2) {
		t.Fatalf("epoch count mismatch: run1=%d run2=%d", len(run1), len(run2))
	}
	for i := range run1 {
		t.Logf("deterministic run1: %s", run1[i])
		t.Logf("deterministic run2: %s", run2[i])
		if run1[i] != run2[i] {
			t.Errorf("ZTENSOR_DETERMINISTIC=1 double-run diverged at epoch %d:\n  run1: %s\n  run2: %s",
				i+1, run1[i], run2[i])
		}
	}
	if !t.Failed() {
		t.Logf("ZTENSOR_DETERMINISTIC=1: %d epochs bitwise-identical across two processes", len(run1))
	}

	// Control pair: flag unset. Logged, never asserted.
	ctrl1 := spawnDeterministicChild(t, false)
	ctrl2 := spawnDeterministicChild(t, false)
	identical := len(ctrl1) == len(ctrl2)
	if identical {
		for i := range ctrl1 {
			if ctrl1[i] != ctrl2[i] {
				identical = false
				break
			}
		}
	}
	for i := range ctrl1 {
		t.Logf("control run1: %s", ctrl1[i])
		if i < len(ctrl2) {
			t.Logf("control run2: %s", ctrl2[i])
		}
	}
	t.Logf("control (flag unset) double-run bitwise-identical: %v "+
		"(informational only; default-mode cuBLAS heuristics are often stable for a fixed shape+GPU)", identical)
}
