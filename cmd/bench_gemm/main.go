// Command bench_gemm micro-benchmarks low-precision (fp8 / Q4_K) weight GEMMs
// against an f32 baseline in ztensor's engine, at the n>1 ("batched", not
// n==1 decode) shapes representative of the LTX-2 DiT denoise regime.
//
// This is the measurement tool for E127/T127.1.0b PART 2: it sizes whether the
// denoise-regime dequant-to-f32 path is acceptable or whether new kernels are
// needed before the converter storage mapping and perf budget are committed.
// The fp8 sub-format is F8_E4M3 (confirmed by Phase-0 header read; see
// docs/devlog.md), which is what tensor.NewFP8E4M3Storage produces.
//
// IMPORTANT: the real numbers MUST be produced on the GB10 via Spark (see
// docs/bench/manifests/ltx2-fp8-spike.yaml and the Hardware doctrine in
// CLAUDE.md), NOT via interactive SSH. On a host without CUDA the GPU engine
// is unavailable; -cpu runs the harness against the CPU engine for a local
// smoke (f32, plus any quant path the CPU engine supports via dequant).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func main() {
	m := flag.Int("m", 4096, "output rows (DiT hidden/out dim; video stream inner = 4096)")
	k := flag.Int("k", 4096, "contraction dim (in features; must be 256-aligned for Q4_K)")
	nsArg := flag.String("ns", "256,1024,4096", "comma-separated n values (token/batch counts; the n>1 regime)")
	iters := flag.Int("iters", 50, "timed iterations per variant")
	warmup := flag.Int("warmup", 10, "warmup iterations per variant")
	cpu := flag.Bool("cpu", false, "use the CPU engine (local smoke only; real numbers need the GB10 GPU)")
	out := flag.String("out", "", "optional output file (results appended); default stdout")
	flag.Parse()

	ns, err := parseInts(*nsArg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bench_gemm: bad -ns %q: %v\n", *nsArg, err)
		os.Exit(2)
	}

	ctx := context.Background()
	ops := numeric.Float32Ops{}

	var engine compute.Engine[float32]
	var closeFn func()
	if *cpu {
		engine = compute.NewCPUEngine[float32](ops)
		closeFn = func() {}
		fmt.Fprintln(os.Stderr, "bench_gemm: CPU engine (local smoke; f32 only is meaningful)")
	} else {
		gpu, gerr := compute.NewGPUEngine[float32](ops)
		if gerr != nil {
			fmt.Fprintf(os.Stderr, "bench_gemm: GPU engine unavailable: %v\n(run on the GB10 via Spark, or pass -cpu for a local smoke)\n", gerr)
			os.Exit(1)
		}
		engine = gpu
		closeFn = func() { _ = gpu.Close() }
	}
	defer closeFn()

	var sb strings.Builder
	fmt.Fprintf(&sb, "# bench_gemm  m=%d k=%d  iters=%d warmup=%d  engine=%s\n", *m, *k, *iters, *warmup, engineName(*cpu))
	fmt.Fprintf(&sb, "# %-10s %-8s %12s %12s %10s\n", "variant", "n", "ms/op", "GFLOP/s", "vs f32")

	for _, n := range ns {
		base, ok := runVariant(ctx, engine, "f32", *m, *k, n, *iters, *warmup)
		report(&sb, "f32", *m, *k, n, base, base, ok)
		for _, v := range []string{"fp8", "q4k"} {
			r, vok := runVariant(ctx, engine, v, *m, *k, n, *iters, *warmup)
			report(&sb, v, *m, *k, n, r, base, vok)
		}
	}

	result := sb.String()
	if *out != "" {
		f, ferr := os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if ferr != nil {
			fmt.Fprintf(os.Stderr, "bench_gemm: open -out: %v\n", ferr)
			os.Exit(1)
		}
		defer func() { _ = f.Close() }()
		if _, werr := f.WriteString(result); werr != nil {
			fmt.Fprintf(os.Stderr, "bench_gemm: write -out: %v\n", werr)
			os.Exit(1)
		}
	}
	fmt.Print(result)
}

// runVariant times one (variant, n) GEMM: A=[m,k] weights, B=[k,n] f32
// activations, output [m,n]. f32 keeps A in f32; fp8/q4k quantize the weights
// so MatMul dispatches to the corresponding low-precision path. Returns the
// mean seconds/op and whether it ran.
func runVariant(ctx context.Context, e compute.Engine[float32], variant string, m, k, n, iters, warmup int) (float64, bool) {
	wData := randData(m*k, 1)
	xData := randData(k*n, 2)

	x, err := tensor.New[float32]([]int{k, n}, xData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "  %s n=%d: activation alloc: %v\n", variant, n, err)
		return 0, false
	}

	var a *tensor.TensorNumeric[float32]
	switch variant {
	case "f32":
		a, err = tensor.New[float32]([]int{m, k}, wData)
	case "fp8":
		a, err = tensor.NewWithStorage[float32]([]int{m, k}, tensor.NewFP8E4M3Storage(wData))
	case "q4k":
		if (m*k)%256 != 0 {
			fmt.Fprintf(os.Stderr, "  q4k n=%d: skipped (m*k=%d not 256-aligned)\n", n, m*k)
			return 0, false
		}
		a, err = tensor.NewWithStorage[float32]([]int{m, k}, tensor.QuantizeQ4K(wData))
	default:
		return 0, false
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "  %s n=%d: weight alloc: %v\n", variant, n, err)
		return 0, false
	}

	// Make operands device-resident where the engine supports it (GPU). The
	// CPU engine has no UploadWeights; the type assertion keeps this generic.
	if up, ok := any(e).(interface {
		UploadWeights([]*tensor.TensorNumeric[float32]) error
	}); ok {
		if uerr := up.UploadWeights([]*tensor.TensorNumeric[float32]{a, x}); uerr != nil {
			fmt.Fprintf(os.Stderr, "  %s n=%d: UploadWeights: %v\n", variant, n, uerr)
			return 0, false
		}
	}

	syncer, _ := any(e).(interface{ Sync() error })
	sync := func() {
		if syncer != nil {
			_ = syncer.Sync()
		}
	}

	for range warmup {
		if _, werr := e.MatMul(ctx, a, x); werr != nil {
			fmt.Fprintf(os.Stderr, "  %s n=%d: MatMul: %v\n", variant, n, werr)
			return 0, false
		}
	}
	sync()

	start := time.Now()
	for range iters {
		if _, werr := e.MatMul(ctx, a, x); werr != nil {
			fmt.Fprintf(os.Stderr, "  %s n=%d: MatMul: %v\n", variant, n, werr)
			return 0, false
		}
	}
	sync()
	elapsed := time.Since(start).Seconds()
	return elapsed / float64(iters), true
}

func report(sb *strings.Builder, variant string, m, k, n int, secPerOp, baseSecPerOp float64, ok bool) {
	if !ok {
		fmt.Fprintf(sb, "  %-10s %-8d %12s %12s %10s\n", variant, n, "n/a", "n/a", "n/a")
		return
	}
	flops := 2.0 * float64(m) * float64(k) * float64(n)
	gflops := flops / secPerOp / 1e9
	speedup := "-"
	if variant != "f32" && baseSecPerOp > 0 && secPerOp > 0 {
		speedup = fmt.Sprintf("%.2fx", baseSecPerOp/secPerOp)
	}
	fmt.Fprintf(sb, "  %-10s %-8d %12.4f %12.1f %10s\n", variant, n, secPerOp*1e3, gflops, speedup)
}

// randData returns a deterministic pseudo-random slice in [-1, 1) seeded by
// `seed` (no math/rand: keeps the bench reproducible across runs/hosts).
func randData(n, seed int) []float32 {
	d := make([]float32, n)
	s := uint64(seed*2654435761 + 1)
	for i := range d {
		s = s*6364136223846793005 + 1442695040888963407
		d[i] = float32(int64(s>>11))/float32(1<<52)*2 - 1
	}
	return d
}

func parseInts(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no values")
	}
	return out, nil
}

func engineName(cpu bool) string {
	if cpu {
		return "cpu"
	}
	return "gpu"
}
