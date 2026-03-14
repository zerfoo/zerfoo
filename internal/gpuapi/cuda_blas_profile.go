package gpuapi

import (
	"fmt"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// cublasProfileEnabled is true when ZERFOO_PROFILE_CUBLAS=1.
var cublasProfileEnabled = os.Getenv("ZERFOO_PROFILE_CUBLAS") == "1"

// globalProfiler holds the active profiler instance for PrintCUBLASProfile.
var globalProfiler atomic.Pointer[CUDABlasProfiler]

// PrintCUBLASProfile prints the cuBLAS profiling summary if profiling is enabled.
// Safe to call even when profiling is disabled (no-op).
func PrintCUBLASProfile() {
	if p := globalProfiler.Load(); p != nil {
		p.PrintSummary()
	}
}

// callRecord stores timing for a single cuBLAS call.
type callRecord struct {
	op       string
	m, n, k  int
	batch    int
	duration time.Duration
}

// CUDABlasProfiler wraps CUDABlas with optional per-call timing.
// When ZERFOO_PROFILE_CUBLAS=1, each Sgemm/SgemmNT/batched call is timed
// and recorded. Call PrintSummary to dump stats.
type CUDABlasProfiler struct {
	inner   *CUDABlas
	mu      sync.Mutex
	records []callRecord
	enabled bool
	generation atomic.Int64
}

// WrapWithProfiler returns a profiling wrapper if ZERFOO_PROFILE_CUBLAS=1,
// otherwise returns the original CUDABlas unchanged.
func WrapWithProfiler(b *CUDABlas) *CUDABlasProfiler {
	p := &CUDABlasProfiler{
		inner:   b,
		enabled: cublasProfileEnabled,
	}
	globalProfiler.Store(p)
	return p
}

// IsEnabled returns whether profiling is active.
func (p *CUDABlasProfiler) IsEnabled() bool {
	return p.enabled
}

func (p *CUDABlasProfiler) record(op string, m, n, k, batch int, d time.Duration) {
	p.mu.Lock()
	p.records = append(p.records, callRecord{op: op, m: m, n: n, k: k, batch: batch, duration: d})
	p.mu.Unlock()
}

func (p *CUDABlasProfiler) Sgemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	if !p.enabled {
		return p.inner.Sgemm(m, n, k, alpha, a, bPtr, beta, c)
	}
	start := time.Now()
	err := p.inner.Sgemm(m, n, k, alpha, a, bPtr, beta, c)
	p.record("Sgemm", m, n, k, 1, time.Since(start))
	return err
}

func (p *CUDABlasProfiler) BFloat16Gemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	if !p.enabled {
		return p.inner.BFloat16Gemm(m, n, k, alpha, a, bPtr, beta, c)
	}
	start := time.Now()
	err := p.inner.BFloat16Gemm(m, n, k, alpha, a, bPtr, beta, c)
	p.record("BFloat16Gemm", m, n, k, 1, time.Since(start))
	return err
}

func (p *CUDABlasProfiler) Float16Gemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	if !p.enabled {
		return p.inner.Float16Gemm(m, n, k, alpha, a, bPtr, beta, c)
	}
	start := time.Now()
	err := p.inner.Float16Gemm(m, n, k, alpha, a, bPtr, beta, c)
	p.record("Float16Gemm", m, n, k, 1, time.Since(start))
	return err
}

func (p *CUDABlasProfiler) MixedFP16Gemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	if !p.enabled {
		return p.inner.MixedFP16Gemm(m, n, k, alpha, a, bPtr, beta, c)
	}
	start := time.Now()
	err := p.inner.MixedFP16Gemm(m, n, k, alpha, a, bPtr, beta, c)
	p.record("MixedFP16Gemm", m, n, k, 1, time.Since(start))
	return err
}

func (p *CUDABlasProfiler) MixedBF16Gemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	if !p.enabled {
		return p.inner.MixedBF16Gemm(m, n, k, alpha, a, bPtr, beta, c)
	}
	start := time.Now()
	err := p.inner.MixedBF16Gemm(m, n, k, alpha, a, bPtr, beta, c)
	p.record("MixedBF16Gemm", m, n, k, 1, time.Since(start))
	return err
}

func (p *CUDABlasProfiler) SgemmNT(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	if !p.enabled {
		return p.inner.SgemmNT(m, n, k, alpha, a, bPtr, beta, c)
	}
	start := time.Now()
	err := p.inner.SgemmNT(m, n, k, alpha, a, bPtr, beta, c)
	p.record("SgemmNT", m, n, k, 1, time.Since(start))
	return err
}

func (p *CUDABlasProfiler) SgemmStridedBatched(m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	bPtr unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	if !p.enabled {
		return p.inner.SgemmStridedBatched(m, n, k, alpha, a, strideA, bPtr, strideB, beta, c, strideC, batch)
	}
	start := time.Now()
	err := p.inner.SgemmStridedBatched(m, n, k, alpha, a, strideA, bPtr, strideB, beta, c, strideC, batch)
	p.record("SgemmStridedBatched", m, n, k, batch, time.Since(start))
	return err
}

func (p *CUDABlasProfiler) SgemmNTStridedBatched(m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	bPtr unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	if !p.enabled {
		return p.inner.SgemmNTStridedBatched(m, n, k, alpha, a, strideA, bPtr, strideB, beta, c, strideC, batch)
	}
	start := time.Now()
	err := p.inner.SgemmNTStridedBatched(m, n, k, alpha, a, strideA, bPtr, strideB, beta, c, strideC, batch)
	p.record("SgemmNTStridedBatched", m, n, k, batch, time.Since(start))
	return err
}

func (p *CUDABlasProfiler) SetStream(stream Stream) error {
	return p.inner.SetStream(stream)
}

func (p *CUDABlasProfiler) Destroy() error {
	return p.inner.Destroy()
}

func (p *CUDABlasProfiler) Handle() *CUDABlas {
	return p.inner
}

// ResetProfile clears all recorded calls and increments the generation.
func (p *CUDABlasProfiler) ResetProfile() {
	p.mu.Lock()
	p.records = p.records[:0]
	p.mu.Unlock()
	p.generation.Add(1)
}

// ProfileSummary holds aggregated cuBLAS profiling stats.
type ProfileSummary struct {
	TotalCalls    int
	TotalDuration time.Duration
	ByOp          []OpSummary
}

// OpSummary holds per-operation stats.
type OpSummary struct {
	Op          string
	M, N, K     int
	Batch       int
	Calls       int
	TotalTime   time.Duration
	AvgTime     time.Duration
}

// Summary returns aggregated profiling statistics.
func (p *CUDABlasProfiler) Summary() ProfileSummary {
	p.mu.Lock()
	records := make([]callRecord, len(p.records))
	copy(records, p.records)
	p.mu.Unlock()

	type key struct {
		op      string
		m, n, k int
		batch   int
	}

	agg := map[key]*OpSummary{}
	var total time.Duration
	for _, r := range records {
		total += r.duration
		k := key{r.op, r.m, r.n, r.k, r.batch}
		s, ok := agg[k]
		if !ok {
			s = &OpSummary{Op: r.op, M: r.m, N: r.n, K: r.k, Batch: r.batch}
			agg[k] = s
		}
		s.Calls++
		s.TotalTime += r.duration
	}

	ops := make([]OpSummary, 0, len(agg))
	for _, s := range agg {
		s.AvgTime = s.TotalTime / time.Duration(s.Calls)
		ops = append(ops, *s)
	}
	sort.Slice(ops, func(i, j int) bool {
		return ops[i].TotalTime > ops[j].TotalTime
	})

	return ProfileSummary{
		TotalCalls:    len(records),
		TotalDuration: total,
		ByOp:          ops,
	}
}

// PrintSummary prints cuBLAS profiling stats to stderr.
func (p *CUDABlasProfiler) PrintSummary() {
	s := p.Summary()
	if s.TotalCalls == 0 {
		return
	}

	fmt.Fprintf(os.Stderr, "\n=== cuBLAS Profile Summary ===\n")
	fmt.Fprintf(os.Stderr, "Total calls: %d\n", s.TotalCalls)
	fmt.Fprintf(os.Stderr, "Total cuBLAS time: %v\n", s.TotalDuration)
	if s.TotalCalls > 0 {
		fmt.Fprintf(os.Stderr, "Avg per call: %v\n", s.TotalDuration/time.Duration(s.TotalCalls))
	}
	fmt.Fprintf(os.Stderr, "\nPer-operation breakdown (sorted by total time):\n")
	fmt.Fprintf(os.Stderr, "%-30s %6s %6s %6s %6s %8s %12s %12s\n",
		"Operation", "M", "N", "K", "Batch", "Calls", "Total", "Avg")
	fmt.Fprintf(os.Stderr, "%s\n", "--------------------------------------------------------------------------------------------")
	for _, op := range s.ByOp {
		fmt.Fprintf(os.Stderr, "%-30s %6d %6d %6d %6d %8d %12v %12v\n",
			op.Op, op.M, op.N, op.K, op.Batch, op.Calls, op.TotalTime, op.AvgTime)
	}
	fmt.Fprintf(os.Stderr, "==============================\n\n")
}

// Compile-time interface assertions.
var _ BLAS = (*CUDABlasProfiler)(nil)
var _ BLASTransposeB = (*CUDABlasProfiler)(nil)
var _ BLASBatched = (*CUDABlasProfiler)(nil)
var _ BLASBatchedTransposeB = (*CUDABlasProfiler)(nil)
