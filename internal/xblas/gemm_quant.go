package xblas

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

// q4GemvParallelThreshold is the minimum N*K for M=1 Q4 GEMV parallelization.
const q4GemvParallelThreshold = 256 * 256

// GemmQ4F32 computes C = A * B where A is Q4_0 quantized and B, C are float32.
// A has logical shape (m, k), B has shape (k, n), C has shape (m, n).
// Uses the fused dequant+multiply path that avoids heap-allocating the full
// dequantized A matrix. Falls back to dequant+SgemmSimd for non-32-aligned K.
func GemmQ4F32(m, n, k int, a *tensor.Q4Storage, b, c []float32) {
	GemmQ4F32Fused(m, n, k, a, b, c)
}

// GemmQ8F32 computes C = A * B where A is Q8_0 quantized and B, C are float32.
// Dequantizes A once upfront, then uses SIMD-accelerated SGEMM.
func GemmQ8F32(m, n, k int, a *tensor.Q8Storage, b, c []float32) {
	af32 := make([]float32, a.Len())
	a.Dequantize(af32)
	SgemmSimd(m, n, k, af32, b, c)
}

// GemmQ4F32Fused computes C = dequant(A) * B with fused dequant+multiply.
// Instead of allocating a full M*K dequantized buffer, it dequantizes one
// Q4 block (32 values = 128 bytes) at a time into a stack buffer, then
// multiplies using SIMD-accelerated sgemmAccRow. This eliminates the O(M*K)
// heap allocation of GemmQ4F32, making it faster for decode (M=1) where
// the dequant allocation dominates, and equally fast for larger M.
// K must be a multiple of 32; falls back to GemmQ4F32 otherwise.
func GemmQ4F32Fused(m, n, k int, a *tensor.Q4Storage, b, c []float32) {
	if k%32 != 0 {
		GemmQ4F32(m, n, k, a, b, c)
		return
	}

	blocksPerRow := k / 32

	// Zero output.
	for i := range c {
		c[i] = 0
	}

	// Stack buffer for one dequantized Q4 block (32 float32 = 128 bytes).
	var buf [32]float32

	for i := range m {
		cRow := c[i*n : (i+1)*n]
		for bi := range blocksPerRow {
			blkIdx := i*blocksPerRow + bi
			scale := a.BlockScaleF32(blkIdx)
			if scale == 0 {
				continue
			}

			// Dequantize one block into stack buffer.
			dequantQ4Block(a.BlockData(blkIdx), scale, &buf)

			// Accumulate: c[j] += buf[p] * b[(kBase+p)*n + j] for p=0..31
			kBase := bi * 32
			for p := range 32 {
				if aVal := buf[p]; aVal != 0 {
					sgemmAccRow(unsafe.Pointer(&cRow[0]), unsafe.Pointer(&b[(kBase+p)*n]), aVal, n)
				}
			}
		}
	}
}

// GemmF32Q4NT computes C = A * B^T where A is float32 [M,K] and B is Q4_0 [N,K].
// B is stored in row-major Q4 format: each row j of B (length K) is contiguous
// in Q4 blocks. The "NT" suffix means B is Not Transposed — the caller passes B
// in its original [N,K] layout and this function computes the transpose implicitly.
// K must be a multiple of 32. Falls back to dequant+transpose+SGEMM otherwise.
func GemmF32Q4NT(m, n, k int, a []float32, b *tensor.Q4Storage, c []float32) {
	if k%32 != 0 {
		// Fallback: dequant, transpose, regular SGEMM.
		bF32 := make([]float32, n*k)
		b.Dequantize(bF32)
		bT := make([]float32, k*n)
		for r := range n {
			for col := range k {
				bT[col*n+r] = bF32[r*k+col]
			}
		}
		SgemmSimd(m, n, k, a, bT, c)
		return
	}

	blocksPerRow := k / 32

	// M=1 GEMV: parallelize across N (rows of B) when beneficial.
	if m == 1 && n*k >= q4GemvParallelThreshold {
		nCores := runtime.NumCPU()
		nCores = min(nCores, n/4)
		if nCores > 1 {
			gemmF32Q4NTParallel(n, a, b, c, blocksPerRow, nCores)
			return
		}
	}

	// For each row i of A and each row j of B, compute C[i,j] = dot(A[i,:], B[j,:]).
	// B[j,:] is contiguous in Q4 format starting at block j*blocksPerRow.
	// q4DotRow processes an entire row of Q4 blocks in a single call,
	// eliminating per-block Go function call overhead.
	for i := range m {
		aRow := a[i*k:]
		for j := range n {
			c[i*n+j] = q4DotRow(unsafe.Pointer(b.BlockPtr(j*blocksPerRow)), &aRow[0], blocksPerRow)
		}
	}
}

// gemmF32Q4NTParallel splits M=1 Q4 GEMV across nCores workers along N.
// Uses the shared worker pool if available, otherwise falls back to goroutines.
func gemmF32Q4NTParallel(n int, a []float32, b *tensor.Q4Storage, c []float32, blocksPerRow, nCores int) {
	chunkSize := (n + nCores - 1) / nCores
	if defaultPool != nil {
		tasks := make([]func(), 0, nCores)
		for t := range nCores {
			jStart := t * chunkSize
			jEnd := min(jStart+chunkSize, n)
			if jStart >= n {
				break
			}
			tasks = append(tasks, func() {
				for j := jStart; j < jEnd; j++ {
					c[j] = q4DotRow(unsafe.Pointer(b.BlockPtr(j*blocksPerRow)), &a[0], blocksPerRow)
				}
			})
		}
		defaultPool.Submit(tasks)
		return
	}
	var wg sync.WaitGroup
	for t := range nCores {
		jStart := t * chunkSize
		jEnd := min(jStart+chunkSize, n)
		if jStart >= n {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := jStart; j < jEnd; j++ {
				c[j] = q4DotRow(unsafe.Pointer(b.BlockPtr(j*blocksPerRow)), &a[0], blocksPerRow)
			}
		}()
	}
	wg.Wait()
}

// GemmF32Q8NT computes C = A * B^T where A is float32 [M,K] and B is Q8_0 [N,K].
// B is stored in row-major Q8 format: each row j of B (length K) is contiguous
// in Q8 blocks. The "NT" suffix means B is Not Transposed.
// K must be a multiple of 32. Falls back to dequant+transpose+SGEMM otherwise.
func GemmF32Q8NT(m, n, k int, a []float32, b *tensor.Q8Storage, c []float32) {
	if k%32 != 0 {
		bf32 := make([]float32, n*k)
		b.Dequantize(bf32)
		bT := make([]float32, k*n)
		for r := range n {
			for col := range k {
				bT[col*n+r] = bf32[r*k+col]
			}
		}
		SgemmSimd(m, n, k, a, bT, c)
		return
	}

	blocksPerRow := k / 32

	// M=1 GEMV: parallelize across N.
	if m == 1 && n*k >= q4GemvParallelThreshold {
		nCores := runtime.NumCPU()
		nCores = min(nCores, n/4)
		if nCores > 1 {
			gemmF32Q8NTParallel(n, k, a, b, c, blocksPerRow, nCores)
			return
		}
	}

	// For each row i of A and each row j of B, compute C[i,j] = dot(A[i,:], B[j,:]).
	var buf [32]float32
	for i := range m {
		aRow := a[i*k:]
		for j := range n {
			var sum float32
			for bi := range blocksPerRow {
				blkIdx := j*blocksPerRow + bi
				b.DequantizeBlock(blkIdx, &buf)
				kBase := bi * 32
				for p := range 32 {
					sum += aRow[kBase+p] * buf[p]
				}
			}
			c[i*n+j] = sum
		}
	}
}

// gemmF32Q8NTParallel splits M=1 Q8 GEMV across nCores workers along N.
func gemmF32Q8NTParallel(n, k int, a []float32, b *tensor.Q8Storage, c []float32, blocksPerRow, nCores int) {
	chunkSize := (n + nCores - 1) / nCores
	if defaultPool != nil {
		tasks := make([]func(), 0, nCores)
		for t := range nCores {
			jStart := t * chunkSize
			jEnd := min(jStart+chunkSize, n)
			if jStart >= n {
				break
			}
			tasks = append(tasks, func() {
				var buf [32]float32
				for j := jStart; j < jEnd; j++ {
					var sum float32
					for bi := range blocksPerRow {
						blkIdx := j*blocksPerRow + bi
						b.DequantizeBlock(blkIdx, &buf)
						kBase := bi * 32
						for p := range 32 {
							sum += a[kBase+p] * buf[p]
						}
					}
					c[j] = sum
				}
			})
		}
		defaultPool.Submit(tasks)
		return
	}
	var wg sync.WaitGroup
	for t := range nCores {
		jStart := t * chunkSize
		jEnd := min(jStart+chunkSize, n)
		if jStart >= n {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			var buf [32]float32
			for j := jStart; j < jEnd; j++ {
				var sum float32
				for bi := range blocksPerRow {
					blkIdx := j*blocksPerRow + bi
					b.DequantizeBlock(blkIdx, &buf)
					kBase := bi * 32
					for p := range 32 {
						sum += a[kBase+p] * buf[p]
					}
				}
				c[j] = sum
			}
		}()
	}
	wg.Wait()
}

// dequantQ4Block unpacks 16 packed bytes into 32 float32 values.
// GGML Q4_0 split format: low nibbles → positions 0-15, high nibbles → positions 16-31.
func dequantQ4Block(data *byte, scale float32, buf *[32]float32) {
	packed := unsafe.Slice(data, 16)
	for p := range 16 {
		byteVal := packed[p]
		buf[p] = float32(int(byteVal&0x0F)-8) * scale
		buf[p+16] = float32(int(byteVal>>4)-8) * scale
	}
}
