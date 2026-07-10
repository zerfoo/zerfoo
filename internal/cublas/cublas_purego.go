package cublas

import (
	"fmt"
	"os"
	"sync"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// Available returns true if the cuBLAS library can be loaded at runtime.
// The result is cached after the first call.
func Available() bool {
	_, err := getCublasLib()
	return err == nil
}

// CudaDataType identifies the element data type for cublasGemmEx.
type CudaDataType int

const (
	CudaR32F    CudaDataType = 0  // CUDA_R_32F  (float32)
	CudaR16F    CudaDataType = 2  // CUDA_R_16F  (float16)
	CudaR16BF   CudaDataType = 14 // CUDA_R_16BF (bfloat16)
	CudaR8F_E4M3 CudaDataType = 28 // CUDA_R_8F_E4M3 (fp8 e4m3)
)

// CublasComputeType identifies the compute precision for cublasGemmEx.
type CublasComputeType int

const (
	CublasCompute32F CublasComputeType = 68 // CUBLAS_COMPUTE_32F
)

// cuBLAS status codes.
const cublasStatusSuccess = 0

// cublasLib holds dlopen function pointers for cuBLAS.
type cublasLib struct {
	create              uintptr // cublasCreate_v2
	destroy             uintptr // cublasDestroy_v2
	setStream           uintptr // cublasSetStream_v2
	sgemm               uintptr // cublasSgemm_v2
	gemmEx              uintptr // cublasGemmEx
	sgemmStridedBatched uintptr // cublasSgemmStridedBatched
}

var (
	cblasLib     *cublasLib
	cblasOnce    sync.Once
	errCblasLoad error
)

// cublasLibPathOverrideEnv names an environment variable that, if set to a
// vetted absolute path, is tried before trustedCublasLibPaths. See
// cuda.VetAbsoluteLibPath for the safety checks applied.
const cublasLibPathOverrideEnv = "ZERFOO_CUBLAS_LIB_PATH"

// trustedCublasLibPaths mirror the CUDA toolkit install location zerfoo's
// own deployments use (docs/bench/manifests/*.yaml mount /usr/local/cuda
// read-only and set LD_LIBRARY_PATH=/usr/local/cuda/lib64). Trying these
// absolute paths first means dlopen never has to consult
// LD_LIBRARY_PATH/RPATH at all on a host that matches this convention.
var trustedCublasLibPaths = []string{
	"/usr/local/cuda/lib64/libcublas.so.12",
	"/usr/local/cuda/lib64/libcublas.so",
}

// cuBLAS library paths to try, in order.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): the trailing
// bare-soname entries are a DOCUMENTED residual trust assumption, not an
// oversight -- see the equivalent comment on cudartPaths in
// internal/cuda/purego.go for the full rationale (cuBLAS ships as part of
// the CUDA toolkit, a third-party install whose location genuinely varies
// across hosts, unlike zerfoo's own libkernels.so). We try the trusted
// zerfoo-deployment path (and any vetted operator override) first, and
// only fall back to a bare soname when no trusted absolute install is
// found.
var cublasLibPaths = buildCublasLibPaths()

func buildCublasLibPaths() []string {
	paths := make([]string, 0, len(trustedCublasLibPaths)+3)
	if override := os.Getenv(cublasLibPathOverrideEnv); override != "" {
		if vetted, ok := cuda.VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
	}
	paths = append(paths, trustedCublasLibPaths...)
	paths = append(paths, "libcublas.so.12", "libcublas.so")
	return paths
}

func loadCublas() (*cublasLib, error) {
	var handle uintptr
	var lastErr string
	for _, path := range cublasLibPaths {
		var err error
		handle, err = cuda.DlopenPath(path)
		if err == nil {
			break
		}
		lastErr = err.Error()
	}
	if handle == 0 {
		return nil, fmt.Errorf("cublas: dlopen failed: %s", lastErr)
	}

	lib := &cublasLib{}
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"cublasCreate_v2", &lib.create},
		{"cublasDestroy_v2", &lib.destroy},
		{"cublasSetStream_v2", &lib.setStream},
		{"cublasSgemm_v2", &lib.sgemm},
		{"cublasGemmEx", &lib.gemmEx},
		{"cublasSgemmStridedBatched", &lib.sgemmStridedBatched},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("cublas: %w", err)
		}
		*s.ptr = addr
	}
	return lib, nil
}

func getCublasLib() (*cublasLib, error) {
	cblasOnce.Do(func() {
		cblasLib, errCblasLoad = loadCublas()
	})
	return cblasLib, errCblasLoad
}

// Handle wraps a cuBLAS handle (opaque pointer).
type Handle struct {
	ptr uintptr // cublasHandle_t is a pointer
}

// CreateHandle creates a new cuBLAS context handle.
func CreateHandle() (*Handle, error) {
	lib, err := getCublasLib()
	if err != nil {
		return nil, err
	}
	var h uintptr
	status := cuda.Ccall(lib.create, uintptr(unsafe.Pointer(&h)))
	if status != cublasStatusSuccess {
		return nil, fmt.Errorf("cublasCreate failed with status %d", status)
	}
	return &Handle{ptr: h}, nil
}

// Destroy releases the cuBLAS handle resources.
func (h *Handle) Destroy() error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.destroy, h.ptr)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasDestroy failed with status %d", status)
	}
	return nil
}

// SetStream associates a CUDA stream with this cuBLAS handle.
func (h *Handle) SetStream(streamPtr unsafe.Pointer) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.setStream, h.ptr, uintptr(streamPtr))
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSetStream failed with status %d", status)
	}
	return nil
}

// cuBLAS operation constants.
const cublasOpN = 0 // CUBLAS_OP_N
const cublasOpT = 1 // CUBLAS_OP_T

// Sgemm performs single-precision general matrix multiplication.
// Row-major to column-major conversion: swap A/B and m/n.
func Sgemm(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	// cuBLAS Sgemm takes pointers to alpha and beta.
	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: swap A<->B, swap m<->n.
	// cublasSgemm_v2(handle, transB, transA, n, m, k, &alpha, B, n, A, k, &beta, C, n)
	status := cuda.Ccall(lib.sgemm,
		h.ptr,
		uintptr(cublasOpN), // transa (for B)
		uintptr(cublasOpN), // transb (for A)
		uintptr(n),         // rows of op(B) = cols of C
		uintptr(m),         // cols of op(A) = rows of C
		uintptr(k),         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b),  // B first
		uintptr(n),  // ldb
		uintptr(a),  // A second
		uintptr(k),  // lda
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(n), // ldc
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSgemm failed with status %d", status)
	}
	return nil
}

// SgemmNT performs single-precision C = A * B^T where A is [m, k] and
// B is [n, k] (row-major). Uses CUBLAS_OP_T on the first cuBLAS argument.
func SgemmNT(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: B comes first with CUBLAS_OP_T, A second with CUBLAS_OP_N.
	status := cuda.Ccall(lib.sgemm,
		h.ptr,
		uintptr(cublasOpT), // transpose B (cuBLAS first arg)
		uintptr(cublasOpN), // no-transpose A (cuBLAS second arg)
		uintptr(n),         // rows of op(B) = n
		uintptr(m),         // cols of op(A) = m
		uintptr(k),         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b),  // B first (cuBLAS convention)
		uintptr(k),  // ldb = k (B_rm row width)
		uintptr(a),  // A second
		uintptr(k),  // lda = k (A_rm row width)
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(n), // ldc = n (C_rm row width)
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSgemm(NT) failed with status %d", status)
	}
	return nil
}

// SgemmStridedBatched performs batched single-precision GEMM with strided access.
// Row-major to column-major conversion: swap A/B and m/n (same trick as Sgemm).
//
// Parameters (in row-major terms):
//
//	m        - rows of A and C per batch
//	n        - columns of B and C per batch
//	k        - columns of A / rows of B
//	alpha    - scalar multiplier for A*B
//	a        - device pointer to A[0] (m x k, row-major)
//	strideA  - element stride between consecutive A matrices
//	b        - device pointer to B[0] (k x n, row-major)
//	strideB  - element stride between consecutive B matrices
//	beta     - scalar multiplier for C
//	c        - device pointer to C[0] (m x n, row-major), output
//	strideC  - element stride between consecutive C matrices
//	batch    - number of matrices in the batch
//nolint:dupl // SgemmStridedBatched and SgemmNTStridedBatched differ in transpose flags
func SgemmStridedBatched(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	b unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: swap A<->B, swap m<->n, swap strides.
	// cublasSgemmStridedBatched(handle, transB, transA, n, m, k,
	//   &alpha, B, n, strideB, A, k, strideA, &beta, C, n, strideC, batchCount)
	status := cuda.Ccall(lib.sgemmStridedBatched,
		h.ptr,
		uintptr(cublasOpN),                 // transa (for B)
		uintptr(cublasOpN),                 // transb (for A)
		uintptr(n),                         // rows of op(B) = cols of C
		uintptr(m),                         // cols of op(A) = rows of C
		uintptr(k),                         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b),                         // B first
		uintptr(n),                         // ldb
		uintptr(strideB),                   // strideB
		uintptr(a),                         // A second
		uintptr(k),                         // lda
		uintptr(strideA),                   // strideA
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(n),                         // ldc
		uintptr(strideC),                   // strideC
		uintptr(batch),                     // batchCount
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSgemmStridedBatched failed with status %d", status)
	}
	return nil
}

// SgemmNTStridedBatched performs batched C = A * B^T using strided batched GEMM
// with CUBLAS_OP_T on the B operand.
//nolint:dupl // differs from SgemmStridedBatched in transpose flags
func SgemmNTStridedBatched(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	b unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: B with OP_T first, A with OP_N second.
	status := cuda.Ccall(lib.sgemmStridedBatched,
		h.ptr,
		uintptr(cublasOpT),                 // transpose B (cuBLAS first arg)
		uintptr(cublasOpN),                 // no-transpose A (cuBLAS second arg)
		uintptr(n),                         // rows of op(B) = n
		uintptr(m),                         // cols of op(A) = m
		uintptr(k),                         // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b),                         // B first
		uintptr(k),                         // ldb = k (B_rm row width)
		uintptr(strideB),                   // strideB
		uintptr(a),                         // A second
		uintptr(k),                         // lda = k (A_rm row width)
		uintptr(strideA),                   // strideA
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(n),                         // ldc = n (C_rm row width)
		uintptr(strideC),                   // strideC
		uintptr(batch),                     // batchCount
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasSgemmNTStridedBatched failed with status %d", status)
	}
	return nil
}

// cublasGemmDefault is the CUBLAS_GEMM_DEFAULT algorithm selector.
// The C enum value is -1; as an unsigned 32-bit integer this is 0xFFFFFFFF.
const cublasGemmDefault uintptr = 0xFFFFFFFF

// GemmEx performs mixed-precision general matrix multiplication.
// Row-major to column-major conversion: swap A/B and m/n.
func GemmEx(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, aType CudaDataType,
	b unsafe.Pointer, bType CudaDataType,
	beta float32,
	c unsafe.Pointer, cType CudaDataType,
	computeType CublasComputeType,
) error {
	lib, err := getCublasLib()
	if err != nil {
		return err
	}

	cAlpha := alpha
	cBeta := beta

	// Row-major to column-major: swap A<->B, swap m<->n.
	// cublasGemmEx(handle, transa, transb, m, n, k,
	//   alpha, B, Btype, ldb, A, Atype, lda,
	//   beta, C, Ctype, ldc, computeType, algo)
	status := cuda.Ccall(lib.gemmEx,
		h.ptr,
		uintptr(cublasOpN),                  // transa (for B)
		uintptr(cublasOpN),                  // transb (for A)
		uintptr(n),                          // rows of op(B) = cols of C
		uintptr(m),                          // cols of op(A) = rows of C
		uintptr(k),                          // inner dimension
		uintptr(unsafe.Pointer(&cAlpha)),
		uintptr(b),                          // B first
		uintptr(bType),                      // Btype
		uintptr(n),                          // ldb
		uintptr(a),                          // A second
		uintptr(aType),                      // Atype
		uintptr(k),                          // lda
		uintptr(unsafe.Pointer(&cBeta)),
		uintptr(c),
		uintptr(cType),                      // Ctype
		uintptr(n),                          // ldc
		uintptr(computeType),                // computeType
		cublasGemmDefault,                   // algo
	)
	if status != cublasStatusSuccess {
		return fmt.Errorf("cublasGemmEx failed with status %d", status)
	}
	return nil
}
