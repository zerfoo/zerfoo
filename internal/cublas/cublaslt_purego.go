package cublas

import (
	"fmt"
	"os"
	"sync"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cublasLt status codes.
const cublasLtStatusSuccess = 0

// LtComputeType specifies the compute precision for cublasLt matmul.
type LtComputeType int

const (
	LtComputeF32 LtComputeType = 68 // CUBLAS_COMPUTE_32F
	LtComputeF16 LtComputeType = 64 // CUBLAS_COMPUTE_16F
)

// LtMatmulDescAttribute identifies attributes of a matmul descriptor.
type LtMatmulDescAttribute int

const (
	// LtMatmulDescScaleType sets the scale type for the matmul operation.
	LtMatmulDescScaleType LtMatmulDescAttribute = 0 // CUBLASLT_MATMUL_DESC_SCALE_TYPE
	// LtMatmulDescPointerMode sets the pointer mode.
	LtMatmulDescPointerMode LtMatmulDescAttribute = 1 // CUBLASLT_MATMUL_DESC_POINTER_MODE
	// LtMatmulDescTransA sets the transpose mode for matrix A.
	LtMatmulDescTransA LtMatmulDescAttribute = 2 // CUBLASLT_MATMUL_DESC_TRANSA
	// LtMatmulDescTransB sets the transpose mode for matrix B.
	LtMatmulDescTransB LtMatmulDescAttribute = 3 // CUBLASLT_MATMUL_DESC_TRANSB
	// LtMatmulDescEpilogue sets the epilogue function.
	LtMatmulDescEpilogue LtMatmulDescAttribute = 5 // CUBLASLT_MATMUL_DESC_EPILOGUE
	// LtMatmulDescEpilogueAuxPointer sets the auxiliary epilogue pointer.
	LtMatmulDescEpilogueAuxPointer LtMatmulDescAttribute = 6 // CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
	// LtMatmulDescEpilogueAuxLd sets the leading dimension of the auxiliary epilogue buffer.
	LtMatmulDescEpilogueAuxLd LtMatmulDescAttribute = 7 // CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
	// LtMatmulDescAScalePointer sets the A scale pointer (for FP8).
	LtMatmulDescAScalePointer LtMatmulDescAttribute = 17 // CUBLASLT_MATMUL_DESC_A_SCALE_POINTER
	// LtMatmulDescBScalePointer sets the B scale pointer (for FP8).
	LtMatmulDescBScalePointer LtMatmulDescAttribute = 18 // CUBLASLT_MATMUL_DESC_B_SCALE_POINTER
	// LtMatmulDescDScalePointer sets the D scale pointer (for FP8).
	LtMatmulDescDScalePointer LtMatmulDescAttribute = 20 // CUBLASLT_MATMUL_DESC_D_SCALE_POINTER
)

// LtMatrixLayoutAttribute identifies attributes of a matrix layout.
type LtMatrixLayoutAttribute int

const (
	// LtMatrixLayoutType sets the data type of the matrix.
	LtMatrixLayoutType LtMatrixLayoutAttribute = 0 // CUBLASLT_MATRIX_LAYOUT_TYPE
	// LtMatrixLayoutOrder sets the memory order (row/col major).
	LtMatrixLayoutOrder LtMatrixLayoutAttribute = 1 // CUBLASLT_MATRIX_LAYOUT_ORDER
	// LtMatrixLayoutRows sets the number of rows.
	LtMatrixLayoutRows LtMatrixLayoutAttribute = 2 // CUBLASLT_MATRIX_LAYOUT_ROWS
	// LtMatrixLayoutCols sets the number of columns.
	LtMatrixLayoutCols LtMatrixLayoutAttribute = 3 // CUBLASLT_MATRIX_LAYOUT_COLS
	// LtMatrixLayoutLD sets the leading dimension.
	LtMatrixLayoutLD LtMatrixLayoutAttribute = 4 // CUBLASLT_MATRIX_LAYOUT_LD
	// LtMatrixLayoutBatchCount sets the batch count.
	LtMatrixLayoutBatchCount LtMatrixLayoutAttribute = 5 // CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
	// LtMatrixLayoutStridedBatchOffset sets the strided batch offset.
	LtMatrixLayoutStridedBatchOffset LtMatrixLayoutAttribute = 6 // CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
)

// LtOrder specifies memory layout order.
type LtOrder int32

const (
	LtOrderCol LtOrder = 0 // CUBLASLT_ORDER_COL
	LtOrderRow LtOrder = 1 // CUBLASLT_ORDER_ROW
)

// LtEpilogue specifies the epilogue operation applied after the matmul.
type LtEpilogue int32

const (
	LtEpilogueDefault LtEpilogue = 1   // CUBLASLT_EPILOGUE_DEFAULT
	LtEpilogueReLU    LtEpilogue = 2   // CUBLASLT_EPILOGUE_RELU
	LtEpilogueGeLU    LtEpilogue = 32  // CUBLASLT_EPILOGUE_GELU
	LtEpilogueBias    LtEpilogue = 128 // CUBLASLT_EPILOGUE_BIAS
)

// cublasLtResultEntry maps to cublasLtMatmulHeuristicResult_t.
// The struct is 1056 bytes: algo (1024 bytes) + workspaceSize (8 bytes) +
// state (4 bytes) + wavesCount (4 bytes) + 16 bytes reserved.
const sizeofLtHeuristicResult = 1056

// cublasLtLib holds dlopen function pointers for cuBLASLt.
type cublasLtLib struct {
	create              uintptr // cublasLtCreate
	destroy             uintptr // cublasLtDestroy
	matmulDescCreate    uintptr // cublasLtMatmulDescCreate
	matmulDescDestroy   uintptr // cublasLtMatmulDescDestroy
	matmulDescSetAttr   uintptr // cublasLtMatmulDescSetAttribute
	matLayoutCreate     uintptr // cublasLtMatrixLayoutCreate
	matLayoutDestroy    uintptr // cublasLtMatrixLayoutDestroy
	matLayoutSetAttr    uintptr // cublasLtMatrixLayoutSetAttribute
	matmul              uintptr // cublasLtMatmul
	matmulPrefCreate    uintptr // cublasLtMatmulPreferenceCreate
	matmulPrefDestroy   uintptr // cublasLtMatmulPreferenceDestroy
	matmulAlgoGetHeur   uintptr // cublasLtMatmulAlgoGetHeuristic
}

var (
	ltLib     *cublasLtLib
	ltOnce    sync.Once
	errLtLoad error
)

// cublasLtLibPathOverrideEnv names an environment variable that, if set to
// a vetted absolute path, is tried before trustedCublasLtLibPaths. See
// cuda.VetAbsoluteLibPath for the safety checks applied.
const cublasLtLibPathOverrideEnv = "ZERFOO_CUBLASLT_LIB_PATH"

// trustedCublasLtLibPaths mirror the CUDA toolkit install location zerfoo's
// own deployments use (docs/bench/manifests/*.yaml mount /usr/local/cuda
// read-only and set LD_LIBRARY_PATH=/usr/local/cuda/lib64).
var trustedCublasLtLibPaths = []string{
	"/usr/local/cuda/lib64/libcublasLt.so.12",
	"/usr/local/cuda/lib64/libcublasLt.so",
}

// cublasLt library paths to try, in order.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): the trailing
// bare-soname entries are a DOCUMENTED residual trust assumption -- see the
// equivalent comment on cublasLibPaths above and on cudartPaths in
// internal/cuda/purego.go for the full rationale.
var cublasLtLibPaths = buildCublasLtLibPaths()

func buildCublasLtLibPaths() []string {
	paths := make([]string, 0, len(trustedCublasLtLibPaths)+3)
	if override := os.Getenv(cublasLtLibPathOverrideEnv); override != "" {
		if vetted, ok := cuda.VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
	}
	paths = append(paths, trustedCublasLtLibPaths...)
	paths = append(paths, "libcublasLt.so.12", "libcublasLt.so")
	return paths
}

func loadCublasLt() (*cublasLtLib, error) {
	var handle uintptr
	var lastErr string
	for _, path := range cublasLtLibPaths {
		var err error
		handle, err = cuda.DlopenPath(path)
		if err == nil {
			break
		}
		lastErr = err.Error()
	}
	if handle == 0 {
		return nil, fmt.Errorf("cublasLt: dlopen failed: %s", lastErr)
	}

	lib := &cublasLtLib{}
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"cublasLtCreate", &lib.create},
		{"cublasLtDestroy", &lib.destroy},
		{"cublasLtMatmulDescCreate", &lib.matmulDescCreate},
		{"cublasLtMatmulDescDestroy", &lib.matmulDescDestroy},
		{"cublasLtMatmulDescSetAttribute", &lib.matmulDescSetAttr},
		{"cublasLtMatrixLayoutCreate", &lib.matLayoutCreate},
		{"cublasLtMatrixLayoutDestroy", &lib.matLayoutDestroy},
		{"cublasLtMatrixLayoutSetAttribute", &lib.matLayoutSetAttr},
		{"cublasLtMatmul", &lib.matmul},
		{"cublasLtMatmulPreferenceCreate", &lib.matmulPrefCreate},
		{"cublasLtMatmulPreferenceDestroy", &lib.matmulPrefDestroy},
		{"cublasLtMatmulAlgoGetHeuristic", &lib.matmulAlgoGetHeur},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("cublasLt: %w", err)
		}
		*s.ptr = addr
	}
	return lib, nil
}

func getCublasLtLib() (*cublasLtLib, error) {
	ltOnce.Do(func() {
		ltLib, errLtLoad = loadCublasLt()
	})
	return ltLib, errLtLoad
}

// LtAvailable returns true if the cuBLASLt library can be loaded at runtime.
// The result is cached after the first call.
func LtAvailable() bool {
	_, err := getCublasLtLib()
	return err == nil
}

// LtHandle wraps a cublasLtHandle_t (opaque pointer).
type LtHandle struct {
	ptr uintptr
}

// LtCreateHandle creates a new cuBLASLt context handle.
func LtCreateHandle() (*LtHandle, error) {
	lib, err := getCublasLtLib()
	if err != nil {
		return nil, err
	}
	var h uintptr
	status := cuda.Ccall(lib.create, uintptr(unsafe.Pointer(&h)))
	if status != cublasLtStatusSuccess {
		return nil, fmt.Errorf("cublasLtCreate failed with status %d", status)
	}
	return &LtHandle{ptr: h}, nil
}

// Destroy releases the cuBLASLt handle resources.
func (h *LtHandle) Destroy() error {
	lib, err := getCublasLtLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.destroy, h.ptr)
	if status != cublasLtStatusSuccess {
		return fmt.Errorf("cublasLtDestroy failed with status %d", status)
	}
	return nil
}

// MatmulDesc wraps a cublasLtMatmulDesc_t (opaque pointer).
type MatmulDesc struct {
	ptr uintptr
}

// CreateMatmulDesc creates a new matmul descriptor with the given compute type
// and scale type (cudaDataType for the scale, e.g. CudaR32F).
func CreateMatmulDesc(computeType LtComputeType, scaleType CudaDataType) (*MatmulDesc, error) {
	lib, err := getCublasLtLib()
	if err != nil {
		return nil, err
	}
	var desc uintptr
	status := cuda.Ccall(lib.matmulDescCreate,
		uintptr(unsafe.Pointer(&desc)),
		uintptr(computeType),
		uintptr(scaleType),
	)
	if status != cublasLtStatusSuccess {
		return nil, fmt.Errorf("cublasLtMatmulDescCreate failed with status %d", status)
	}
	return &MatmulDesc{ptr: desc}, nil
}

// Destroy releases the matmul descriptor.
func (d *MatmulDesc) Destroy() error {
	lib, err := getCublasLtLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.matmulDescDestroy, d.ptr)
	if status != cublasLtStatusSuccess {
		return fmt.Errorf("cublasLtMatmulDescDestroy failed with status %d", status)
	}
	return nil
}

// SetAttribute sets an attribute on the matmul descriptor.
// value must point to the attribute value, and sizeInBytes is its size.
func (d *MatmulDesc) SetAttribute(attr LtMatmulDescAttribute, value unsafe.Pointer, sizeInBytes int) error {
	lib, err := getCublasLtLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.matmulDescSetAttr,
		d.ptr,
		uintptr(attr),
		uintptr(value),
		uintptr(sizeInBytes),
	)
	if status != cublasLtStatusSuccess {
		return fmt.Errorf("cublasLtMatmulDescSetAttribute(%d) failed with status %d", attr, status)
	}
	return nil
}

// MatrixLayout wraps a cublasLtMatrixLayout_t (opaque pointer).
type MatrixLayout struct {
	ptr uintptr
}

// CreateMatrixLayout creates a new matrix layout descriptor.
// dataType is the element type (e.g. CudaR32F), rows/cols are the matrix
// dimensions, and ld is the leading dimension.
func CreateMatrixLayout(dataType CudaDataType, rows, cols, ld int) (*MatrixLayout, error) {
	lib, err := getCublasLtLib()
	if err != nil {
		return nil, err
	}
	var layout uintptr
	status := cuda.Ccall(lib.matLayoutCreate,
		uintptr(unsafe.Pointer(&layout)),
		uintptr(dataType),
		uintptr(rows),
		uintptr(cols),
		uintptr(ld),
	)
	if status != cublasLtStatusSuccess {
		return nil, fmt.Errorf("cublasLtMatrixLayoutCreate failed with status %d", status)
	}
	return &MatrixLayout{ptr: layout}, nil
}

// Destroy releases the matrix layout descriptor.
func (l *MatrixLayout) Destroy() error {
	lib, err := getCublasLtLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.matLayoutDestroy, l.ptr)
	if status != cublasLtStatusSuccess {
		return fmt.Errorf("cublasLtMatrixLayoutDestroy failed with status %d", status)
	}
	return nil
}

// SetAttribute sets an attribute on the matrix layout.
func (l *MatrixLayout) SetAttribute(attr LtMatrixLayoutAttribute, value unsafe.Pointer, sizeInBytes int) error {
	lib, err := getCublasLtLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.matLayoutSetAttr,
		l.ptr,
		uintptr(attr),
		uintptr(value),
		uintptr(sizeInBytes),
	)
	if status != cublasLtStatusSuccess {
		return fmt.Errorf("cublasLtMatrixLayoutSetAttribute(%d) failed with status %d", attr, status)
	}
	return nil
}

// MatmulPreference wraps a cublasLtMatmulPreference_t (opaque pointer).
type MatmulPreference struct {
	ptr uintptr
}

// CreateMatmulPreference creates a new matmul preference descriptor.
func CreateMatmulPreference() (*MatmulPreference, error) {
	lib, err := getCublasLtLib()
	if err != nil {
		return nil, err
	}
	var pref uintptr
	status := cuda.Ccall(lib.matmulPrefCreate, uintptr(unsafe.Pointer(&pref)))
	if status != cublasLtStatusSuccess {
		return nil, fmt.Errorf("cublasLtMatmulPreferenceCreate failed with status %d", status)
	}
	return &MatmulPreference{ptr: pref}, nil
}

// Destroy releases the matmul preference descriptor.
func (p *MatmulPreference) Destroy() error {
	lib, err := getCublasLtLib()
	if err != nil {
		return err
	}
	status := cuda.Ccall(lib.matmulPrefDestroy, p.ptr)
	if status != cublasLtStatusSuccess {
		return fmt.Errorf("cublasLtMatmulPreferenceDestroy failed with status %d", status)
	}
	return nil
}

// LtMatmulAlgoResult holds the result of a heuristic algorithm search.
// The raw bytes correspond to cublasLtMatmulHeuristicResult_t.
type LtMatmulAlgoResult struct {
	raw [sizeofLtHeuristicResult]byte
}

// AlgoPtr returns a pointer to the embedded cublasLtMatmulAlgo_t
// (the first 1024 bytes of the heuristic result).
func (r *LtMatmulAlgoResult) AlgoPtr() unsafe.Pointer {
	return unsafe.Pointer(&r.raw[0])
}

// MatmulAlgoGetHeuristic finds the best algorithm for the given matmul configuration.
// Returns up to requestedCount results. The actual number found is returned.
func MatmulAlgoGetHeuristic(
	h *LtHandle,
	desc *MatmulDesc,
	layoutA, layoutB, layoutC, layoutD *MatrixLayout,
	pref *MatmulPreference,
	requestedCount int,
) ([]LtMatmulAlgoResult, error) {
	lib, err := getCublasLtLib()
	if err != nil {
		return nil, err
	}

	results := make([]LtMatmulAlgoResult, requestedCount)
	var returnedCount int32
	status := cuda.Ccall(lib.matmulAlgoGetHeur,
		h.ptr,
		desc.ptr,
		layoutA.ptr,
		layoutB.ptr,
		layoutC.ptr,
		layoutD.ptr,
		pref.ptr,
		uintptr(requestedCount),
		uintptr(unsafe.Pointer(&results[0])),
		uintptr(unsafe.Pointer(&returnedCount)),
	)
	if status != cublasLtStatusSuccess {
		return nil, fmt.Errorf("cublasLtMatmulAlgoGetHeuristic failed with status %d", status)
	}
	return results[:returnedCount], nil
}

// LtMatmul performs a matrix multiplication using cublasLt.
// alpha and beta are pointers to host scalars of the scale type.
// stream is the CUDA stream handle (0 for default stream).
// workspace is optional device workspace memory (can be nil with workspaceSize=0).
func LtMatmul(
	h *LtHandle,
	desc *MatmulDesc,
	alpha unsafe.Pointer,
	a unsafe.Pointer, layoutA *MatrixLayout,
	b unsafe.Pointer, layoutB *MatrixLayout,
	beta unsafe.Pointer,
	c unsafe.Pointer, layoutC *MatrixLayout,
	d unsafe.Pointer, layoutD *MatrixLayout,
	algo *LtMatmulAlgoResult,
	workspace unsafe.Pointer, workspaceSize int,
	stream uintptr,
) error {
	lib, err := getCublasLtLib()
	if err != nil {
		return err
	}

	var algoPtr uintptr
	if algo != nil {
		algoPtr = uintptr(algo.AlgoPtr())
	}
	var wsPtr uintptr
	if workspace != nil {
		wsPtr = uintptr(workspace)
	}

	status := cuda.Ccall(lib.matmul,
		h.ptr,
		desc.ptr,
		uintptr(alpha),
		uintptr(a),
		layoutA.ptr,
		uintptr(b),
		layoutB.ptr,
		uintptr(beta),
		uintptr(c),
		layoutC.ptr,
		uintptr(d),
		layoutD.ptr,
		algoPtr,
		wsPtr,
		uintptr(workspaceSize),
		stream,
	)
	if status != cublasLtStatusSuccess {
		return fmt.Errorf("cublasLtMatmul failed with status %d", status)
	}
	return nil
}
