package gpuapi

// BLASFactory creates a BLAS instance. Registered by cuda_blas_purego.go via init()
// when the cuBLAS library is available at runtime.
var BLASFactory func() (BLAS, error)

// DNNFactory creates a DNN instance. Registered by cuda_dnn.go via init().
var DNNFactory func() (DNN, error)
