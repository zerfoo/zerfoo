package gpuapi

// BLASFactory creates a BLAS instance. Registered by cuda_blas.go via init().
// nil when not compiled with -tags cuda.
var BLASFactory func() (BLAS, error)

// DNNFactory creates a DNN instance. Registered by cuda_dnn.go via init().
// nil when not compiled with -tags cuda.
var DNNFactory func() (DNN, error)
