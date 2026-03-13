package cublas

import "testing"

func TestGemmExCallable(t *testing.T) {
	tests := []struct {
		name        string
		aType       CudaDataType
		bType       CudaDataType
		cType       CudaDataType
		computeType CublasComputeType
	}{
		{
			name:        "FP32",
			aType:       CudaR32F,
			bType:       CudaR32F,
			cType:       CudaR32F,
			computeType: CublasCompute32F,
		},
		{
			name:        "BF16",
			aType:       CudaR16BF,
			bType:       CudaR16BF,
			cType:       CudaR16BF,
			computeType: CublasCompute32F,
		},
		{
			name:        "FP16",
			aType:       CudaR16F,
			bType:       CudaR16F,
			cType:       CudaR16F,
			computeType: CublasCompute32F,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// GemmEx should return an error since cuBLAS library is not available
			// in this test environment (no GPU). This verifies the wrapper compiles
			// and is callable with the correct signature.
			h := &Handle{}
			err := GemmEx(h, 2, 2, 2, 1.0,
				nil, tc.aType,
				nil, tc.bType,
				0.0,
				nil, tc.cType,
				tc.computeType,
			)
			if err == nil {
				t.Fatal("expected error from GemmEx without cuBLAS library, got nil")
			}
		})
	}
}

func TestCublasGemmDefaultValue(t *testing.T) {
	// CUBLAS_GEMM_DEFAULT is -1 in C, which is 0xFFFFFFFF as uint32.
	if cublasGemmDefault != 0xFFFFFFFF {
		t.Errorf("cublasGemmDefault = %#x, want 0xFFFFFFFF", cublasGemmDefault)
	}
}
