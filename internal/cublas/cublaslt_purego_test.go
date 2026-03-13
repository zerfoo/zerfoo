package cublas

import "testing"

func TestLtAvailable(t *testing.T) {
	// LtAvailable should not panic regardless of GPU presence.
	avail := LtAvailable()
	t.Logf("cublasLt available: %v", avail)
}

func TestLtCreateDestroyHandle(t *testing.T) {
	if !LtAvailable() {
		t.Skip("cublasLt not available (no GPU)")
	}
	h, err := LtCreateHandle()
	if err != nil {
		t.Fatalf("LtCreateHandle: %v", err)
	}
	if err := h.Destroy(); err != nil {
		t.Fatalf("LtHandle.Destroy: %v", err)
	}
}

func TestLtMatmulDescLifecycle(t *testing.T) {
	if !LtAvailable() {
		t.Skip("cublasLt not available (no GPU)")
	}
	desc, err := CreateMatmulDesc(LtComputeF32, CudaR32F)
	if err != nil {
		t.Fatalf("CreateMatmulDesc: %v", err)
	}
	if err := desc.Destroy(); err != nil {
		t.Fatalf("MatmulDesc.Destroy: %v", err)
	}
}

func TestLtMatrixLayoutLifecycle(t *testing.T) {
	if !LtAvailable() {
		t.Skip("cublasLt not available (no GPU)")
	}
	layout, err := CreateMatrixLayout(CudaR32F, 64, 64, 64)
	if err != nil {
		t.Fatalf("CreateMatrixLayout: %v", err)
	}
	if err := layout.Destroy(); err != nil {
		t.Fatalf("MatrixLayout.Destroy: %v", err)
	}
}

func TestLtMatmulPreferenceLifecycle(t *testing.T) {
	if !LtAvailable() {
		t.Skip("cublasLt not available (no GPU)")
	}
	pref, err := CreateMatmulPreference()
	if err != nil {
		t.Fatalf("CreateMatmulPreference: %v", err)
	}
	if err := pref.Destroy(); err != nil {
		t.Fatalf("MatmulPreference.Destroy: %v", err)
	}
}
