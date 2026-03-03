package health

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/log"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestEngineCheck_Healthy(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	check := EngineCheck(engine, time.Second)

	err := check()
	if err != nil {
		t.Fatalf("expected healthy engine check, got: %v", err)
	}
}

func TestEngineCheck_RegisteredAsReadiness(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	srv := NewServer(log.Nop())
	srv.AddReadinessCheck("engine", EngineCheck(engine, time.Second))

	handler := srv.Handler()
	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("readyz status = %d, want 200", rec.Code)
	}
}

func TestEngineCheckGeneric_Healthy(t *testing.T) {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	check := EngineCheckGeneric(engine, numeric.Float64Ops{}, time.Second)

	err := check()
	if err != nil {
		t.Fatalf("expected healthy engine check, got: %v", err)
	}
}

func TestPprofEndpoints(t *testing.T) {
	srv := NewServer(log.Nop())
	handler := srv.Handler()

	endpoints := []string{
		"/debug/pprof/",
		"/debug/pprof/cmdline",
		"/debug/pprof/symbol",
	}

	for _, ep := range endpoints {
		req := httptest.NewRequest(http.MethodGet, ep, nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("%s status = %d, want 200", ep, rec.Code)
		}
	}
}

func TestPprofIndex_ContainsProfiles(t *testing.T) {
	srv := NewServer(log.Nop())
	handler := srv.Handler()

	req := httptest.NewRequest(http.MethodGet, "/debug/pprof/", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	body := rec.Body.String()
	if !strings.Contains(body, "heap") {
		t.Error("pprof index should contain 'heap' profile")
	}
	if !strings.Contains(body, "goroutine") {
		t.Error("pprof index should contain 'goroutine' profile")
	}
}
