package integration

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/config"
	"github.com/zerfoo/zerfoo/health"
	"github.com/zerfoo/zerfoo/log"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/shutdown"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestProductionSmokeTest exercises the full lifecycle:
// config loading -> engine creation -> health check -> operation -> graceful shutdown.
func TestProductionSmokeTest(t *testing.T) {
	// 1. Write a temporary config file.
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "engine.json")
	cfgData, err := json.Marshal(config.EngineConfig{
		Device:        "cpu",
		MemoryLimitMB: 100,
		LogLevel:      "info",
	})
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(cfgPath, cfgData, 0600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	// 2. Load config from file.
	cfg, err := config.Load[config.EngineConfig](cfgPath)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if cfg.Device != "cpu" {
		t.Fatalf("device = %s, want cpu", cfg.Device)
	}

	// 3. Create engine with memory limit.
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	if cfg.MemoryLimitMB > 0 {
		engine.SetMemoryLimit(int64(cfg.MemoryLimitMB) * 1024 * 1024)
	}

	// 4. Register engine with shutdown coordinator.
	coord := shutdown.New()
	coord.Register(engine)

	// 5. Set up health check server with engine check.
	hs := health.NewServer(log.Nop())
	hs.AddReadinessCheck("engine", health.EngineCheck(engine, 5*time.Second))

	// Verify health endpoint responses.
	handler := hs.Handler()
	{
		req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("healthz: status = %d, want 200", w.Code)
		}
	}
	{
		req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("readyz: status = %d, want 200", w.Code)
		}
	}

	// 6. Run a tensor operation (forward pass simulation).
	ctx := context.Background()
	a, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("new tensor a: %v", err)
	}
	b, err := tensor.New[float32]([]int{3, 2}, []float32{7, 8, 9, 10, 11, 12})
	if err != nil {
		t.Fatalf("new tensor b: %v", err)
	}
	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}
	if len(result.Data()) != 4 {
		t.Errorf("result size = %d, want 4", len(result.Data()))
	}

	// 7. Trigger graceful shutdown.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	errs := coord.Shutdown(shutdownCtx)
	if len(errs) != 0 {
		t.Errorf("shutdown errors: %v", errs)
	}
}
