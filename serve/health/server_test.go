package health

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/zerfoo/ztensor/log"
)

func TestHealthz_ReturnsOK(t *testing.T) {
	s := NewServer(log.Nop())

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
	}

	var body Response
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.Status != "ok" {
		t.Errorf("status = %q, want %q", body.Status, "ok")
	}
}

func TestReadyz_NoChecks_ReturnsOK(t *testing.T) {
	s := NewServer(log.Nop())

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
	}
}

func TestReadyz_AllChecksPass(t *testing.T) {
	s := NewServer(log.Nop())
	s.AddReadinessCheck("db", func() error { return nil })
	s.AddReadinessCheck("cache", func() error { return nil })

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
	}

	var body Response
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.Status != "ok" {
		t.Errorf("status = %q, want %q", body.Status, "ok")
	}
}

func TestReadyz_CheckFails_Returns503(t *testing.T) {
	s := NewServer(log.Nop())
	s.AddReadinessCheck("ok-check", func() error { return nil })
	s.AddReadinessCheck("bad-check", func() error {
		return errUnhealthy
	})

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
	}

	var body Response
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.Status != "unavailable" {
		t.Errorf("status = %q, want %q", body.Status, "unavailable")
	}

	// Should report the failing check.
	if body.Checks == nil {
		t.Fatal("expected checks map in response")
	}
	if body.Checks["bad-check"] != "unhealthy" {
		t.Errorf("bad-check = %q, want %q", body.Checks["bad-check"], "unhealthy")
	}
	if body.Checks["ok-check"] != "ok" {
		t.Errorf("ok-check = %q, want %q", body.Checks["ok-check"], "ok")
	}
}

func TestReadyz_ConcurrentAccess(t *testing.T) {
	s := NewServer(log.Nop())
	s.AddReadinessCheck("concurrent", func() error { return nil })

	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
			}
		}()
	}
	wg.Wait()
}

func TestHealthz_MethodNotAllowed(t *testing.T) {
	s := NewServer(log.Nop())

	req := httptest.NewRequest(http.MethodPost, "/healthz", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}

func TestReadyz_MethodNotAllowed(t *testing.T) {
	s := NewServer(log.Nop())

	req := httptest.NewRequest(http.MethodPut, "/readyz", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}

func TestPprof_DisabledByDefault(t *testing.T) {
	s := NewServer(log.Nop())

	paths := []string{
		"/debug/pprof/",
		"/debug/pprof/cmdline",
		"/debug/pprof/symbol",
	}
	for _, p := range paths {
		req := httptest.NewRequest(http.MethodGet, p, nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("GET %s: status = %d, want %d", p, rec.Code, http.StatusNotFound)
		}
	}
}

func TestPprof_EnabledWithOption(t *testing.T) {
	s := NewServer(log.Nop(), WithPprof())

	req := httptest.NewRequest(http.MethodGet, "/debug/pprof/", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("GET /debug/pprof/: status = %d, want %d", rec.Code, http.StatusOK)
	}
}

var errUnhealthy = errors.New("unhealthy")
