// Package health provides HTTP health check endpoints for Kubernetes-style
// liveness and readiness probes.
//
// NewServer creates a Server with /healthz (liveness) and /readyz (readiness)
// endpoints. Readiness checks are configurable via AddReadinessCheck.
package health

import (
	"encoding/json"
	"net/http"
	"sync"

	"github.com/zerfoo/zerfoo/log"
)

// CheckFunc is a function that returns nil when healthy or an error when not.
type CheckFunc func() error

// Response is the JSON body returned by health endpoints.
type Response struct {
	Status string            `json:"status"`
	Checks map[string]string `json:"checks,omitempty"`
}

// Server provides HTTP health check endpoints.
type Server struct {
	logger log.Logger
	mu     sync.RWMutex
	checks map[string]CheckFunc
}

// NewServer creates a new health check server using the given logger.
func NewServer(logger log.Logger) *Server {
	return &Server{
		logger: logger,
		checks: make(map[string]CheckFunc),
	}
}

// AddReadinessCheck registers a named readiness check. All registered checks
// must pass for the /readyz endpoint to return 200 OK.
func (s *Server) AddReadinessCheck(name string, check CheckFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.checks[name] = check
}

// Handler returns an http.Handler with /healthz and /readyz routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealthz)
	mux.HandleFunc("/readyz", s.handleReadyz)
	return mux
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	writeJSON(w, http.StatusOK, Response{Status: "ok"})
}

func (s *Server) handleReadyz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.mu.RLock()
	checks := make(map[string]CheckFunc, len(s.checks))
	for name, fn := range s.checks {
		checks[name] = fn
	}
	s.mu.RUnlock()

	results := make(map[string]string, len(checks))
	healthy := true

	for name, fn := range checks {
		if err := fn(); err != nil {
			results[name] = err.Error()
			healthy = false
			s.logger.Warn("readiness check failed", "check", name, "error", err.Error())
		} else {
			results[name] = "ok"
		}
	}

	if healthy {
		resp := Response{Status: "ok"}
		if len(results) > 0 {
			resp.Checks = results
		}
		writeJSON(w, http.StatusOK, resp)
		return
	}

	writeJSON(w, http.StatusServiceUnavailable, Response{
		Status: "unavailable",
		Checks: results,
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
