package serve

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/zerfoo/zerfoo/inference/guardian"
	"github.com/zerfoo/ztensor/metrics/runtime"
)

// maxGuardBatch is the maximum number of inputs in a single guard batch request.
const maxGuardBatch = 256

// GuardEvaluator abstracts a Guardian safety evaluator for testability.
type GuardEvaluator interface {
	Evaluate(ctx context.Context, req guardian.GuardianRequest) ([]guardian.Verdict, error)
	EvaluateBatch(ctx context.Context, inputs []guardian.GuardianInput, risks []string) (*guardian.BatchResult, error)
	Scan(ctx context.Context, input guardian.GuardianInput) (*guardian.ScanResult, error)
}

// WithGuardEvaluator sets the Guardian evaluator for the /v1/guard endpoints.
func WithGuardEvaluator(e GuardEvaluator) ServerOption {
	return func(s *Server) {
		s.guardEvaluator = e
	}
}

// GuardRequest is the request body for POST /v1/guard.
type GuardRequest struct {
	Model  string                 `json:"model"`
	Input  guardian.GuardianInput `json:"input"`
	Risks  []string               `json:"risks"`
	Format string                 `json:"format,omitempty"`
	Think  bool                   `json:"think,omitempty"`
}

// GuardResponse is the response body for POST /v1/guard.
type GuardResponse struct {
	Model     string        `json:"model"`
	Flagged   bool          `json:"flagged"`
	Verdicts  []VerdictData `json:"verdicts"`
	LatencyMs int64         `json:"latency_ms"`
}

// VerdictData holds a single risk verdict in the API response.
type VerdictData struct {
	Risk       string  `json:"risk"`
	Unsafe     bool    `json:"unsafe"`
	Confidence float64 `json:"confidence"`
	Reasoning  string  `json:"reasoning"`
}

// GuardBatchRequest is the request body for POST /v1/guard/batch.
type GuardBatchRequest struct {
	Model  string                   `json:"model"`
	Inputs []guardian.GuardianInput `json:"inputs"`
	Risks  []string                 `json:"risks"`
}

// GuardBatchResponse is the response body for POST /v1/guard/batch.
type GuardBatchResponse struct {
	Model     string             `json:"model"`
	Results   []GuardBatchResult `json:"results"`
	LatencyMs int64              `json:"latency_ms"`
}

// GuardBatchResult holds verdicts for a single input in a batch.
type GuardBatchResult struct {
	Index    int           `json:"index"`
	Flagged  bool          `json:"flagged"`
	Verdicts []VerdictData `json:"verdicts"`
}

// GuardScanRequest is the request body for POST /v1/guard/scan.
type GuardScanRequest struct {
	Model string                 `json:"model"`
	Input guardian.GuardianInput `json:"input"`
}

// GuardScanResponse is the response body for POST /v1/guard/scan.
type GuardScanResponse struct {
	Model       string        `json:"model"`
	Flagged     bool          `json:"flagged"`
	HighestRisk string        `json:"highest_risk,omitempty"`
	Verdicts    []VerdictData `json:"verdicts"`
	LatencyMs   int64         `json:"latency_ms"`
}

// GuardMetrics records Guardian-specific Prometheus metrics.
type GuardMetrics struct {
	requestsTotal runtime.CounterMetric
	latencyMs     runtime.HistogramMetric
}

// NewGuardMetrics creates Guardian metrics backed by the given collector.
func NewGuardMetrics(c runtime.Collector) *GuardMetrics {
	return &GuardMetrics{
		requestsTotal: c.Counter("guard_requests_total"),
		latencyMs:     c.Histogram("guard_latency_ms", latencyBuckets),
	}
}

func (s *Server) handleGuard(w http.ResponseWriter, r *http.Request) {
	s.modelMu.RLock()
	defer s.modelMu.RUnlock()
	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model not available")
		return
	}

	if s.guardEvaluator == nil {
		writeError(w, http.StatusNotImplemented, "guardian evaluation is not configured")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20)
	var req GuardRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		s.logger.Debug("invalid request body", "error", err.Error())
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}

	if req.Input.User == "" {
		writeError(w, http.StatusBadRequest, "input.user is required")
		return
	}

	// Validate risk categories.
	for _, risk := range req.Risks {
		if _, ok := guardian.RiskDefinitions[risk]; !ok {
			writeError(w, http.StatusBadRequest, "invalid risk category: "+risk)
			return
		}
	}

	start := time.Now()

	verdicts, err := s.guardEvaluator.Evaluate(r.Context(), guardian.GuardianRequest{
		Input:  req.Input,
		Risks:  req.Risks,
		Format: req.Format,
		Think:  req.Think,
	})
	if err != nil {
		writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
		return
	}

	flagged := false
	data := make([]VerdictData, len(verdicts))
	for i, v := range verdicts {
		if v.Unsafe {
			flagged = true
		}
		data[i] = VerdictData{
			Risk:       v.Risk,
			Unsafe:     v.Unsafe,
			Confidence: v.Confidence,
			Reasoning:  v.Reasoning,
		}
	}

	latency := time.Since(start).Milliseconds()

	s.guardMetrics.requestsTotal.Inc()
	s.guardMetrics.latencyMs.Observe(float64(latency))

	writeJSON(w, http.StatusOK, GuardResponse{
		Model:     req.Model,
		Flagged:   flagged,
		Verdicts:  data,
		LatencyMs: latency,
	})
}

func (s *Server) handleGuardBatch(w http.ResponseWriter, r *http.Request) {
	s.modelMu.RLock()
	defer s.modelMu.RUnlock()
	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model not available")
		return
	}

	if s.guardEvaluator == nil {
		writeError(w, http.StatusNotImplemented, "guardian evaluation is not configured")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20)
	var req GuardBatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		s.logger.Debug("invalid request body", "error", err.Error())
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}

	if len(req.Inputs) == 0 {
		writeError(w, http.StatusBadRequest, "inputs is required")
		return
	}

	if len(req.Inputs) > maxGuardBatch {
		writeError(w, http.StatusBadRequest, "inputs exceeds maximum batch size of 256")
		return
	}

	// Validate risk categories.
	for _, risk := range req.Risks {
		if _, ok := guardian.RiskDefinitions[risk]; !ok {
			writeError(w, http.StatusBadRequest, "invalid risk category: "+risk)
			return
		}
	}

	start := time.Now()

	result, err := s.guardEvaluator.EvaluateBatch(r.Context(), req.Inputs, req.Risks)
	if err != nil {
		writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
		return
	}

	results := make([]GuardBatchResult, len(result.Results))
	for i, ir := range result.Results {
		data := make([]VerdictData, len(ir.Verdicts))
		for j, v := range ir.Verdicts {
			data[j] = VerdictData{
				Risk:       v.Risk,
				Unsafe:     v.Unsafe,
				Confidence: v.Confidence,
				Reasoning:  v.Reasoning,
			}
		}
		results[i] = GuardBatchResult{
			Index:    ir.Index,
			Flagged:  ir.Flagged,
			Verdicts: data,
		}
	}

	latency := time.Since(start).Milliseconds()

	s.guardMetrics.requestsTotal.Inc()
	s.guardMetrics.latencyMs.Observe(float64(latency))

	writeJSON(w, http.StatusOK, GuardBatchResponse{
		Model:     req.Model,
		Results:   results,
		LatencyMs: latency,
	})
}

func (s *Server) handleGuardScan(w http.ResponseWriter, r *http.Request) {
	s.modelMu.RLock()
	defer s.modelMu.RUnlock()
	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model not available")
		return
	}

	if s.guardEvaluator == nil {
		writeError(w, http.StatusNotImplemented, "guardian evaluation is not configured")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20)
	var req GuardScanRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		s.logger.Debug("invalid request body", "error", err.Error())
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}

	if req.Input.User == "" {
		writeError(w, http.StatusBadRequest, "input.user is required")
		return
	}

	start := time.Now()

	result, err := s.guardEvaluator.Scan(r.Context(), req.Input)
	if err != nil {
		writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
		return
	}

	data := make([]VerdictData, len(result.Verdicts))
	for i, v := range result.Verdicts {
		data[i] = VerdictData{
			Risk:       v.Risk,
			Unsafe:     v.Unsafe,
			Confidence: v.Confidence,
			Reasoning:  v.Reasoning,
		}
	}

	latency := time.Since(start).Milliseconds()

	s.guardMetrics.requestsTotal.Inc()
	s.guardMetrics.latencyMs.Observe(float64(latency))

	writeJSON(w, http.StatusOK, GuardScanResponse{
		Model:       req.Model,
		Flagged:     result.Flagged,
		HighestRisk: result.HighestRisk,
		Verdicts:    data,
		LatencyMs:   latency,
	})
}
