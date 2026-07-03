package serve

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/zerfoo/zerfoo/inference/sentiment"
	"github.com/zerfoo/ztensor/metrics/runtime"
)

// maxClassifyBatch is the maximum number of texts in a single classify request.
const maxClassifyBatch = 256

// Classifier abstracts a text classification pipeline for testability.
type Classifier interface {
	Classify(ctx context.Context, texts []string) ([]sentiment.SentimentResult, error)
}

// WithClassifier sets the text classification pipeline for the
// /v1/classify endpoint.
func WithClassifier(c Classifier) ServerOption {
	return func(s *Server) {
		s.classifier = c
	}
}

// ClassifyRequest is the request body for POST /v1/classify.
type ClassifyRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// ClassifyResponse is the response body for POST /v1/classify.
type ClassifyResponse struct {
	Data  []ClassifyData `json:"data"`
	Model string         `json:"model"`
	Usage ClassifyUsage  `json:"usage"`
}

// ClassifyData holds classification output for a single input text.
type ClassifyData struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
	Index int     `json:"index"`
}

// ClassifyUsage reports token counts for the classify request.
type ClassifyUsage struct {
	TotalTokens int `json:"total_tokens"`
}

// ClassifyMetrics records classification-specific Prometheus metrics.
type ClassifyMetrics struct {
	requestsTotal runtime.CounterMetric
	latencyMs     runtime.HistogramMetric
}

// NewClassifyMetrics creates classification metrics backed by the given collector.
func NewClassifyMetrics(c runtime.Collector) *ClassifyMetrics {
	return &ClassifyMetrics{
		requestsTotal: c.Counter("classify_requests_total"),
		latencyMs:     c.Histogram("classify_latency_ms", latencyBuckets),
	}
}

func (s *Server) handleClassify(w http.ResponseWriter, r *http.Request) {
	s.modelMu.RLock()
	defer s.modelMu.RUnlock()
	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model not available")
		return
	}

	if s.classifier == nil {
		writeError(w, http.StatusNotImplemented, "text classification is not configured")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB
	var req ClassifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		s.logger.Debug("invalid request body", "error", err.Error())
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if len(req.Input) == 0 {
		writeError(w, http.StatusBadRequest, "input is required")
		return
	}

	if len(req.Input) > maxClassifyBatch {
		writeError(w, http.StatusBadRequest, "input exceeds maximum batch size of 256")
		return
	}

	start := time.Now()

	results, err := s.classifier.Classify(r.Context(), req.Input)
	if err != nil {
		writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
		return
	}

	// Count tokens if model tokenizer is available.
	var totalTokens int
	if tok := s.model.Tokenizer(); tok != nil {
		for _, text := range req.Input {
			if ids, err := tok.Encode(text); err == nil {
				totalTokens += len(ids)
			}
		}
	}

	data := make([]ClassifyData, len(results))
	for i, res := range results {
		data[i] = ClassifyData{
			Label: res.Label,
			Score: res.Score,
			Index: i,
		}
	}

	modelID := req.Model
	if modelID == "" {
		if info := s.model.Info(); info != nil {
			modelID = info.ID
		}
	}

	s.classifyMetrics.requestsTotal.Inc()
	s.classifyMetrics.latencyMs.Observe(float64(time.Since(start).Microseconds()) / 1000.0)

	writeJSON(w, http.StatusOK, ClassifyResponse{
		Data:  data,
		Model: modelID,
		Usage: ClassifyUsage{TotalTokens: totalTokens},
	})
}
