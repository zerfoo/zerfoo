package serve

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/zerfoo/zerfoo/inference/sentiment"
	"github.com/zerfoo/ztensor/metrics/runtime"
)

// maxSentimentBatch is the maximum number of texts in a single sentiment request.
const maxSentimentBatch = 256

// SentimentClassifier abstracts the sentiment pipeline for testability.
type SentimentClassifier interface {
	Classify(ctx context.Context, texts []string) ([]sentiment.SentimentResult, error)
}

// WithSentiment sets the sentiment classification pipeline for the
// /v1/sentiment endpoint.
func WithSentiment(sc SentimentClassifier) ServerOption {
	return func(s *Server) {
		s.sentiment = sc
	}
}

// SentimentRequest is the request body for POST /v1/sentiment.
type SentimentRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// SentimentResponse is the response body for POST /v1/sentiment.
type SentimentResponse struct {
	Data  []SentimentData `json:"data"`
	Model string          `json:"model"`
	Usage SentimentUsage  `json:"usage"`
}

// SentimentData holds classification output for a single input text.
type SentimentData struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
	Index int     `json:"index"`
}

// SentimentUsage reports token counts for the sentiment request.
type SentimentUsage struct {
	TotalTokens int `json:"total_tokens"`
}

// SentimentMetrics records sentiment-specific Prometheus metrics.
type SentimentMetrics struct {
	requestsTotal runtime.CounterMetric
	latencyMs     runtime.HistogramMetric
}

// NewSentimentMetrics creates sentiment metrics backed by the given collector.
func NewSentimentMetrics(c runtime.Collector) *SentimentMetrics {
	return &SentimentMetrics{
		requestsTotal: c.Counter("sentiment_requests_total"),
		latencyMs:     c.Histogram("sentiment_latency_ms", latencyBuckets),
	}
}

func (s *Server) handleSentiment(w http.ResponseWriter, r *http.Request) {
	if s.sentiment == nil {
		writeError(w, http.StatusNotImplemented, "sentiment classification is not configured")
		return
	}

	var req SentimentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if len(req.Input) == 0 {
		writeError(w, http.StatusBadRequest, "input is required")
		return
	}

	if len(req.Input) > maxSentimentBatch {
		writeError(w, http.StatusBadRequest, "input exceeds maximum batch size of 256")
		return
	}

	start := time.Now()

	results, err := s.sentiment.Classify(r.Context(), req.Input)
	if err != nil {
		writeError(w, inferenceErrorStatus(err), err.Error())
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

	data := make([]SentimentData, len(results))
	for i, res := range results {
		data[i] = SentimentData{
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

	s.sentimentMetrics.requestsTotal.Inc()
	s.sentimentMetrics.latencyMs.Observe(float64(time.Since(start).Microseconds()) / 1000.0)

	writeJSON(w, http.StatusOK, SentimentResponse{
		Data:  data,
		Model: modelID,
		Usage: SentimentUsage{TotalTokens: totalTokens},
	})
}
