package cloud

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"sync"
	"time"
)

// UsageEvent records token consumption for a single request.
type UsageEvent struct {
	TenantID         string `json:"tenant_id"`
	Model            string `json:"model"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	Timestamp        int64  `json:"timestamp"`
}

// UsageRecorder defines the interface for recording usage events.
// The default implementation writes NDJSON; a Kafka adapter can implement
// this interface for production deployments.
type UsageRecorder interface {
	Record(event UsageEvent) error
}

// NDJSONRecorder writes usage events as newline-delimited JSON to an io.Writer.
type NDJSONRecorder struct {
	mu sync.Mutex
	w  io.Writer
}

// NewNDJSONRecorder creates a recorder that writes NDJSON to w.
func NewNDJSONRecorder(w io.Writer) *NDJSONRecorder {
	return &NDJSONRecorder{w: w}
}

// Record serializes the event as a single JSON line followed by a newline.
func (r *NDJSONRecorder) Record(event UsageEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	data = append(data, '\n')

	r.mu.Lock()
	defer r.mu.Unlock()
	_, err = r.w.Write(data)
	return err
}

// chatRequest is a minimal struct for extracting model from the request body.
type chatRequest struct {
	Model string `json:"model"`
}

// usageResponse is a minimal struct for extracting usage from the response body.
type usageResponse struct {
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// responseCapture wraps http.ResponseWriter to capture the response body.
type responseCapture struct {
	http.ResponseWriter
	body       bytes.Buffer
	statusCode int
}

func (rc *responseCapture) WriteHeader(code int) {
	rc.statusCode = code
	rc.ResponseWriter.WriteHeader(code)
}

func (rc *responseCapture) Write(b []byte) (int, error) {
	rc.body.Write(b)
	return rc.ResponseWriter.Write(b)
}

// BillingMiddleware returns an HTTP middleware that meters prompt and completion
// tokens per request and publishes usage events to the given recorder.
// It expects the TenantRegistry middleware to run first so that TenantFromContext
// returns a valid tenant. The tenant ID is taken from the apiKeyHeader value.
func BillingMiddleware(recorder UsageRecorder) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tenant := TenantFromContext(r.Context())
			if tenant == nil {
				// No tenant in context — pass through without metering.
				next.ServeHTTP(w, r)
				return
			}

			// Read and buffer the request body to extract the model name.
			var reqBody bytes.Buffer
			if r.Body != nil {
				io.Copy(&reqBody, r.Body)
				r.Body.Close()
			}
			r.Body = io.NopCloser(bytes.NewReader(reqBody.Bytes()))

			var req chatRequest
			json.Unmarshal(reqBody.Bytes(), &req)

			// Capture the response body for token counting.
			capture := &responseCapture{ResponseWriter: w, statusCode: http.StatusOK}
			next.ServeHTTP(capture, r)

			// Extract usage from the response.
			var resp usageResponse
			if err := json.Unmarshal(capture.body.Bytes(), &resp); err != nil {
				return
			}

			if resp.Usage.PromptTokens == 0 && resp.Usage.CompletionTokens == 0 {
				return
			}

			tenantID := extractBearerToken(r)
			event := UsageEvent{
				TenantID:         tenantID,
				Model:            req.Model,
				PromptTokens:     resp.Usage.PromptTokens,
				CompletionTokens: resp.Usage.CompletionTokens,
				Timestamp:        time.Now().Unix(),
			}
			recorder.Record(event)
		})
	}
}
