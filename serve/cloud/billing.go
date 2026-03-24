package cloud

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/zerfoo/zerfoo/generate"
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

// maxBillingCaptureSize limits the response body captured for billing fallback.
const maxBillingCaptureSize = 64 * 1024

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
	if rc.body.Len() < maxBillingCaptureSize {
		remaining := maxBillingCaptureSize - rc.body.Len()
		if len(b) > remaining {
			rc.body.Write(b[:remaining])
		} else {
			rc.body.Write(b)
		}
	}
	return rc.ResponseWriter.Write(b)
}

// BillingMiddleware returns an HTTP middleware that meters prompt and completion
// tokens per request and publishes usage events to the given recorder.
// It expects the TenantRegistry middleware to run first so that TenantFromContext
// returns a valid tenant. The tenant ID is taken from the apiKeyHeader value.
//
// For streaming (SSE) responses, the JSON response body cannot be parsed as a
// single object. The middleware injects a [generate.TokenUsage] into the request
// context; the generation session writes prompt/completion counts there, which
// works for both streaming and non-streaming responses. JSON body parsing is
// used as a fallback for handlers that do not use context-based usage tracking.
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
				io.Copy(&reqBody, r.Body) //nolint:errcheck
				r.Body.Close()
			}
			r.Body = io.NopCloser(bytes.NewReader(reqBody.Bytes()))

			var req chatRequest
			json.Unmarshal(reqBody.Bytes(), &req) //nolint:errcheck

			// Inject TokenUsage into context for the generation layer.
			usage := &generate.TokenUsage{}
			ctx := generate.WithTokenUsage(r.Context(), usage)
			r = r.WithContext(ctx)

			// Capture the response body for token counting.
			capture := &responseCapture{ResponseWriter: w, statusCode: http.StatusOK}
			next.ServeHTTP(capture, r)

			// Prefer context-based usage (works for both streaming and non-streaming).
			promptTokens := usage.PromptTokens()
			completionTokens := usage.CompletionTokens()

			// Fall back to JSON body parsing for handlers that don't use context-based usage.
			if promptTokens == 0 && completionTokens == 0 {
				var resp usageResponse
				if err := json.Unmarshal(capture.body.Bytes(), &resp); err == nil {
					promptTokens = resp.Usage.PromptTokens
					completionTokens = resp.Usage.CompletionTokens
				}
			}

			if promptTokens == 0 && completionTokens == 0 {
				return
			}

			tenantID := extractBearerToken(r)
			event := UsageEvent{
				TenantID:         tenantID,
				Model:            req.Model,
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
				Timestamp:        time.Now().Unix(),
			}
			recorder.Record(event) //nolint:errcheck
		})
	}
}
