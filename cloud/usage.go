package cloud

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
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

// BillingMiddleware returns an HTTP middleware that meters prompt and completion
// tokens per request and publishes usage events to the given recorder.
// It expects the tenant authentication middleware to run first so that
// tenantFromContext returns a valid tenant. The tenant ID is taken from
// the Authorization header's Bearer token value.
//
// For streaming (SSE) responses, the JSON response body cannot be parsed as a
// single object. The middleware injects a [generate.TokenUsage] into the request
// context; the generation session writes prompt/completion counts there, which
// works for both streaming and non-streaming responses. JSON body parsing is
// used as a fallback for handlers that do not use context-based usage tracking.
func BillingMiddleware(recorder UsageRecorder) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tenant := tenantFromContext(r.Context())
			if tenant == nil {
				http.Error(w, `{"error":{"message":"billing: tenant context required"}}`, http.StatusInternalServerError)
				return
			}

			// Read and buffer the request body to extract the model name.
			var reqBody bytes.Buffer
			if r.Body != nil {
				io.Copy(&reqBody, io.LimitReader(r.Body, 10<<20)) //nolint:errcheck
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
			capture := &billingResponseCapture{ResponseWriter: w, statusCode: http.StatusOK}
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

			raw := extractBearerToken(r)
			h := sha256.Sum256([]byte(raw))
			tenantID := hex.EncodeToString(h[:])
			event := UsageEvent{
				TenantID:         tenantID,
				Model:            req.Model,
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
				Timestamp:        time.Now().Unix(),
			}
			if err := recorder.Record(event); err != nil {
				fmt.Fprintf(os.Stderr, "billing: failed to record usage event tenant=%s model=%s prompt=%d completion=%d: %v\n",
					tenantID, req.Model, promptTokens, completionTokens, err)
			}
		})
	}
}

// maxBillingCaptureSize limits the response body captured for billing fallback.
const maxBillingCaptureSize = 64 * 1024

// billingResponseCapture wraps http.ResponseWriter to capture the response body
// for the standalone BillingMiddleware. This is separate from responseCapture in
// server.go to avoid type conflicts (different buffer types).
type billingResponseCapture struct {
	http.ResponseWriter
	body       bytes.Buffer
	statusCode int
}

func (rc *billingResponseCapture) WriteHeader(code int) {
	rc.statusCode = code
	rc.ResponseWriter.WriteHeader(code)
}

func (rc *billingResponseCapture) Write(b []byte) (int, error) {
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
