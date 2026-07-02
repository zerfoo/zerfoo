package cloud

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/zerfoo/zerfoo/generate"
)

// tenantKey is the private context key for storing the authenticated Tenant.
type tenantKey struct{}

// tenantFromContext returns the Tenant stored in ctx by authMiddleware,
// or nil if no tenant is present.
func tenantFromContext(ctx context.Context) *Tenant {
	t, _ := ctx.Value(tenantKey{}).(*Tenant)
	return t
}

// defaultMaxTokens is the estimated token budget consumed when a request
// does not specify max_tokens. It acts as a conservative upper bound for
// pre-authorization so that budget checks happen before inference runs.
const defaultMaxTokens = 4096

// CloudServer wraps an HTTP handler with multi-tenant isolation, token billing,
// rate limiting, and health checking for cloud deployments.
type CloudServer struct {
	tenants   *TenantManager
	meter     *TokenMeter
	inner     http.Handler
	maxTokens int64
	healthy   atomic.Bool
}

// NewCloudServer creates a CloudServer that routes authenticated requests
// to the given handler through tenant isolation middleware.
func NewCloudServer(handler http.Handler, tenants *TenantManager, meter *TokenMeter) *CloudServer {
	cs := &CloudServer{
		tenants:   tenants,
		meter:     meter,
		inner:     handler,
		maxTokens: defaultMaxTokens,
	}
	cs.healthy.Store(true)
	return cs
}

// Tenants returns the TenantManager for external CRUD operations.
func (cs *CloudServer) Tenants() *TenantManager {
	return cs.tenants
}

// Meter returns the TokenMeter for external billing queries.
func (cs *CloudServer) Meter() *TokenMeter {
	return cs.meter
}

// SetHealthy sets the health status of the cloud server.
func (cs *CloudServer) SetHealthy(healthy bool) {
	cs.healthy.Store(healthy)
}

// Handler returns the root HTTP handler with all middleware applied.
func (cs *CloudServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", cs.handleHealth)
	mux.Handle("/", cs.authMiddleware(cs.rateLimitMiddleware(cs.billingMiddleware(cs.inner))))
	return mux
}

// handleHealth returns 200 when healthy, 503 when degraded.
func (cs *CloudServer) handleHealth(w http.ResponseWriter, _ *http.Request) {
	if !cs.healthy.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"status":"degraded"}`)) //nolint:errcheck
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`)) //nolint:errcheck
}

// authMiddleware extracts the Bearer token, resolves the tenant, and stores
// it in the request context for downstream middleware.
func (cs *CloudServer) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiKey := extractBearerToken(r)
		if apiKey == "" {
			writeError(w, http.StatusUnauthorized, "missing or invalid authorization header")
			return
		}

		tenant, err := cs.tenants.GetByAPIKey(apiKey)
		if err != nil {
			writeError(w, http.StatusUnauthorized, "invalid API key")
			return
		}

		// Pass tenant to downstream middleware via context.
		ctx := context.WithValue(r.Context(), tenantKey{}, tenant)
		r = r.WithContext(ctx)
		next.ServeHTTP(w, r)
	})
}

// rateLimitMiddleware enforces per-tenant request rate limits and token budgets.
func (cs *CloudServer) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenant := tenantFromContext(r.Context())
		if tenant == nil {
			writeError(w, http.StatusInternalServerError, "tenant lookup failed")
			return
		}

		if !tenant.AllowRequest() {
			w.Header().Set("Retry-After", "60")
			writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
			return
		}

		next.ServeHTTP(w, r)
	})
}

// billingMiddleware pre-authorizes token budgets before inference runs
// and reconciles actual usage afterward. For streaming (SSE) responses,
// JSON parsing fails silently, so the middleware also checks for token
// counts stored in the request context by the generation session via
// [generate.TokenUsage].
func (cs *CloudServer) billingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenant := tenantFromContext(r.Context())

		// Parse max_tokens from the request body to estimate the token budget
		// needed. Default to cs.maxTokens when not specified or unparseable.
		estimated := cs.maxTokens
		if r.Body != nil {
			bodyBytes, err := io.ReadAll(io.LimitReader(r.Body, 10<<20))
			r.Body.Close()
			if err == nil && len(bodyBytes) > 0 {
				var req struct {
					MaxTokens *int64 `json:"max_tokens"`
				}
				if json.Unmarshal(bodyBytes, &req) == nil && req.MaxTokens != nil && *req.MaxTokens > 0 {
					estimated = *req.MaxTokens
				}
				// Restore the body so downstream handlers can read it.
				r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			}
		}

		// Pre-authorize: reject before inference if budget is exhausted.
		if tenant != nil && !tenant.ConsumeTokens(estimated) {
			w.Header().Set("Retry-After", "60")
			writeError(w, http.StatusTooManyRequests, "token budget exhausted")
			return
		}

		// Inject a TokenUsage into the context so the generation layer can
		// record prompt/completion counts regardless of response format.
		usage := &generate.TokenUsage{}
		ctx := generate.WithTokenUsage(r.Context(), usage)
		r = r.WithContext(ctx)

		capture := &responseCapture{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(capture, r)

		// Prefer context-based usage (works for both streaming and non-streaming).
		input := usage.PromptTokens()
		output := usage.CompletionTokens()

		// Fall back to JSON body parsing for handlers that don't use context-based usage.
		if input == 0 && output == 0 {
			var resp struct {
				Usage struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(capture.body, &resp); err == nil {
				input = resp.Usage.PromptTokens
				output = resp.Usage.CompletionTokens
			}
		}

		// Reconcile: return unused tokens from the pre-authorized estimate,
		// or deduct excess when actual usage exceeds the estimate (e.g. a
		// client sends max_tokens=1 but the model generates more tokens).
		actual := int64(input + output)
		if tenant != nil {
			if actual < estimated {
				tenant.RefundTokens(estimated - actual)
			} else if actual > estimated {
				tenant.DeductTokens(actual - estimated)
			}
		}

		// Record billing.
		if cs.meter != nil && tenant != nil && actual > 0 {
			cs.meter.Record(tenant.ID, input, output)
		}
	})
}

// maxCaptureSize limits the response body captured for billing fallback parsing.
// Context-based token usage is the primary path; body capture is only used when
// the context values are zero. 64 KB is enough for any non-streaming JSON response.
const maxCaptureSize = 64 * 1024

// responseCapture wraps http.ResponseWriter to capture the response body.
type responseCapture struct {
	http.ResponseWriter
	body       []byte
	statusCode int
}

func (rc *responseCapture) WriteHeader(code int) {
	rc.statusCode = code
	rc.ResponseWriter.WriteHeader(code)
}

func (rc *responseCapture) Write(b []byte) (int, error) {
	if len(rc.body) < maxCaptureSize {
		remaining := maxCaptureSize - len(rc.body)
		if len(b) > remaining {
			rc.body = append(rc.body, b[:remaining]...)
		} else {
			rc.body = append(rc.body, b...)
		}
	}
	return rc.ResponseWriter.Write(b)
}

// Flush delegates to the wrapped ResponseWriter if it implements http.Flusher.
// This is required for SSE streaming through the cloud middleware chain.
func (rc *responseCapture) Flush() {
	if f, ok := rc.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func extractBearerToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return ""
	}
	const prefix = "Bearer "
	if !strings.HasPrefix(auth, prefix) {
		return ""
	}
	return auth[len(prefix):]
}

func writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{"message": message},
	}) //nolint:errcheck
}

// retryAfterSeconds returns the Retry-After header value in seconds,
// or 0 if the header is not present or unparseable.
func retryAfterSeconds(h http.Header) int {
	v := h.Get("Retry-After")
	if v == "" {
		return 0
	}

	// Try seconds first.
	if secs, err := strconv.Atoi(v); err == nil {
		return secs
	}

	// Try HTTP-date format.
	if t, err := time.Parse(time.RFC1123, v); err == nil {
		d := time.Until(t)
		if d < 0 {
			return 0
		}
		return int(d.Seconds()) + 1
	}
	return 0
}
