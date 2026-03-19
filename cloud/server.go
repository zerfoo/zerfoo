package cloud

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// CloudServer wraps an HTTP handler with multi-tenant isolation, token billing,
// rate limiting, and health checking for cloud deployments.
type CloudServer struct {
	tenants *TenantManager
	meter   *TokenMeter
	inner   http.Handler
	healthy atomic.Bool
}

// NewCloudServer creates a CloudServer that routes authenticated requests
// to the given handler through tenant isolation middleware.
func NewCloudServer(handler http.Handler, tenants *TenantManager, meter *TokenMeter) *CloudServer {
	cs := &CloudServer{
		tenants: tenants,
		meter:   meter,
		inner:   handler,
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

// authMiddleware extracts the Bearer token, resolves the tenant, and injects
// the tenant ID into the request header for downstream use.
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

		// Pass tenant ID to downstream middleware via header.
		r.Header.Set("X-Tenant-ID", tenant.ID)
		next.ServeHTTP(w, r)
	})
}

// rateLimitMiddleware enforces per-tenant request rate limits and token budgets.
func (cs *CloudServer) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenantID := r.Header.Get("X-Tenant-ID")
		tenant, err := cs.tenants.Get(tenantID)
		if err != nil {
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

// billingMiddleware captures usage from response bodies and meters tokens.
func (cs *CloudServer) billingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenantID := r.Header.Get("X-Tenant-ID")
		capture := &responseCapture{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(capture, r)

		// Try to extract usage from the response.
		var resp struct {
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(capture.body, &resp); err != nil {
			return
		}

		input := resp.Usage.PromptTokens
		output := resp.Usage.CompletionTokens
		if input == 0 && output == 0 {
			return
		}

		// Consume from token budget.
		if tenant, err := cs.tenants.Get(tenantID); err == nil {
			tenant.ConsumeTokens(int64(input + output))
		}

		// Record billing.
		if cs.meter != nil {
			cs.meter.Record(tenantID, input, output)
		}
	})
}

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
	rc.body = append(rc.body, b...)
	return rc.ResponseWriter.Write(b)
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
