// Package serve provides an OpenAI-compatible HTTP API server for model inference.
package serve

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	_ "embed"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve/security"
	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/metrics/runtime"
)

//go:embed openapi.yaml
var openapiSpec []byte

// Server wraps a loaded model and serves OpenAI-compatible HTTP endpoints.
type Server struct {
	model      *inference.Model
	draftModel *inference.Model // optional; enables speculative decoding
	mux        *http.ServeMux
	batch      *BatchScheduler // optional; nil means direct calls
	unloaded   atomic.Bool     // true after DELETE /v1/models/:id
	// modelMu serializes handler access to model against DELETE /v1/models/:id.
	// Handlers that touch the model take RLock for the duration of their work;
	// delete takes Lock before flipping unloaded and closing the model. This
	// makes both the use-after-close and the "Add after Wait started" races
	// structurally impossible (see CONC-H2).
	modelMu         sync.RWMutex
	transcriber     Transcriber    // optional; enables /v1/audio/transcriptions
	classifier      Classifier     // optional; enables /v1/classify
	guardEvaluator  GuardEvaluator // optional; enables /v1/guard endpoints
	logger          log.Logger
	metrics         *ServerMetrics
	classifyMetrics *ClassifyMetrics
	guardMetrics    *GuardMetrics
	collector       runtime.Collector
	gpus            []int                 // GPU IDs to distribute model across
	apiKey          string                // optional; enables Bearer token auth
	keyStore        *security.KeyStore    // optional; enables scope-based authorization
	rateLimiter     *security.RateLimiter // optional; enables per-IP rate limiting
	maxTokens       int                   // server-side upper bound for max_tokens (default 8192)
	adapterCache    *AdapterCacheHandle   // optional; enables per-request LoRA adapter selection
}

// ServerOption configures the server.
type ServerOption func(*Server)

// WithDraftModel enables speculative decoding using the given draft model.
// When set, completion requests use speculative decode with the draft model
// proposing tokens and the target model verifying them.
func WithDraftModel(draft *inference.Model) ServerOption {
	return func(s *Server) {
		s.draftModel = draft
	}
}

// WithLogger sets the logger for request logging.
func WithLogger(l log.Logger) ServerOption {
	return func(s *Server) {
		s.logger = l
	}
}

// WithMetrics sets the metrics collector for token rate and request tracking.
func WithMetrics(c runtime.Collector) ServerOption {
	return func(s *Server) {
		s.collector = c
	}
}

// WithBatchScheduler attaches a batch scheduler for non-streaming requests.
// When set, incoming completion requests are routed through the scheduler
// to be grouped into batches for higher throughput.
func WithBatchScheduler(bs *BatchScheduler) ServerOption {
	return func(s *Server) {
		s.batch = bs
	}
}

// WithAPIKey enables Bearer token authentication on all endpoints
// except health checks (/healthz, /readyz), metrics (/metrics), and
// the OpenAPI spec (/openapi.yaml).
func WithAPIKey(key string) ServerOption {
	return func(s *Server) {
		s.apiKey = key
	}
}

// WithGPUs sets the GPU IDs to distribute the model across.
func WithGPUs(ids []int) ServerOption {
	return func(s *Server) {
		s.gpus = ids
	}
}

// WithMaxTokens sets the server-side upper bound for max_tokens in completion
// requests. Any request asking for more tokens than this limit will be clamped.
// The default is 8192.
func WithMaxTokens(n int) ServerOption {
	return func(s *Server) {
		s.maxTokens = n
	}
}

// WithRateLimiter enables per-IP rate limiting using the provided RateLimiter.
// When set, requests that exceed the rate limit receive 429 Too Many Requests.
func WithRateLimiter(rl *security.RateLimiter) ServerOption {
	return func(s *Server) {
		s.rateLimiter = rl
	}
}

// WithTrustedProxies configures the set of reverse-proxy IPs whose
// X-Forwarded-For and X-Real-IP headers are trusted for client-IP
// extraction. When the rate limiter is enabled, only requests arriving
// from these addresses will have their forwarding headers honoured;
// all other requests use RemoteAddr directly.
func WithTrustedProxies(proxies []string) ServerOption {
	return func(s *Server) {
		if s.rateLimiter != nil {
			s.rateLimiter.SetTrustedProxies(proxies)
		}
	}
}

// WithKeyStore enables scope-based authorization using the provided KeyStore.
// When set, after Bearer token validation the middleware looks up the key in the
// store and checks that it has a sufficient scope for the endpoint.
func WithKeyStore(ks *security.KeyStore) ServerOption {
	return func(s *Server) {
		s.keyStore = ks
	}
}

// GPUs returns the configured GPU IDs, or nil if not set.
func (s *Server) GPUs() []int {
	return s.gpus
}

// NewServer creates a Server for the given model.
func NewServer(m *inference.Model, opts ...ServerOption) *Server {
	s := &Server{model: m, mux: http.NewServeMux(), maxTokens: 8192}
	for _, opt := range opts {
		opt(s)
	}
	if s.logger == nil {
		s.logger = log.Nop()
	}
	if s.collector == nil {
		s.collector = runtime.Nop()
	}
	// Start the rate limiter's background cleanup loop so its per-IP bucket
	// map does not grow unbounded under sustained traffic from many distinct
	// client IPs (CONC-M1). Close stops it again.
	if s.rateLimiter != nil {
		s.rateLimiter.Start()
	}
	// Auto-wire batch handler to use GenerateBatch when a BatchScheduler
	// is attached but no handler has been configured.
	if s.batch != nil && s.batch.config.Handler == nil {
		s.batch.config.Handler = func(ctx context.Context, reqs []BatchRequest) []BatchResult {
			prompts := make([]string, len(reqs))
			for i, r := range reqs {
				prompts[i] = r.Prompt
			}
			outputs, err := s.model.GenerateBatch(ctx, prompts)
			results := make([]BatchResult, len(reqs))
			for i := range results {
				if err != nil {
					results[i] = BatchResult{Err: err}
				} else {
					results[i] = BatchResult{Value: outputs[i]}
				}
			}
			return results
		}
	}
	s.metrics = NewServerMetrics(s.collector)
	s.classifyMetrics = NewClassifyMetrics(s.collector)
	s.guardMetrics = NewGuardMetrics(s.collector)
	s.mux.HandleFunc("POST /v1/chat/completions", s.recoveryMiddleware(s.handleChatCompletions))
	s.mux.HandleFunc("POST /v1/completions", s.recoveryMiddleware(s.handleCompletions))
	s.mux.HandleFunc("POST /v1/embeddings", s.recoveryMiddleware(s.handleEmbeddings))
	s.mux.HandleFunc("GET /v1/models", s.recoveryMiddleware(s.handleModels))
	s.mux.HandleFunc("GET /v1/models/{id...}", s.recoveryMiddleware(s.handleModelInfo))
	s.mux.HandleFunc("DELETE /v1/models/{id...}", s.recoveryMiddleware(s.handleModelDelete))
	s.mux.HandleFunc("POST /v1/audio/transcriptions", s.recoveryMiddleware(s.handleAudioTranscriptions))
	s.mux.HandleFunc("POST /v1/classify", s.recoveryMiddleware(s.handleClassify))
	s.mux.HandleFunc("POST /v1/guard", s.recoveryMiddleware(s.handleGuard))
	s.mux.HandleFunc("POST /v1/guard/batch", s.recoveryMiddleware(s.handleGuardBatch))
	s.mux.HandleFunc("POST /v1/guard/scan", s.recoveryMiddleware(s.handleGuardScan))
	s.mux.HandleFunc("GET /healthz", s.recoveryMiddleware(s.handleHealthz))
	s.mux.HandleFunc("GET /readyz", s.recoveryMiddleware(s.handleReadyz))
	s.mux.HandleFunc("GET /openapi.yaml", s.recoveryMiddleware(handleOpenAPISpec))
	s.mux.HandleFunc("GET /metrics", handleMetrics(s.collector))
	return s
}

// Handler returns the HTTP handler for this server.
func (s *Server) Handler() http.Handler {
	var h http.Handler = s.mux
	if s.apiKey != "" || s.keyStore != nil {
		h = s.authMiddleware(h)
	}
	if s.rateLimiter != nil {
		h = s.rateLimitMiddleware(h)
	}
	h = s.requestIDMiddleware(h)
	h = s.logMiddleware(h)
	return s.securityHeadersMiddleware(h)
}

// securityHeadersMiddleware sets standard security headers on every response.
func (s *Server) securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Cache-Control", "no-store")
		next.ServeHTTP(w, r)
	})
}

// requestIDKey is the context key for the request ID.
type requestIDKey struct{}

// RequestID returns the request ID from the context, or an empty string if not set.
func RequestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey{}).(string)
	return id
}

// requestIDMiddleware reads X-Request-Id from the request header; if absent,
// it generates a UUID v4. The ID is stored in the request context and echoed
// back in the X-Request-Id response header.
func (s *Server) requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-Id")
		if id == "" {
			id = generateRequestID()
		}
		w.Header().Set("X-Request-Id", id)
		ctx := context.WithValue(r.Context(), requestIDKey{}, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// generateRequestID produces a UUID v4 string using crypto/rand.
func generateRequestID() string {
	var buf [16]byte
	_, _ = rand.Read(buf[:])
	buf[6] = (buf[6] & 0x0f) | 0x40 // version 4
	buf[8] = (buf[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		buf[0:4], buf[4:6], buf[6:8], buf[8:10], buf[10:16])
}

// authMiddleware rejects requests that do not carry a valid Bearer token.
// Health-check, metrics, and OpenAPI spec paths are exempt.
func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics", "/healthz", "/readyz", "/openapi.yaml":
			next.ServeHTTP(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		const prefix = "Bearer "
		if !strings.HasPrefix(auth, prefix) {
			writeError(w, http.StatusUnauthorized, "missing or invalid authorization header")
			return
		}
		token := auth[len(prefix):]

		// When a KeyStore is configured, validate against it and enforce scopes.
		if s.keyStore != nil {
			key := s.keyStore.Lookup(token)
			if key == nil {
				writeError(w, http.StatusUnauthorized, "invalid API key")
				return
			}
			if !key.Valid(time.Now()) {
				writeError(w, http.StatusUnauthorized, "invalid API key")
				return
			}
			if required := requiredScope(r.Method, r.URL.Path); required != "" {
				if !key.HasScope(required) {
					writeError(w, http.StatusForbidden, "insufficient scope")
					return
				}
			}
			next.ServeHTTP(w, r)
			return
		}

		// Static API key mode — no scope checks.
		if subtle.ConstantTimeCompare([]byte(token), []byte(s.apiKey)) != 1 {
			writeError(w, http.StatusUnauthorized, "invalid API key")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// requiredScope returns the minimum scope required for the given HTTP method and path.
// DELETE /v1/models requires ScopeAdmin. POST /v1/* requires ScopeInference.
// All /v1/ routes require at least ScopeReadOnly. Returns empty string for non-/v1/ paths.
func requiredScope(method, path string) security.Scope {
	if method == http.MethodDelete && strings.HasPrefix(path, "/v1/models") {
		return security.ScopeAdmin
	}
	if method == http.MethodPost && strings.HasPrefix(path, "/v1/") {
		return security.ScopeInference
	}
	if method == http.MethodGet && strings.HasPrefix(path, "/v1/models") {
		return security.ScopeReadOnly
	}
	if strings.HasPrefix(path, "/v1/") {
		return security.ScopeReadOnly
	}
	return ""
}

// rateLimitMiddleware rejects requests that exceed the configured rate limit
// for the client IP. Returns 429 Too Many Requests when the limit is exceeded.
func (s *Server) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := security.ClientIPTrusted(r, s.rateLimiter.TrustedProxies())
		if !s.rateLimiter.Allow(ip) {
			writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// statusRecorder wraps http.ResponseWriter to capture the status code.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *Server) logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.metrics.IncActiveRequests()
		defer s.metrics.DecActiveRequests()

		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		latency := time.Since(start).Milliseconds()

		// Record error metrics for non-2xx responses. The path is normalized
		// to a bounded label set before recording: logMiddleware runs before
		// authMiddleware, so an unauthenticated caller must never be able to
		// mint an unbounded number of permanent counter entries by probing
		// distinct paths (SERVE-1).
		if rec.status >= 400 {
			s.metrics.RecordError(normalizeRoute(r.URL.Path), rec.status)
		}

		modelID := ""
		if info := s.model.Info(); info != nil {
			modelID = info.ID
		}

		fields := []string{
			"method", r.Method,
			// EscapedPath (not the percent-decoded Path) is logged here: an
			// attacker-controlled path can otherwise embed control characters
			// (e.g. CR/LF) that split or forge log lines (SERVE-6, CWE-117).
			"path", r.URL.EscapedPath(),
			"model", modelID,
			"prompt_tokens", "0",
			"completion_tokens", "0",
			"latency_ms", strconv.FormatInt(latency, 10),
			"status_code", strconv.Itoa(rec.status),
		}

		switch {
		case rec.status >= 500:
			s.logger.Error("request completed", fields...)
		case rec.status >= 400:
			s.logger.Warn("request completed", fields...)
		default:
			s.logger.Info("request completed", fields...)
		}
	})
}

func (s *Server) recoveryMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				msg := fmt.Sprintf("%v", rec)
				s.logger.Error("panic recovered", "error", msg, "method", r.Method, "path", r.URL.Path)
				fmt.Fprintf(os.Stderr, "panic recovered: %s %s: %s\n", r.Method, r.URL.Path, msg)
				writeError(w, http.StatusInternalServerError, "internal server error")
			}
		}()
		next(w, r)
	}
}

// Close implements shutdown.Closer for graceful shutdown integration.
func (s *Server) Close(_ context.Context) error {
	if s.batch != nil {
		s.batch.Stop()
	}
	if s.rateLimiter != nil {
		s.rateLimiter.Stop()
	}
	return nil
}
