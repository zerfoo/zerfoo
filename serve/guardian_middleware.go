package serve

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/zerfoo/zerfoo/inference/guardian"
)

// guardianMaxRequestBodyBytes caps the request body GuardianMiddleware will
// buffer for parsing. Matches the 10 MB cap used elsewhere in serve/ (see
// handlers.go, classify.go, guard.go).
const guardianMaxRequestBodyBytes = 10 << 20 // 10 MB

// GuardianMiddlewareConfig controls how the Guardian safety middleware
// intercepts chat completion requests.
type GuardianMiddlewareConfig struct {
	Model       string   // Guardian model path
	Risks       []string // risk categories (default: HarmRiskCategories)
	CheckInput  bool     // scan user prompts (default: true)
	CheckOutput bool     // scan assistant responses (default: false)
	BlockOnFlag bool     // return 400 if flagged (default: true)
}

// guardianFlaggedResponse is returned when content is flagged and BlockOnFlag is true.
type guardianFlaggedResponse struct {
	Error    string        `json:"error"`
	Verdicts []VerdictData `json:"verdicts"`
}

// GuardianMiddleware returns HTTP middleware that wraps chat completion
// requests with Guardian safety checks. If evaluator is nil the middleware
// is a no-op pass-through.
func GuardianMiddleware(evaluator GuardEvaluator, config GuardianMiddlewareConfig) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Pass through when evaluator is nil or not a chat completions request.
			if evaluator == nil || r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
				next.ServeHTTP(w, r)
				return
			}

			// Read and buffer the request body so we can parse it and
			// still forward it to the inner handler. Bound the read so an
			// attacker can't force unbounded memory growth (SERVE-4).
			r.Body = http.MaxBytesReader(w, r.Body, guardianMaxRequestBodyBytes)
			body, err := io.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
				if isMaxBytesError(err) {
					writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
					return
				}
				writeError(w, http.StatusBadRequest, "failed to read request body")
				return
			}

			var req ChatCompletionRequest
			if err := json.Unmarshal(body, &req); err != nil {
				// Let the inner handler deal with malformed JSON.
				r.Body = io.NopCloser(bytes.NewReader(body))
				next.ServeHTTP(w, r)
				return
			}

			risks := config.Risks
			if len(risks) == 0 {
				risks = guardian.HarmRiskCategories()
			}

			// --- Input check ---
			if config.CheckInput {
				userMsg := lastUserMessage(req.Messages)
				if userMsg != "" {
					verdicts, evalErr := evaluator.Evaluate(r.Context(), guardian.GuardianRequest{
						Input: guardian.GuardianInput{User: userMsg},
						Risks: risks,
					})
					if evalErr == nil {
						flagged, data := toVerdictData(verdicts)
						if flagged {
							if config.BlockOnFlag {
								writeJSON(w, http.StatusBadRequest, guardianFlaggedResponse{
									Error:    "content_flagged",
									Verdicts: data,
								})
								return
							}
							w.Header().Set("X-Guardian-Flagged", "true")
						}
					}
				}
			}

			// --- Forward to inner handler ---
			r.Body = io.NopCloser(bytes.NewReader(body))

			if !config.CheckOutput {
				next.ServeHTTP(w, r)
				return
			}

			// Capture the inner handler's response for output checking. If
			// the handler turns out to be streaming a Server-Sent Events
			// response, the buffer switches to passing bytes straight
			// through to the real ResponseWriter (flushing as it goes)
			// instead of accumulating the whole response in memory, since
			// SSE streams can be unbounded/long-lived and output scanning
			// requires a complete response body (SERVE-4).
			rec := &responseBuffer{header: make(http.Header), underlying: w}
			next.ServeHTTP(rec, r)

			if rec.passthrough {
				// Response was already streamed directly to the client;
				// there is nothing left to check or forward.
				return
			}

			// --- Output check ---
			assistantMsg := extractAssistantContent(rec.body.Bytes())
			if assistantMsg != "" {
				verdicts, evalErr := evaluator.Evaluate(r.Context(), guardian.GuardianRequest{
					Input: guardian.GuardianInput{
						User:      lastUserMessage(req.Messages),
						Assistant: assistantMsg,
					},
					Risks: risks,
				})
				if evalErr == nil {
					flagged, data := toVerdictData(verdicts)
					if flagged {
						if config.BlockOnFlag {
							writeJSON(w, http.StatusBadRequest, guardianFlaggedResponse{
								Error:    "content_flagged",
								Verdicts: data,
							})
							return
						}
						w.Header().Set("X-Guardian-Flagged", "true")
					}
				}
			}

			// Forward the captured response.
			copyHeader(w.Header(), rec.header)
			w.WriteHeader(rec.status)
			w.Write(rec.body.Bytes()) //nolint:errcheck
		})
	}
}

// responseBuffer captures an HTTP response in memory so GuardianMiddleware
// can inspect it before forwarding to the client. If the response turns out
// to be a Server-Sent Events stream (Content-Type: text/event-stream), it
// switches to passthrough mode: bytes are written directly to the
// underlying ResponseWriter (and flushed) instead of being buffered, so
// streaming responses are neither delayed nor unboundedly accumulated in
// memory.
type responseBuffer struct {
	underlying    http.ResponseWriter
	header        http.Header
	body          bytes.Buffer
	status        int
	headerWritten bool
	passthrough   bool
}

func (rb *responseBuffer) Header() http.Header { return rb.header }

func (rb *responseBuffer) WriteHeader(code int) {
	if rb.headerWritten {
		return
	}
	rb.commit(code)
}

func (rb *responseBuffer) Write(b []byte) (int, error) {
	if !rb.headerWritten {
		rb.commit(http.StatusOK)
	}
	if rb.passthrough {
		n, err := rb.underlying.Write(b)
		if f, ok := rb.underlying.(http.Flusher); ok {
			f.Flush()
		}
		return n, err
	}
	return rb.body.Write(b)
}

// commit finalizes the response status/headers on first WriteHeader/Write
// and decides whether to switch into SSE passthrough mode.
func (rb *responseBuffer) commit(code int) {
	rb.status = code
	rb.headerWritten = true
	if isEventStream(rb.header) {
		rb.passthrough = true
		copyHeader(rb.underlying.Header(), rb.header)
		rb.underlying.WriteHeader(code)
	}
}

// isEventStream reports whether the given response header declares an SSE
// (text/event-stream) body.
func isEventStream(h http.Header) bool {
	ct := h.Get("Content-Type")
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(ct)), "text/event-stream")
}

// lastUserMessage returns the content of the last message with role "user".
func lastUserMessage(msgs []ChatMessage) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return msgs[i].Content
		}
	}
	return ""
}

// extractAssistantContent parses a ChatCompletionResponse and returns the
// first choice's assistant message content.
func extractAssistantContent(data []byte) string {
	var resp ChatCompletionResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return ""
	}
	if len(resp.Choices) == 0 {
		return ""
	}
	return resp.Choices[0].Message.Content
}

// toVerdictData converts guardian verdicts to API verdict data and reports
// whether any verdict is flagged as unsafe.
func toVerdictData(verdicts []guardian.Verdict) (bool, []VerdictData) {
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
	return flagged, data
}

// copyHeader copies all headers from src to dst.
func copyHeader(dst, src http.Header) {
	for k, vv := range src {
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}
