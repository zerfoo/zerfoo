package serve

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/zerfoo/zerfoo/inference/guardian"
)

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
	Error    string       `json:"error"`
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
			// still forward it to the inner handler.
			body, err := io.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
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

			// Capture the inner handler's response for output checking.
			rec := &responseBuffer{header: make(http.Header)}
			next.ServeHTTP(rec, r)

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

// responseBuffer captures an HTTP response in memory.
type responseBuffer struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func (rb *responseBuffer) Header() http.Header { return rb.header }

func (rb *responseBuffer) WriteHeader(code int) { rb.status = code }

func (rb *responseBuffer) Write(b []byte) (int, error) {
	if rb.status == 0 {
		rb.status = http.StatusOK
	}
	return rb.body.Write(b)
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
