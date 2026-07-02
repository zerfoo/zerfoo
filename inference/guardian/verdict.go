package guardian

import (
	"math"
	"regexp"
	"strings"
)

// Verdict represents a Guardian safety evaluation result.
type Verdict struct {
	Unsafe     bool    // true if risk detected (model answered "Yes")
	Risk       string  // risk category name
	Confidence float64 // 0.0-1.0
	Reasoning  string  // thinking trace (3.3 only)
}

var (
	scoreRe      = regexp.MustCompile(`(?i)<score>\s*(yes|no)\s*</score>`)
	thinkRe      = regexp.MustCompile(`(?s)<think>(.*?)</think>`)
	confidenceRe = regexp.MustCompile(`(?i)<confidence>\s*(high|low)\s*</confidence>`)
)

// ParseVerdict extracts a safety verdict from Guardian model output.
// Handles three output format variants:
//   - 3.0: single token "Yes"/"No", confidence from logprobs
//   - 3.2: "Yes"/"No" + <confidence>High</confidence> or <confidence>Low</confidence>
//   - 3.3: optional <think>...</think> trace + <score>yes</score> or <score>no</score>
func ParseVerdict(output string, risk string, logprobs []float64) Verdict {
	// Try 3.3 format: <score>yes</score> or <score>no</score>
	if m := scoreRe.FindStringSubmatch(output); m != nil {
		v := Verdict{
			Unsafe:     strings.EqualFold(m[1], "yes"),
			Risk:       risk,
			Confidence: 1.0,
		}
		if tm := thinkRe.FindStringSubmatch(output); tm != nil {
			v.Reasoning = strings.TrimSpace(tm[1])
		}
		return v
	}

	// Try 3.2 format: <confidence>High</confidence> or <confidence>Low</confidence>
	if m := confidenceRe.FindStringSubmatch(output); m != nil {
		conf := 0.3
		if strings.EqualFold(m[1], "high") {
			conf = 0.9
		}
		trimmed := strings.TrimSpace(output)
		unsafe := strings.HasPrefix(strings.ToLower(trimmed), "yes")
		return Verdict{
			Unsafe:     unsafe,
			Risk:       risk,
			Confidence: conf,
		}
	}

	// Try 3.0 format: plain "Yes"/"No"
	trimmed := strings.TrimSpace(output)
	lower := strings.ToLower(trimmed)
	if strings.HasPrefix(lower, "yes") {
		return Verdict{
			Unsafe:     true,
			Risk:       risk,
			Confidence: softmaxConfidence(logprobs),
		}
	}
	if strings.HasPrefix(lower, "no") {
		return Verdict{
			Unsafe:     false,
			Risk:       risk,
			Confidence: softmaxConfidence(logprobs),
		}
	}

	// No recognized format.
	return Verdict{
		Unsafe:     false,
		Risk:       risk,
		Confidence: 0,
	}
}

// softmaxConfidence computes confidence from the first two logprob values
// using softmax. Returns 0 if fewer than 2 values are provided.
func softmaxConfidence(logprobs []float64) float64 {
	if len(logprobs) < 2 {
		return 0
	}
	// Numerical stability: subtract max.
	maxLP := logprobs[0]
	if logprobs[1] > maxLP {
		maxLP = logprobs[1]
	}
	e0 := math.Exp(logprobs[0] - maxLP)
	e1 := math.Exp(logprobs[1] - maxLP)
	return e0 / (e0 + e1)
}
