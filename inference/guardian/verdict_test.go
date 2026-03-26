package guardian

import (
	"math"
	"testing"
)

func TestParseVerdict(t *testing.T) {
	tests := []struct {
		name       string
		output     string
		risk       string
		logprobs   []float64
		wantUnsafe bool
		wantConf   float64
		wantReason string
	}{
		{
			name:       "3.0 yes",
			output:     "Yes",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   0,
		},
		{
			name:       "3.0 no",
			output:     "No",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   0,
		},
		{
			name:       "3.2 yes high confidence",
			output:     "Yes\n<confidence>High</confidence>",
			risk:       "violence",
			wantUnsafe: true,
			wantConf:   0.9,
		},
		{
			name:       "3.2 no low confidence",
			output:     "No\n<confidence>Low</confidence>",
			risk:       "profanity",
			wantUnsafe: false,
			wantConf:   0.3,
		},
		{
			name:       "3.3 yes with thinking",
			output:     "<think>The message contains violence</think>\n<score>yes</score>",
			risk:       "violence",
			wantUnsafe: true,
			wantConf:   1.0,
			wantReason: "The message contains violence",
		},
		{
			name:       "3.3 no without thinking",
			output:     "<score>no</score>",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   1.0,
		},
		{
			name:       "empty output",
			output:     "",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   0,
		},
		{
			name:       "malformed output",
			output:     "Maybe",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   0,
		},
		{
			name:       "case insensitive YES",
			output:     "YES",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   0,
		},
		{
			name:       "case insensitive no",
			output:     "no",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   0,
		},
		{
			name:       "case insensitive Yes mixed",
			output:     "Yes",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   0,
		},
		{
			name:       "logprob confidence yes",
			output:     "Yes",
			risk:       "harm",
			logprobs:   []float64{-0.1, -2.0},
			wantUnsafe: true,
			wantConf:   softmaxExpected(-0.1, -2.0),
		},
		{
			name:       "logprob confidence no",
			output:     "No",
			risk:       "harm",
			logprobs:   []float64{-1.5, -0.2},
			wantUnsafe: false,
			wantConf:   softmaxExpected(-1.5, -0.2),
		},
		{
			name:       "3.3 score case insensitive",
			output:     "<score>YES</score>",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   1.0,
		},
		{
			name:       "3.2 confidence case insensitive",
			output:     "Yes\n<confidence>HIGH</confidence>",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   0.9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseVerdict(tt.output, tt.risk, tt.logprobs)

			if got.Unsafe != tt.wantUnsafe {
				t.Errorf("Unsafe = %v, want %v", got.Unsafe, tt.wantUnsafe)
			}
			if got.Risk != tt.risk {
				t.Errorf("Risk = %q, want %q", got.Risk, tt.risk)
			}
			if math.Abs(got.Confidence-tt.wantConf) > 1e-9 {
				t.Errorf("Confidence = %v, want %v", got.Confidence, tt.wantConf)
			}
			if got.Reasoning != tt.wantReason {
				t.Errorf("Reasoning = %q, want %q", got.Reasoning, tt.wantReason)
			}
		})
	}
}

// softmaxExpected computes the expected softmax confidence for a pair of logprobs.
func softmaxExpected(lp0, lp1 float64) float64 {
	maxLP := lp0
	if lp1 > maxLP {
		maxLP = lp1
	}
	e0 := math.Exp(lp0 - maxLP)
	e1 := math.Exp(lp1 - maxLP)
	return e0 / (e0 + e1)
}
