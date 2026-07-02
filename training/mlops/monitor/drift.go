package monitor

import (
	"math"
	"sync"
)

// DriftDetector observes a stream of scalar values and returns true
// when it detects a statistically significant distributional shift.
type DriftDetector interface {
	// Observe ingests the next value in the stream.
	// It returns true when drift is detected.
	Observe(value float64) bool
}

// PageHinkleyConfig holds parameters for the Page-Hinkley test.
type PageHinkleyConfig struct {
	// Delta is the magnitude of allowed changes (tolerance).
	// A smaller delta makes the test more sensitive.
	// Defaults to 0.005 if zero.
	Delta float64

	// Lambda is the detection threshold. When the test statistic
	// exceeds Lambda, drift is signalled.
	// Defaults to 50 if zero.
	Lambda float64
}

// PageHinkley implements the Page-Hinkley test for detecting changes
// in the mean of a sequential stream of values.
//
// The test maintains a cumulative sum of deviations from the running
// mean, adjusted by a tolerance parameter delta. Drift is detected
// when the difference between the maximum cumulative sum and the
// current cumulative sum exceeds a threshold lambda.
type PageHinkley struct {
	mu     sync.Mutex
	delta  float64
	lambda float64

	n      int
	sum    float64
	cumSum float64
	minSum float64
}

// NewPageHinkley creates a Page-Hinkley detector with the given config.
func NewPageHinkley(cfg PageHinkleyConfig) *PageHinkley {
	delta := cfg.Delta
	if delta == 0 {
		delta = 0.005
	}
	lambda := cfg.Lambda
	if lambda == 0 {
		lambda = 50
	}
	return &PageHinkley{
		delta:  delta,
		lambda: lambda,
	}
}

// Observe ingests a value and returns true if drift is detected.
func (ph *PageHinkley) Observe(value float64) bool {
	ph.mu.Lock()
	defer ph.mu.Unlock()

	ph.n++
	ph.sum += value
	mean := ph.sum / float64(ph.n)

	ph.cumSum += value - mean - ph.delta
	if ph.cumSum < ph.minSum {
		ph.minSum = ph.cumSum
	}

	return (ph.cumSum - ph.minSum) > ph.lambda
}

// Reset clears all internal state so the detector can be reused.
func (ph *PageHinkley) Reset() {
	ph.mu.Lock()
	defer ph.mu.Unlock()

	ph.n = 0
	ph.sum = 0
	ph.cumSum = 0
	ph.minSum = 0
}

// ADWINConfig holds parameters for the ADWIN detector.
type ADWINConfig struct {
	// Confidence controls the sensitivity of the change detector.
	// Lower values make it more sensitive. Typical range is 0.001 to 0.01.
	// Defaults to 0.002 if zero.
	Confidence float64
}

// ADWIN (ADaptive WINdowing) detects distributional shifts by maintaining
// a variable-length window of observations and testing whether any two
// sub-windows have statistically different means.
//
// When a change is detected, the older sub-window is dropped, allowing
// the detector to adapt to the new distribution.
type ADWIN struct {
	mu         sync.Mutex
	confidence float64
	window     []float64
	sum        float64
}

// NewADWIN creates an ADWIN detector with the given config.
func NewADWIN(cfg ADWINConfig) *ADWIN {
	conf := cfg.Confidence
	if conf == 0 {
		conf = 0.002
	}
	return &ADWIN{
		confidence: conf,
	}
}

// Observe ingests a value and returns true if drift is detected.
// When drift is detected, the older portion of the window is dropped.
func (a *ADWIN) Observe(value float64) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.window = append(a.window, value)
	a.sum += value

	n := len(a.window)
	if n < 4 {
		return false
	}

	// Check all possible cut points for a statistically significant
	// difference in means between the two resulting sub-windows.
	// We scan from the oldest cut to the newest.
	var prefixSum float64
	for i := 1; i < n-1; i++ {
		prefixSum += a.window[i-1]
		n0 := float64(i)
		n1 := float64(n - i)
		mean0 := prefixSum / n0
		mean1 := (a.sum - prefixSum) / n1

		// Hoeffding bound for the difference of two means.
		nHarmonic := 1.0/n0 + 1.0/n1
		epsilon := math.Sqrt(0.5 * nHarmonic * math.Log(4.0/a.confidence))

		if math.Abs(mean0-mean1) >= epsilon {
			// Drift detected: drop the older sub-window.
			a.window = a.window[i:]
			a.sum = 0
			for _, v := range a.window {
				a.sum += v
			}
			return true
		}
	}
	return false
}

// Reset clears all internal state.
func (a *ADWIN) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.window = nil
	a.sum = 0
}
