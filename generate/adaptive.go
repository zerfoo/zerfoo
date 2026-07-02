package generate

// adaptiveDraftLen tracks the acceptance rate over a rolling window and
// adjusts the draft length dynamically. When acceptance is high (>80%),
// the draft length increases. When low (<40%), it decreases.
type adaptiveDraftLen struct {
	window   []bool // ring buffer of accept/reject outcomes
	pos      int    // next write position in ring buffer
	count    int    // total entries written (capped at len(window))
	accepts  int    // number of true values in window
	current  int    // current draft length
	minLen   int
	maxLen   int
	highRate float64 // increase N when acceptance > this
	lowRate  float64 // decrease N when acceptance < this
}

// newAdaptiveDraftLen creates an adaptive draft length tracker.
func newAdaptiveDraftLen(initial, minLen, maxLen, windowSize int) *adaptiveDraftLen {
	if initial < minLen {
		initial = minLen
	}
	if initial > maxLen {
		initial = maxLen
	}
	return &adaptiveDraftLen{
		window:   make([]bool, windowSize),
		current:  initial,
		minLen:   minLen,
		maxLen:   maxLen,
		highRate: 0.80,
		lowRate:  0.40,
	}
}

// Record records a batch of accept/reject outcomes. accepted is how many
// of proposed tokens were accepted. proposed is the total proposed.
func (a *adaptiveDraftLen) Record(accepted, proposed int) {
	for i := range proposed {
		wasAccepted := i < accepted

		// If overwriting an old entry, adjust accepts count.
		if a.count >= len(a.window) && a.window[a.pos] {
			a.accepts--
		}

		a.window[a.pos] = wasAccepted
		if wasAccepted {
			a.accepts++
		}

		a.pos = (a.pos + 1) % len(a.window)
		if a.count < len(a.window) {
			a.count++
		}
	}

	a.adjust()
}

// Current returns the current draft length.
func (a *adaptiveDraftLen) Current() int {
	return a.current
}

// Rate returns the current acceptance rate (0.0-1.0).
func (a *adaptiveDraftLen) Rate() float64 {
	if a.count == 0 {
		return 1.0
	}
	return float64(a.accepts) / float64(a.count)
}

func (a *adaptiveDraftLen) adjust() {
	// Only adjust after accumulating enough data.
	if a.count < len(a.window)/2 {
		return
	}

	rate := a.Rate()
	switch {
	case rate > a.highRate && a.current < a.maxLen:
		a.current++
	case rate < a.lowRate && a.current > a.minLen:
		a.current--
	}
}
