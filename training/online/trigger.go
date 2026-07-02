// Package online provides online learning components for continuous model
// adaptation. It includes triggers that decide when retraining should occur
// based on loss drift or sample count schedules.
package online

import "time"

// TriggerConfig holds parameters that control when retraining is triggered.
type TriggerConfig struct {
	// DriftThreshold is the relative increase in rolling mean loss over
	// baseline that triggers retraining (e.g. 0.1 means 10% increase).
	DriftThreshold float64

	// MinSampleCount is the minimum number of samples that must be
	// recorded before a trigger can fire.
	MinSampleCount int

	// EvalWindowSize is the number of recent losses used to compute the
	// rolling mean for drift detection.
	EvalWindowSize int

	// CooldownPeriod is the minimum duration between consecutive triggers.
	CooldownPeriod time.Duration
}

// TriggerState tracks the mutable state for a trigger across evaluations.
type TriggerState struct {
	// LastTriggerTime is the time the trigger last fired.
	LastTriggerTime time.Time

	// SampleCount is the total number of samples recorded.
	SampleCount int

	// RecentLosses holds the most recent loss values, up to
	// EvalWindowSize entries.
	RecentLosses []float64
}

// Trigger decides when a model should be retrained.
type Trigger interface {
	// ShouldRetrain returns true if conditions are met for retraining
	// given the current state and a new loss observation.
	ShouldRetrain(state *TriggerState, newLoss float64) bool

	// RecordSample records a new loss observation into the trigger state.
	RecordSample(state *TriggerState, loss float64)
}

// DriftTrigger fires when the rolling mean loss increases by more than
// DriftThreshold relative to the baseline (first half of the eval window).
type DriftTrigger struct {
	Config TriggerConfig
	// Now returns the current time. If nil, time.Now is used.
	Now func() time.Time
}

func (d *DriftTrigger) now() time.Time {
	if d.Now != nil {
		return d.Now()
	}
	return time.Now()
}

// ShouldRetrain returns true when the rolling mean of the most recent half
// of the eval window exceeds the baseline (first half) by more than
// DriftThreshold, provided enough samples have been collected and the
// cooldown period has elapsed.
func (d *DriftTrigger) ShouldRetrain(state *TriggerState, newLoss float64) bool {
	if state.SampleCount < d.Config.MinSampleCount {
		return false
	}
	if d.Config.CooldownPeriod > 0 && !state.LastTriggerTime.IsZero() {
		if d.now().Sub(state.LastTriggerTime) < d.Config.CooldownPeriod {
			return false
		}
	}

	n := len(state.RecentLosses)
	if n < 2 {
		return false
	}

	half := n / 2
	baseline := mean(state.RecentLosses[:half])
	recent := mean(state.RecentLosses[half:])

	if baseline == 0 {
		return false
	}
	drift := (recent - baseline) / baseline
	return drift > d.Config.DriftThreshold
}

// RecordSample appends the loss to RecentLosses, trimming to EvalWindowSize.
func (d *DriftTrigger) RecordSample(state *TriggerState, loss float64) {
	state.SampleCount++
	state.RecentLosses = append(state.RecentLosses, loss)
	if d.Config.EvalWindowSize > 0 && len(state.RecentLosses) > d.Config.EvalWindowSize {
		state.RecentLosses = state.RecentLosses[len(state.RecentLosses)-d.Config.EvalWindowSize:]
	}
}

// ScheduledTrigger fires every N samples regardless of loss values.
type ScheduledTrigger struct {
	Config TriggerConfig
	// Interval is the number of samples between triggers.
	Interval int
	// Now returns the current time. If nil, time.Now is used.
	Now func() time.Time
}

func (s *ScheduledTrigger) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

// ShouldRetrain returns true every Interval samples, provided the cooldown
// period has elapsed.
func (s *ScheduledTrigger) ShouldRetrain(state *TriggerState, _ float64) bool {
	if s.Interval <= 0 {
		return false
	}
	if s.Config.CooldownPeriod > 0 && !state.LastTriggerTime.IsZero() {
		if s.now().Sub(state.LastTriggerTime) < s.Config.CooldownPeriod {
			return false
		}
	}
	return state.SampleCount > 0 && state.SampleCount%s.Interval == 0
}

// RecordSample appends the loss to RecentLosses and increments SampleCount.
func (s *ScheduledTrigger) RecordSample(state *TriggerState, loss float64) {
	state.SampleCount++
	state.RecentLosses = append(state.RecentLosses, loss)
	if s.Config.EvalWindowSize > 0 && len(state.RecentLosses) > s.Config.EvalWindowSize {
		state.RecentLosses = state.RecentLosses[len(state.RecentLosses)-s.Config.EvalWindowSize:]
	}
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}
