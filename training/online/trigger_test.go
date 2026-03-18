package online

import (
	"testing"
	"time"
)

func TestDriftTrigger_NoTriggerBelowThreshold(t *testing.T) {
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 4,
			EvalWindowSize: 8,
		},
	}
	state := &TriggerState{}

	// Record 8 samples with stable loss (no drift).
	losses := []float64{1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 1.05, 1.05}
	for _, l := range losses {
		dt.RecordSample(state, l)
	}

	// 5% drift is below the 10% threshold.
	if dt.ShouldRetrain(state, 0) {
		t.Error("expected no trigger when drift is below threshold")
	}
}

func TestDriftTrigger_TriggersAboveThreshold(t *testing.T) {
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 4,
			EvalWindowSize: 8,
		},
	}
	state := &TriggerState{}

	// First half: baseline ~1.0, second half: ~1.2 (20% drift).
	losses := []float64{1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.2}
	for _, l := range losses {
		dt.RecordSample(state, l)
	}

	if !dt.ShouldRetrain(state, 0) {
		t.Error("expected trigger when drift exceeds threshold")
	}
}

func TestDriftTrigger_NotEnoughSamples(t *testing.T) {
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 10,
			EvalWindowSize: 8,
		},
	}
	state := &TriggerState{}

	// Only 4 samples, need 10.
	for _, l := range []float64{1.0, 1.0, 2.0, 2.0} {
		dt.RecordSample(state, l)
	}

	if dt.ShouldRetrain(state, 0) {
		t.Error("expected no trigger when sample count is below minimum")
	}
}

func TestScheduledTrigger_TriggersEveryN(t *testing.T) {
	st := &ScheduledTrigger{
		Interval: 5,
	}
	state := &TriggerState{}

	triggered := []int{}
	for i := 1; i <= 15; i++ {
		st.RecordSample(state, 1.0)
		if st.ShouldRetrain(state, 1.0) {
			triggered = append(triggered, i)
		}
	}

	want := []int{5, 10, 15}
	if len(triggered) != len(want) {
		t.Fatalf("expected triggers at %v, got %v", want, triggered)
	}
	for i, v := range want {
		if triggered[i] != v {
			t.Errorf("trigger %d: want sample %d, got %d", i, v, triggered[i])
		}
	}
}

func TestScheduledTrigger_NoTriggerBetweenIntervals(t *testing.T) {
	st := &ScheduledTrigger{
		Interval: 10,
	}
	state := &TriggerState{}

	for i := 0; i < 9; i++ {
		st.RecordSample(state, 1.0)
		if st.ShouldRetrain(state, 1.0) {
			t.Errorf("unexpected trigger at sample %d", i+1)
		}
	}
}

func TestTriggerCooldown_DriftTrigger(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 4,
			EvalWindowSize: 8,
			CooldownPeriod: time.Hour,
		},
		Now: func() time.Time { return now },
	}
	state := &TriggerState{}

	// Build drifting losses.
	for _, l := range []float64{1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5} {
		dt.RecordSample(state, l)
	}

	// First trigger should fire.
	if !dt.ShouldRetrain(state, 0) {
		t.Fatal("expected first trigger to fire")
	}
	state.LastTriggerTime = now

	// Advance 30 minutes — within cooldown.
	now = now.Add(30 * time.Minute)
	if dt.ShouldRetrain(state, 0) {
		t.Error("expected no trigger within cooldown period")
	}

	// Advance past cooldown.
	now = now.Add(31 * time.Minute)
	if !dt.ShouldRetrain(state, 0) {
		t.Error("expected trigger after cooldown period elapsed")
	}
}

func TestTriggerCooldown_ScheduledTrigger(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	st := &ScheduledTrigger{
		Interval: 5,
		Config: TriggerConfig{
			CooldownPeriod: 2 * time.Hour,
		},
		Now: func() time.Time { return now },
	}
	state := &TriggerState{}

	// Record 5 samples — should trigger.
	for i := 0; i < 5; i++ {
		st.RecordSample(state, 1.0)
	}
	if !st.ShouldRetrain(state, 1.0) {
		t.Fatal("expected scheduled trigger at 5 samples")
	}
	state.LastTriggerTime = now

	// Record 5 more — at 10 samples but within cooldown.
	for i := 0; i < 5; i++ {
		st.RecordSample(state, 1.0)
	}
	if st.ShouldRetrain(state, 1.0) {
		t.Error("expected no trigger within cooldown period")
	}

	// Advance past cooldown.
	now = now.Add(3 * time.Hour)
	if !st.ShouldRetrain(state, 1.0) {
		t.Error("expected trigger after cooldown elapsed")
	}
}

func TestDriftTrigger_DefaultNow(t *testing.T) {
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 4,
			EvalWindowSize: 8,
			CooldownPeriod: time.Hour,
		},
		// Now is nil — uses time.Now.
	}
	state := &TriggerState{}

	for _, l := range []float64{1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5} {
		dt.RecordSample(state, l)
	}
	// Should trigger since LastTriggerTime is zero (no cooldown applies).
	if !dt.ShouldRetrain(state, 0) {
		t.Error("expected trigger with default time.Now and zero LastTriggerTime")
	}
}

func TestScheduledTrigger_DefaultNow(t *testing.T) {
	st := &ScheduledTrigger{
		Interval: 3,
		Config: TriggerConfig{
			CooldownPeriod: time.Hour,
		},
		// Now is nil — uses time.Now.
	}
	state := &TriggerState{}

	for i := 0; i < 3; i++ {
		st.RecordSample(state, 1.0)
	}
	if !st.ShouldRetrain(state, 1.0) {
		t.Error("expected trigger with default time.Now and zero LastTriggerTime")
	}
}

func TestScheduledTrigger_ZeroInterval(t *testing.T) {
	st := &ScheduledTrigger{
		Interval: 0,
	}
	state := &TriggerState{}

	st.RecordSample(state, 1.0)
	if st.ShouldRetrain(state, 1.0) {
		t.Error("expected no trigger with zero interval")
	}
}

func TestScheduledTrigger_WindowTrimming(t *testing.T) {
	st := &ScheduledTrigger{
		Interval: 10,
		Config: TriggerConfig{
			EvalWindowSize: 3,
		},
	}
	state := &TriggerState{}

	for i := 0; i < 5; i++ {
		st.RecordSample(state, float64(i))
	}
	if len(state.RecentLosses) != 3 {
		t.Fatalf("expected window size 3, got %d", len(state.RecentLosses))
	}
}

func TestDriftTrigger_SingleSample(t *testing.T) {
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 1,
			EvalWindowSize: 10,
		},
	}
	state := &TriggerState{}

	dt.RecordSample(state, 1.0)
	// Only 1 sample — need at least 2 for baseline vs recent split.
	if dt.ShouldRetrain(state, 0) {
		t.Error("expected no trigger with only one sample")
	}
}

func TestDriftTrigger_ZeroBaseline(t *testing.T) {
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 2,
			EvalWindowSize: 4,
		},
	}
	state := &TriggerState{}

	// Baseline is 0 — should not trigger (division by zero guard).
	for _, l := range []float64{0.0, 0.0, 1.0, 1.0} {
		dt.RecordSample(state, l)
	}
	if dt.ShouldRetrain(state, 0) {
		t.Error("expected no trigger when baseline is zero")
	}
}

func TestDriftTrigger_NoWindowLimit(t *testing.T) {
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 2,
			EvalWindowSize: 0, // no trimming
		},
	}
	state := &TriggerState{}

	for _, l := range []float64{1.0, 1.0, 1.5, 1.5} {
		dt.RecordSample(state, l)
	}
	if len(state.RecentLosses) != 4 {
		t.Fatalf("expected 4 losses retained, got %d", len(state.RecentLosses))
	}
	if !dt.ShouldRetrain(state, 0) {
		t.Error("expected trigger with 50% drift")
	}
}

func TestScheduledTrigger_NoWindowLimit(t *testing.T) {
	st := &ScheduledTrigger{
		Interval: 10,
		Config: TriggerConfig{
			EvalWindowSize: 0, // no trimming
		},
	}
	state := &TriggerState{}

	for i := 0; i < 5; i++ {
		st.RecordSample(state, float64(i))
	}
	if len(state.RecentLosses) != 5 {
		t.Fatalf("expected 5 losses retained, got %d", len(state.RecentLosses))
	}
}

func TestDriftTrigger_WindowTrimming(t *testing.T) {
	dt := &DriftTrigger{
		Config: TriggerConfig{
			DriftThreshold: 0.1,
			MinSampleCount: 4,
			EvalWindowSize: 4,
		},
	}
	state := &TriggerState{}

	// Record 8 samples; window should only keep last 4.
	for _, l := range []float64{5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0} {
		dt.RecordSample(state, l)
	}

	if len(state.RecentLosses) != 4 {
		t.Fatalf("expected window size 4, got %d", len(state.RecentLosses))
	}

	// All values in window are 1.0 — no drift.
	if dt.ShouldRetrain(state, 0) {
		t.Error("expected no trigger when window contains uniform losses")
	}
}
