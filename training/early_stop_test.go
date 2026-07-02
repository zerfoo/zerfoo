package training

import (
	"math"
	"testing"
)

func TestSmoothedEarlyStopping_Improving(t *testing.T) {
	es := NewEarlyStopping(EarlyStopConfig{
		Patience: 5,
		Alpha:    0.3,
		MinDelta: 0.001,
		Mode:     "min",
	})

	// Steadily decreasing loss — should never trigger.
	for i := 0; i < 50; i++ {
		loss := 1.0 - float64(i)*0.01
		if es.Step(loss) {
			t.Fatalf("early stop triggered at epoch %d with improving loss", i)
		}
	}
}

func TestSmoothedEarlyStopping_Plateau(t *testing.T) {
	es := NewEarlyStopping(EarlyStopConfig{
		Patience: 3,
		Alpha:    0.5,
		MinDelta: 0.001,
		Mode:     "min",
	})

	// Initial improvement.
	es.Step(1.0)
	es.Step(0.5)

	// Plateau at 0.5 — should trigger after patience epochs.
	var stopped bool
	for i := 0; i < 20; i++ {
		if es.Step(0.5) {
			stopped = true
			break
		}
	}
	if !stopped {
		t.Fatal("early stop did not trigger on plateau")
	}
}

func TestSmoothedEarlyStopping_Noisy(t *testing.T) {
	es := NewEarlyStopping(EarlyStopConfig{
		Patience: 5,
		Alpha:    0.1, // Heavy smoothing to filter noise.
		MinDelta: 0.001,
		Mode:     "min",
	})

	// Noisy but downward-trending loss.
	losses := []float64{
		1.0, 1.1, 0.9, 1.0, 0.85,
		0.95, 0.80, 0.90, 0.75, 0.85,
		0.70, 0.80, 0.65, 0.75, 0.60,
		0.70, 0.55, 0.65, 0.50, 0.60,
	}

	for i, loss := range losses {
		if es.Step(loss) {
			t.Fatalf("early stop triggered at epoch %d; EMA smoothing should prevent premature stop on noisy data", i)
		}
	}
}

func TestSmoothedEarlyStopping_MaxMode(t *testing.T) {
	es := NewEarlyStopping(EarlyStopConfig{
		Patience: 3,
		Alpha:    0.3,
		MinDelta: 0.001,
		Mode:     "max",
	})

	// Increasing accuracy — should not trigger.
	for i := 0; i < 10; i++ {
		acc := 0.5 + float64(i)*0.04
		if es.Step(acc) {
			t.Fatalf("early stop triggered at epoch %d with improving accuracy", i)
		}
	}

	// Plateau — should trigger after patience.
	var stopped bool
	for i := 0; i < 20; i++ {
		if es.Step(0.9) {
			stopped = true
			break
		}
	}
	if !stopped {
		t.Fatal("early stop did not trigger on accuracy plateau in max mode")
	}
}

func TestSmoothedEarlyStopping_MinDelta(t *testing.T) {
	es := NewEarlyStopping(EarlyStopConfig{
		Patience: 3,
		Alpha:    1.0, // No smoothing — raw metric used directly.
		MinDelta: 0.1,
		Mode:     "min",
	})

	// Tiny improvements below MinDelta should count as no improvement.
	// With alpha=1.0, smoothed = raw metric each step.
	// best starts at 1.0 (init). Improvement requires best-smoothed > 0.1.
	steps := []struct {
		metric float64
		want   bool
	}{
		{1.0, false},  // init: best=1.0
		{0.95, false}, // delta=0.05 < 0.1 → no improve (count=1)
		{0.92, false}, // delta=0.08 < 0.1 → no improve (count=2)
		{0.905, true}, // delta=0.095 < 0.1 → no improve (count=3 >= patience)
	}

	for i, s := range steps {
		got := es.Step(s.metric)
		if got != s.want {
			t.Fatalf("step %d: metric=%f, got stop=%v, want %v", i, s.metric, got, s.want)
		}
	}
}

func TestSmoothedEarlyStopping_Reset(t *testing.T) {
	es := NewEarlyStopping(EarlyStopConfig{
		Patience: 2,
		Alpha:    0.5,
		MinDelta: 0.001,
		Mode:     "min",
	})

	// Drive to near-stop.
	es.Step(1.0)
	es.Step(1.0)
	es.Step(1.0)

	es.Reset()

	// After reset, should behave like a fresh instance.
	if es.initialized {
		t.Fatal("initialized should be false after reset")
	}
	if es.BestMetric() != 0 {
		t.Fatalf("BestMetric should be 0 after reset, got %f", es.BestMetric())
	}

	// First step after reset should not trigger.
	if es.Step(0.5) {
		t.Fatal("early stop triggered on first step after reset")
	}
	if math.Abs(es.BestMetric()-0.5) > 1e-9 {
		t.Fatalf("BestMetric should be 0.5 after first step, got %f", es.BestMetric())
	}
}

func TestNewEarlyStopping_Defaults(t *testing.T) {
	es := NewEarlyStopping(EarlyStopConfig{Patience: 5})
	if es.config.Alpha != 0.1 {
		t.Fatalf("expected default alpha 0.1, got %f", es.config.Alpha)
	}
	if es.config.Mode != "min" {
		t.Fatalf("expected default mode 'min', got %q", es.config.Mode)
	}
}
