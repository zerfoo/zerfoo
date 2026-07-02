package online

import (
	"math"
	"testing"
	"time"
)

func TestDriftDetector_NoAlertWithStablePnL(t *testing.T) {
	dd := NewDriftDetector(DriftConfig{
		WindowSize: 30,
	})

	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	// Feed 60 days of steadily increasing P&L (consistent positive returns).
	for i := 0; i < 60; i++ {
		pnl := 100.0 + float64(i)*1.0 // monotonically increasing
		alert := dd.Observe(base.Add(time.Duration(i)*24*time.Hour), pnl)
		if alert != nil {
			t.Fatalf("unexpected alert on day %d: %+v", i, alert)
		}
	}
}

func TestDriftDetector_AlertOnDegradation(t *testing.T) {
	dd := NewDriftDetector(DriftConfig{
		WindowSize: 30,
	})

	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	day := 0

	// Phase 1: 90 days of steadily increasing P&L to build history.
	for ; day < 90; day++ {
		pnl := 100.0 + float64(day)*1.0 // monotonically increasing
		dd.Observe(base.Add(time.Duration(day)*24*time.Hour), pnl)
	}

	// Phase 2: inject degradation — negative P&L with high variance.
	var gotAlert *DriftAlert
	for ; day < 120; day++ {
		pnl := -50.0 + float64(day%5)*10.0 // bad performance
		alert := dd.Observe(base.Add(time.Duration(day)*24*time.Hour), pnl)
		if alert != nil {
			gotAlert = alert
		}
	}

	if gotAlert == nil {
		t.Fatal("expected drift alert after injecting degradation, got none")
	}
	if gotAlert.CurrentSharpe >= gotAlert.Threshold {
		t.Errorf("alert CurrentSharpe (%f) should be < Threshold (%f)",
			gotAlert.CurrentSharpe, gotAlert.Threshold)
	}
	if gotAlert.WindowSize != 30 {
		t.Errorf("alert WindowSize = %d, want 30", gotAlert.WindowSize)
	}
}

func TestDriftDetector_InsufficientData(t *testing.T) {
	dd := NewDriftDetector(DriftConfig{
		WindowSize: 30,
	})

	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	// Only 29 observations — not enough for a full window.
	for i := 0; i < 29; i++ {
		alert := dd.Observe(base.Add(time.Duration(i)*24*time.Hour), -1000.0)
		if alert != nil {
			t.Fatalf("should not alert with insufficient data on day %d", i)
		}
	}
}

func TestDriftDetector_SharpeComputation(t *testing.T) {
	dd := NewDriftDetector(DriftConfig{
		WindowSize: 5,
	})

	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	pnls := []float64{100, 102, 98, 104, 96}
	for i, pnl := range pnls {
		dd.Observe(base.Add(time.Duration(i)*24*time.Hour), pnl)
	}

	// Returns from P&L: 2, -4, 6, -8
	// mean = (2 + -4 + 6 + -8) / 4 = -1
	// variance = ((2-(-1))^2 + (-4-(-1))^2 + (6-(-1))^2 + (-8-(-1))^2) / 4
	//          = (9 + 9 + 49 + 49) / 4 = 29
	// std = sqrt(29) ≈ 5.385
	// Sharpe = (-1 / 5.385) * sqrt(252) ≈ -2.948
	sharpe := dd.CurrentSharpe()
	expected := (-1.0 / math.Sqrt(29.0)) * math.Sqrt(252.0)
	if math.Abs(sharpe-expected) > 0.001 {
		t.Errorf("CurrentSharpe() = %f, want %f", sharpe, expected)
	}
}

func TestDriftDetector_RollingWindow(t *testing.T) {
	dd := NewDriftDetector(DriftConfig{
		WindowSize: 3,
	})

	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	// Feed 5 observations; window should only keep last 3.
	for i, pnl := range []float64{10, 20, 30, 40, 50} {
		dd.Observe(base.Add(time.Duration(i)*24*time.Hour), pnl)
	}

	window := dd.Window()
	if len(window) != 3 {
		t.Fatalf("window length = %d, want 3", len(window))
	}
	// Should contain the last 3 values: 30, 40, 50
	want := []float64{30, 40, 50}
	for i, v := range want {
		if window[i] != v {
			t.Errorf("window[%d] = %f, want %f", i, window[i], v)
		}
	}
}
