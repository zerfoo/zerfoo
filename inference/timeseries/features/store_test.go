package features

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// writeCSV creates a temporary CSV file with the given number of rows.
// Each row has a timestamp incrementing by 1 minute and two feature columns.
func writeCSV(t *testing.T, n int, baseTime time.Time) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "features.csv")

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	fmt.Fprintln(f, "timestamp,f1,f2")
	for i := 0; i < n; i++ {
		ts := baseTime.Add(time.Duration(i) * time.Minute)
		fmt.Fprintf(f, "%s,%.1f,%.1f\n", ts.Format(time.RFC3339), float64(i)*1.0, float64(i)*2.0)
	}
	return path
}

func TestLoadOffline(t *testing.T) {
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	path := writeCSV(t, 1000, base)

	fs := NewFeatureStore()
	start := base.Add(100 * time.Minute)
	end := base.Add(199 * time.Minute)

	if err := fs.LoadOffline("AAPL", path, start, end); err != nil {
		t.Fatalf("LoadOffline: %v", err)
	}

	// asOf far in the future returns all loaded data.
	ticks, err := fs.GetFeatures("AAPL", end.Add(time.Hour))
	if err != nil {
		t.Fatalf("GetFeatures: %v", err)
	}
	if got := len(ticks); got != 100 {
		t.Fatalf("expected 100 ticks after filtering [100,199], got %d", got)
	}

	// Verify first and last tick timestamps.
	if !ticks[0].Timestamp.Equal(start) {
		t.Errorf("first tick = %v, want %v", ticks[0].Timestamp, start)
	}
	if !ticks[99].Timestamp.Equal(end) {
		t.Errorf("last tick = %v, want %v", ticks[99].Timestamp, end)
	}

	// Verify feature values for first tick (i=100).
	if ticks[0].Features[0] != 100.0 || ticks[0].Features[1] != 200.0 {
		t.Errorf("first tick features = %v, want [100 200]", ticks[0].Features)
	}
}

func TestLoadOfflineEndBeforeStart(t *testing.T) {
	fs := NewFeatureStore()
	err := fs.LoadOffline("X", "unused.csv",
		time.Date(2026, 2, 1, 0, 0, 0, 0, time.UTC),
		time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC))
	if err == nil {
		t.Fatal("expected error when end < start")
	}
}

func TestUpdateOnlineRingBufferCap(t *testing.T) {
	fs := NewFeatureStore()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// Insert 600 ticks.
	for i := 0; i < 600; i++ {
		tick := Tick{
			Timestamp: base.Add(time.Duration(i) * time.Minute),
			Features:  []float64{float64(i)},
		}
		if err := fs.UpdateOnline("BTC", tick); err != nil {
			t.Fatalf("UpdateOnline(%d): %v", i, err)
		}
	}

	// Get all — should be capped at 500.
	ticks, err := fs.GetFeatures("BTC", base.Add(700*time.Minute))
	if err != nil {
		t.Fatalf("GetFeatures: %v", err)
	}
	if got := len(ticks); got != 500 {
		t.Fatalf("expected ring buffer to cap at 500, got %d", got)
	}

	// Oldest should be tick 100 (first 100 evicted).
	if ticks[0].Features[0] != 100.0 {
		t.Errorf("oldest tick feature = %v, want 100", ticks[0].Features[0])
	}
	// Newest should be tick 599.
	if ticks[499].Features[0] != 599.0 {
		t.Errorf("newest tick feature = %v, want 599", ticks[499].Features[0])
	}
}

func TestGetFeaturesPointInTime(t *testing.T) {
	fs := NewFeatureStore()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// Load offline data: 100 ticks, 1 per minute.
	path := writeCSV(t, 100, base)
	if err := fs.LoadOffline("ETH", path, base, base.Add(99*time.Minute)); err != nil {
		t.Fatalf("LoadOffline: %v", err)
	}

	// Add 50 online ticks starting at minute 100.
	for i := 100; i < 150; i++ {
		tick := Tick{
			Timestamp: base.Add(time.Duration(i) * time.Minute),
			Features:  []float64{float64(i)},
		}
		if err := fs.UpdateOnline("ETH", tick); err != nil {
			t.Fatalf("UpdateOnline: %v", err)
		}
	}

	// asOf at minute 50 — should return only ticks 0..50 (51 ticks).
	asOf := base.Add(50 * time.Minute)
	ticks, err := fs.GetFeatures("ETH", asOf)
	if err != nil {
		t.Fatalf("GetFeatures: %v", err)
	}
	if got := len(ticks); got != 51 {
		t.Fatalf("point-in-time at minute 50: expected 51 ticks, got %d", got)
	}

	// No tick should have a timestamp after asOf.
	for _, tick := range ticks {
		if tick.Timestamp.After(asOf) {
			t.Errorf("future data leaked: tick at %v, asOf %v", tick.Timestamp, asOf)
		}
	}

	// asOf far in future — returns all 150 ticks.
	all, err := fs.GetFeatures("ETH", base.Add(1000*time.Minute))
	if err != nil {
		t.Fatalf("GetFeatures (all): %v", err)
	}
	if got := len(all); got != 150 {
		t.Fatalf("expected 150 total ticks, got %d", got)
	}
}

func TestGetFeaturesEmptyAsset(t *testing.T) {
	fs := NewFeatureStore()
	ticks, err := fs.GetFeatures("UNKNOWN", time.Now())
	if err != nil {
		t.Fatalf("GetFeatures on unknown asset: %v", err)
	}
	if len(ticks) != 0 {
		t.Errorf("expected 0 ticks for unknown asset, got %d", len(ticks))
	}
}
