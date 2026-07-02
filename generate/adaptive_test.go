package generate

import "testing"

func TestAdaptiveDraftLen_Initial(t *testing.T) {
	a := newAdaptiveDraftLen(4, 1, 8, 32)
	if got := a.Current(); got != 4 {
		t.Errorf("Current() = %d, want 4", got)
	}
	if got := a.Rate(); got != 1.0 {
		t.Errorf("Rate() = %f, want 1.0", got)
	}
}

func TestAdaptiveDraftLen_ClampInitial(t *testing.T) {
	// Initial below min.
	a := newAdaptiveDraftLen(0, 2, 8, 32)
	if got := a.Current(); got != 2 {
		t.Errorf("Current() = %d, want 2 (clamped to min)", got)
	}
	// Initial above max.
	b := newAdaptiveDraftLen(10, 1, 5, 32)
	if got := b.Current(); got != 5 {
		t.Errorf("Current() = %d, want 5 (clamped to max)", got)
	}
}

func TestAdaptiveDraftLen_HighAcceptanceIncreases(t *testing.T) {
	a := newAdaptiveDraftLen(4, 1, 8, 32)

	// Record 32 all-accepted batches to fill the window.
	for range 32 {
		a.Record(4, 4) // 100% acceptance
	}

	if got := a.Current(); got <= 4 {
		t.Errorf("Current() = %d, want > 4 after high acceptance", got)
	}
}

func TestAdaptiveDraftLen_LowAcceptanceDecreases(t *testing.T) {
	a := newAdaptiveDraftLen(4, 1, 8, 32)

	// Record batches with 0% acceptance to fill the window.
	for range 16 {
		a.Record(0, 4) // 0% acceptance
	}

	if got := a.Current(); got >= 4 {
		t.Errorf("Current() = %d, want < 4 after low acceptance", got)
	}
}

func TestAdaptiveDraftLen_StableNoChange(t *testing.T) {
	a := newAdaptiveDraftLen(4, 1, 8, 32)

	// 60% acceptance: between low (40%) and high (80%) thresholds.
	for range 20 {
		a.Record(3, 5) // 60% acceptance
	}

	if got := a.Current(); got != 4 {
		t.Errorf("Current() = %d, want 4 (stable region)", got)
	}
}

func TestAdaptiveDraftLen_MaxCap(t *testing.T) {
	a := newAdaptiveDraftLen(7, 1, 8, 32)

	// 100% acceptance, already near max.
	for range 64 {
		a.Record(4, 4)
	}

	if got := a.Current(); got > 8 {
		t.Errorf("Current() = %d, want <= 8 (max)", got)
	}
}

func TestAdaptiveDraftLen_MinFloor(t *testing.T) {
	a := newAdaptiveDraftLen(2, 1, 8, 32)

	// 0% acceptance to drive it down.
	for range 64 {
		a.Record(0, 4)
	}

	if got := a.Current(); got < 1 {
		t.Errorf("Current() = %d, want >= 1 (min)", got)
	}
}

func TestAdaptiveDraftLen_RingBufferWraparound(t *testing.T) {
	a := newAdaptiveDraftLen(4, 1, 8, 8) // small window

	// Fill with all accepted.
	for range 8 {
		a.Record(1, 1)
	}
	if r := a.Rate(); r != 1.0 {
		t.Errorf("Rate() = %f, want 1.0 after all accepted", r)
	}

	// Overwrite half with rejections.
	for range 4 {
		a.Record(0, 1)
	}
	// Window should have 4 accepted + 4 rejected = 50%.
	if r := a.Rate(); r < 0.49 || r > 0.51 {
		t.Errorf("Rate() = %f, want ~0.50", r)
	}
}
