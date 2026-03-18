package rl

import (
	"testing"
)

// makeExp returns a minimal Experience with a distinguishable reward value.
func makeExp(reward float64) Experience {
	return Experience{
		State:     []float64{reward},
		Action:    []float64{0},
		Reward:    reward,
		NextState: []float64{reward + 1},
		Done:      false,
	}
}

func TestReplayBuffer_FIFO(t *testing.T) {
	tests := []struct {
		name     string
		capacity int
		add      []float64 // rewards to add in order
		wantSize int
		// wantOldest is the reward of the oldest entry after all adds.
		wantOldest float64
	}{
		{
			name:       "under capacity",
			capacity:   5,
			add:        []float64{1, 2, 3},
			wantSize:   3,
			wantOldest: 1,
		},
		{
			name:       "exactly full",
			capacity:   3,
			add:        []float64{1, 2, 3},
			wantSize:   3,
			wantOldest: 1,
		},
		{
			name:       "one over capacity overwrites oldest",
			capacity:   3,
			add:        []float64{1, 2, 3, 4},
			wantSize:   3,
			wantOldest: 2, // 1 was overwritten
		},
		{
			name:       "double capacity overwrites all original",
			capacity:   3,
			add:        []float64{1, 2, 3, 4, 5, 6},
			wantSize:   3,
			wantOldest: 4, // 1,2,3 all gone
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rb := NewReplayBuffer(tc.capacity)
			for _, r := range tc.add {
				rb.Add(makeExp(r))
			}
			if rb.Len() != tc.wantSize {
				t.Errorf("Len() = %d, want %d", rb.Len(), tc.wantSize)
			}
			// Oldest entry is at logical index 0.
			oldest := (rb.head - rb.size + rb.capacity) % rb.capacity
			got := rb.buf[oldest].Reward
			if got != tc.wantOldest {
				t.Errorf("oldest reward = %v, want %v", got, tc.wantOldest)
			}
		})
	}
}

func TestReplayBuffer_PrioritySampling(t *testing.T) {
	// Fill a buffer with experiences reward 0..4.
	rb := NewReplayBuffer(5)
	for i := 0; i < 5; i++ {
		rb.Add(makeExp(float64(i)))
	}

	// Give all weight to the last entry (reward=4).
	priorities := []float64{0, 0, 0, 0, 1e9}
	const n = 1000
	batch := rb.SamplePrioritized(n, priorities)
	if len(batch) != n {
		t.Fatalf("SamplePrioritized returned %d items, want %d", len(batch), n)
	}
	for _, exp := range batch {
		if exp.Reward != 4 {
			t.Errorf("expected all samples to have reward=4 (highest priority), got %v", exp.Reward)
			break
		}
	}
}

func TestReplayBuffer_PrioritySampling_ZeroWeightsFallbackUniform(t *testing.T) {
	rb := NewReplayBuffer(3)
	for i := 0; i < 3; i++ {
		rb.Add(makeExp(float64(i)))
	}
	priorities := []float64{0, 0, 0}
	batch := rb.SamplePrioritized(30, priorities)
	if len(batch) != 30 {
		t.Fatalf("expected 30 samples, got %d", len(batch))
	}
	// Just verify no panic and rewards are within [0,2].
	for _, exp := range batch {
		if exp.Reward < 0 || exp.Reward > 2 {
			t.Errorf("unexpected reward %v", exp.Reward)
		}
	}
}

func TestReplayBuffer_PrioritiesLengthMismatchPanics(t *testing.T) {
	rb := NewReplayBuffer(3)
	rb.Add(makeExp(1))
	rb.Add(makeExp(2))

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on priorities length mismatch, got none")
		}
	}()
	rb.SamplePrioritized(1, []float64{1, 2, 3}) // len=3 but rb.Len()=2
}

func TestReplayBuffer_SampleEmpty(t *testing.T) {
	rb := NewReplayBuffer(4)
	if got := rb.Sample(5); got != nil {
		t.Errorf("Sample on empty buffer should return nil, got %v", got)
	}
	if got := rb.SamplePrioritized(5, nil); got != nil {
		t.Errorf("SamplePrioritized on empty buffer should return nil, got %v", got)
	}
}
