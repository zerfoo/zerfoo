package inference

import "testing"

func TestExpertPlacementPolicy(t *testing.T) {
	tests := []struct {
		name         string
		numExperts   int
		threshold    float64
		routingStats map[int]float64
		wantDevices  map[int]DeviceType
	}{
		{
			name:       "shared experts always on GPU",
			numExperts: 4,
			threshold:  0.5,
			routingStats: map[int]float64{
				0: 1.0,
				1: 1.0,
				2: 0.3,
				3: 0.1,
			},
			wantDevices: map[int]DeviceType{
				0: GPU, 1: GPU, 2: CPU, 3: CPU,
			},
		},
		{
			name:       "frequency at threshold goes to GPU",
			numExperts: 3,
			threshold:  0.5,
			routingStats: map[int]float64{
				0: 0.5,
				1: 0.49,
				2: 0.51,
			},
			wantDevices: map[int]DeviceType{
				0: GPU, 1: CPU, 2: GPU,
			},
		},
		{
			name:       "missing stats default to CPU",
			numExperts: 3,
			threshold:  0.5,
			routingStats: map[int]float64{
				0: 0.8,
			},
			wantDevices: map[int]DeviceType{
				0: GPU, 1: CPU, 2: CPU,
			},
		},
		{
			name:       "custom threshold",
			numExperts: 4,
			threshold:  0.2,
			routingStats: map[int]float64{
				0: 0.1,
				1: 0.2,
				2: 0.3,
				3: 1.0,
			},
			wantDevices: map[int]DeviceType{
				0: CPU, 1: GPU, 2: GPU, 3: GPU,
			},
		},
		{
			name:         "all experts below threshold",
			numExperts:   3,
			threshold:    0.9,
			routingStats: map[int]float64{0: 0.1, 1: 0.2, 2: 0.3},
			wantDevices:  map[int]DeviceType{0: CPU, 1: CPU, 2: CPU},
		},
		{
			name:         "all experts above threshold",
			numExperts:   3,
			threshold:    0.1,
			routingStats: map[int]float64{0: 0.5, 1: 0.8, 2: 1.0},
			wantDevices:  map[int]DeviceType{0: GPU, 1: GPU, 2: GPU},
		},
		{
			name:         "empty routing stats",
			numExperts:   2,
			threshold:    0.5,
			routingStats: map[int]float64{},
			wantDevices:  map[int]DeviceType{0: CPU, 1: CPU},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewExpertPlacementPolicy(tt.numExperts, WithThreshold(tt.threshold))
			placements := p.Assign(tt.routingStats)

			if got, want := len(placements), tt.numExperts; got != want {
				t.Fatalf("len(placements) = %d, want %d", got, want)
			}

			for _, pl := range placements {
				want, ok := tt.wantDevices[pl.ExpertID]
				if !ok {
					t.Errorf("unexpected expert ID %d in placements", pl.ExpertID)
					continue
				}
				if pl.Device != want {
					t.Errorf("expert %d: device = %v, want %v (reason: %s)",
						pl.ExpertID, pl.Device, want, pl.Reason)
				}
			}

			dm := p.DeviceMap()
			if dm == nil {
				t.Fatal("DeviceMap() returned nil after Assign")
			}
			for id, want := range tt.wantDevices {
				if got := dm[id]; got != want {
					t.Errorf("DeviceMap()[%d] = %v, want %v", id, got, want)
				}
			}
		})
	}
}

func TestExpertPlacementPolicyDefaultThreshold(t *testing.T) {
	p := NewExpertPlacementPolicy(2)
	p.Assign(map[int]float64{0: 0.5, 1: 0.4})
	dm := p.DeviceMap()
	if dm[0] != GPU {
		t.Errorf("expert 0 with freq 0.5 should be GPU with default threshold 0.5")
	}
	if dm[1] != CPU {
		t.Errorf("expert 1 with freq 0.4 should be CPU with default threshold 0.5")
	}
}

func TestExpertPlacementPolicyDeviceMapBeforeAssign(t *testing.T) {
	p := NewExpertPlacementPolicy(4)
	if dm := p.DeviceMap(); dm != nil {
		t.Errorf("DeviceMap() before Assign() = %v, want nil", dm)
	}
}

func TestDeviceTypeString(t *testing.T) {
	if got := CPU.String(); got != "CPU" {
		t.Errorf("CPU.String() = %q, want %q", got, "CPU")
	}
	if got := GPU.String(); got != "GPU" {
		t.Errorf("GPU.String() = %q, want %q", got, "GPU")
	}
}

func TestExpertPlacementReasons(t *testing.T) {
	p := NewExpertPlacementPolicy(3, WithThreshold(0.5))
	placements := p.Assign(map[int]float64{0: 1.0, 1: 0.7, 2: 0.2})

	if placements[0].Reason != "shared expert (always active)" {
		t.Errorf("expert 0: reason = %q, want shared expert reason", placements[0].Reason)
	}
	if placements[1].Reason != "routing frequency above threshold" {
		t.Errorf("expert 1: reason = %q, want above threshold reason", placements[1].Reason)
	}
	if placements[2].Reason != "routing frequency below threshold" {
		t.Errorf("expert 2: reason = %q, want below threshold reason", placements[2].Reason)
	}
}
