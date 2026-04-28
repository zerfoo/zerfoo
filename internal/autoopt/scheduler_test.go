package autoopt

import (
	"sort"
	"testing"
)

func threeDevices() []AcceleratorInfo {
	return []AcceleratorInfo{
		{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.2, QueueDepth: 1},
		{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.5, QueueDepth: 2},
		{DeviceID: 2, Type: DeviceROCm, AvailableMemory: 8 << 30, Utilization: 0.1, QueueDepth: 0},
	}
}

func TestScheduler_RoundRobin(t *testing.T) {
	accs := threeDevices()
	strategy := &RoundRobin{}
	s := NewScheduler(accs, strategy)

	workload := Workload{ID: "w", EstimatedFLOPS: 1e9, MemoryRequired: 1 << 30}

	tests := []struct {
		name     string
		wantDev  int
	}{
		{"first", 0},
		{"second", 1},
		{"third", 2},
		{"wraps to first", 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			workload.ID = tt.name
			deviceID, err := s.Schedule(workload)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if deviceID != tt.wantDev {
				t.Errorf("got device %d, want %d", deviceID, tt.wantDev)
			}
		})
	}
}

func TestScheduler_LoadBalanced(t *testing.T) {
	tests := []struct {
		name    string
		accs    []AcceleratorInfo
		wantDev int
	}{
		{
			name: "picks lowest utilization",
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.8, QueueDepth: 0},
				{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.1, QueueDepth: 0},
				{DeviceID: 2, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.5, QueueDepth: 0},
			},
			wantDev: 1,
		},
		{
			name: "factors in queue depth",
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.3, QueueDepth: 10},
				{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.5, QueueDepth: 0},
			},
			wantDev: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewScheduler(tt.accs, &LoadBalanced{})
			workload := Workload{ID: "w1", EstimatedFLOPS: 1e9, MemoryRequired: 1 << 30}
			deviceID, err := s.Schedule(workload)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if deviceID != tt.wantDev {
				t.Errorf("got device %d, want %d", deviceID, tt.wantDev)
			}
		})
	}
}

func TestScheduler_Priority(t *testing.T) {
	tests := []struct {
		name    string
		order   []DeviceType
		accs    []AcceleratorInfo
		wantDev int
	}{
		{
			name:  "prefers CUDA over ROCm over CPU",
			order: []DeviceType{DeviceCUDA, DeviceROCm, DeviceCPU},
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCPU, AvailableMemory: 32 << 30, Utilization: 0.0},
				{DeviceID: 1, Type: DeviceROCm, AvailableMemory: 16 << 30, Utilization: 0.1},
				{DeviceID: 2, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.5},
			},
			wantDev: 2,
		},
		{
			name:  "falls back when preferred type unavailable",
			order: []DeviceType{DeviceCUDA, DeviceMetal, DeviceCPU},
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCPU, AvailableMemory: 32 << 30, Utilization: 0.0},
				{DeviceID: 1, Type: DeviceMetal, AvailableMemory: 16 << 30, Utilization: 0.2},
			},
			wantDev: 1,
		},
		{
			name:  "load-balanced within same type",
			order: []DeviceType{DeviceCUDA},
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.9},
				{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.1},
			},
			wantDev: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := &Priority{Order: tt.order}
			s := NewScheduler(tt.accs, strategy)
			workload := Workload{ID: "w1", EstimatedFLOPS: 1e9, MemoryRequired: 1 << 30}
			deviceID, err := s.Schedule(workload)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if deviceID != tt.wantDev {
				t.Errorf("got device %d, want %d", deviceID, tt.wantDev)
			}
		})
	}
}

func TestScheduler_AutoMigrate(t *testing.T) {
	tests := []struct {
		name           string
		accs           []AcceleratorInfo
		assignments    map[string]int
		threshold      float64
		wantMigrations int
		wantToDevice   int
	}{
		{
			name: "overloaded device triggers migration",
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.95},
				{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.2},
			},
			assignments:    map[string]int{"w1": 0, "w2": 0},
			threshold:      0.8,
			wantMigrations: 2,
			wantToDevice:   1,
		},
		{
			name: "no migration when all below threshold",
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.3},
				{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.2},
			},
			assignments:    map[string]int{"w1": 0},
			threshold:      0.8,
			wantMigrations: 0,
		},
		{
			name: "no migration when all devices overloaded",
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.9},
				{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.85},
			},
			assignments:    map[string]int{"w1": 0},
			threshold:      0.8,
			wantMigrations: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewScheduler(tt.accs, &RoundRobin{})
			s.assignments = tt.assignments
			migrations := s.AutoMigrate(tt.threshold)
			if len(migrations) != tt.wantMigrations {
				t.Errorf("got %d migrations, want %d", len(migrations), tt.wantMigrations)
			}
			if tt.wantMigrations > 0 {
				for _, m := range migrations {
					if m.ToDevice != tt.wantToDevice {
						t.Errorf("migration to device %d, want %d", m.ToDevice, tt.wantToDevice)
					}
				}
			}
		})
	}
}

func TestScheduler_MemoryCheck(t *testing.T) {
	tests := []struct {
		name      string
		accs      []AcceleratorInfo
		memReq    int64
		wantError bool
	}{
		{
			name: "workload fits",
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.1},
			},
			memReq:    8 << 30,
			wantError: false,
		},
		{
			name: "workload exceeds all devices",
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 4 << 30, Utilization: 0.1},
				{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 6 << 30, Utilization: 0.1},
			},
			memReq:    8 << 30,
			wantError: true,
		},
		{
			name: "skips insufficient device picks sufficient one",
			accs: []AcceleratorInfo{
				{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 4 << 30, Utilization: 0.0},
				{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.5},
			},
			memReq:    8 << 30,
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewScheduler(tt.accs, &LoadBalanced{})
			workload := Workload{ID: "w1", EstimatedFLOPS: 1e9, MemoryRequired: tt.memReq}
			_, err := s.Schedule(workload)
			if tt.wantError && err == nil {
				t.Error("expected error but got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestScheduler_UpdateUtilization(t *testing.T) {
	accs := threeDevices()
	s := NewScheduler(accs, &LoadBalanced{})

	// Device 0 starts at 0.2 utilization. Update to 0.9.
	s.UpdateUtilization(0, 0.9)

	// Now schedule — should pick device 2 (0.1) not device 0 (0.9).
	workload := Workload{ID: "w1", EstimatedFLOPS: 1e9, MemoryRequired: 1 << 30}
	deviceID, err := s.Schedule(workload)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if deviceID != 2 {
		t.Errorf("got device %d, want 2 (lowest utilization after update)", deviceID)
	}
}

func TestScheduler_EmptyAccelerators(t *testing.T) {
	s := NewScheduler(nil, &RoundRobin{})
	workload := Workload{ID: "w1", EstimatedFLOPS: 1e9, MemoryRequired: 1 << 30}
	_, err := s.Schedule(workload)
	if err == nil {
		t.Error("expected error for empty accelerators, got nil")
	}
}

func TestScheduler_AutoMigrate_Deterministic(t *testing.T) {
	// Verify migrations are consistent and contain correct fields.
	accs := []AcceleratorInfo{
		{DeviceID: 10, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.95},
		{DeviceID: 20, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.1},
	}
	s := NewScheduler(accs, &RoundRobin{})
	s.assignments = map[string]int{"job-a": 10}

	migrations := s.AutoMigrate(0.8)
	if len(migrations) != 1 {
		t.Fatalf("got %d migrations, want 1", len(migrations))
	}

	m := migrations[0]
	if m.WorkloadID != "job-a" {
		t.Errorf("WorkloadID = %q, want %q", m.WorkloadID, "job-a")
	}
	if m.FromDevice != 10 {
		t.Errorf("FromDevice = %d, want 10", m.FromDevice)
	}
	if m.ToDevice != 20 {
		t.Errorf("ToDevice = %d, want 20", m.ToDevice)
	}
	if m.Reason == "" {
		t.Error("Reason should not be empty")
	}
}

func TestScheduler_RoundRobin_SkipsInsufficientMemory(t *testing.T) {
	accs := []AcceleratorInfo{
		{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 1 << 30, Utilization: 0.1},
		{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.1},
		{DeviceID: 2, Type: DeviceCUDA, AvailableMemory: 1 << 30, Utilization: 0.1},
	}
	s := NewScheduler(accs, &RoundRobin{})
	workload := Workload{ID: "w1", MemoryRequired: 8 << 30}

	// Should skip device 0 (1 GiB) and land on device 1 (16 GiB).
	deviceID, err := s.Schedule(workload)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if deviceID != 1 {
		t.Errorf("got device %d, want 1", deviceID)
	}
}

func TestScheduler_Priority_AllTypesExhausted(t *testing.T) {
	// All devices have insufficient memory for the workload.
	accs := []AcceleratorInfo{
		{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 2 << 30},
		{DeviceID: 1, Type: DeviceCPU, AvailableMemory: 4 << 30},
	}
	strategy := &Priority{Order: []DeviceType{DeviceCUDA, DeviceCPU}}
	s := NewScheduler(accs, strategy)
	workload := Workload{ID: "w1", MemoryRequired: 8 << 30}

	_, err := s.Schedule(workload)
	if err == nil {
		t.Error("expected error when all types exhausted, got nil")
	}
}

// TestMigration_SortDeterminism verifies migrations can be sorted for deterministic output.
func TestMigration_SortDeterminism(t *testing.T) {
	accs := []AcceleratorInfo{
		{DeviceID: 0, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.95},
		{DeviceID: 1, Type: DeviceCUDA, AvailableMemory: 16 << 30, Utilization: 0.05},
	}
	s := NewScheduler(accs, &RoundRobin{})
	s.assignments = map[string]int{"b": 0, "a": 0, "c": 0}

	migrations := s.AutoMigrate(0.8)
	sort.Slice(migrations, func(i, j int) bool {
		return migrations[i].WorkloadID < migrations[j].WorkloadID
	})

	if len(migrations) != 3 {
		t.Fatalf("got %d migrations, want 3", len(migrations))
	}
	for i, wantID := range []string{"a", "b", "c"} {
		if migrations[i].WorkloadID != wantID {
			t.Errorf("migrations[%d].WorkloadID = %q, want %q", i, migrations[i].WorkloadID, wantID)
		}
	}
}
