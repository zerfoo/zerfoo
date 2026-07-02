package autoopt

import (
	"errors"
	"fmt"
	"sync"
)

// DeviceType identifies the accelerator backend.
type DeviceType string

const (
	DeviceCUDA  DeviceType = "cuda"
	DeviceROCm  DeviceType = "rocm"
	DeviceMetal DeviceType = "metal"
	DeviceCPU   DeviceType = "cpu"
)

// AcceleratorInfo describes a single accelerator device available for scheduling.
type AcceleratorInfo struct {
	// DeviceID is a unique identifier for this device.
	DeviceID int

	// Type is the accelerator backend (cuda, rocm, metal, cpu).
	Type DeviceType

	// AvailableMemory is the amount of free memory in bytes.
	AvailableMemory int64

	// Utilization is the current device load from 0.0 (idle) to 1.0 (fully loaded).
	Utilization float64

	// QueueDepth is the number of workloads currently queued on this device.
	QueueDepth int
}

// Workload describes a unit of work to be scheduled on an accelerator.
type Workload struct {
	// ID uniquely identifies this workload.
	ID string

	// EstimatedFLOPS is the estimated floating-point operations required.
	EstimatedFLOPS float64

	// MemoryRequired is the memory needed in bytes.
	MemoryRequired int64

	// PreferredDevice is the preferred device type (optional, empty means no preference).
	PreferredDevice DeviceType
}

// Migration describes a suggested workload migration from an overloaded device
// to an underloaded one.
type Migration struct {
	// WorkloadID identifies the workload to migrate.
	WorkloadID string

	// FromDevice is the source device ID.
	FromDevice int

	// ToDevice is the destination device ID.
	ToDevice int

	// Reason explains why this migration was suggested.
	Reason string
}

// SchedulingStrategy selects a device for a given workload from a set of accelerators.
type SchedulingStrategy interface {
	// Select picks the best device index (into the accelerators slice) for the workload.
	// Returns -1 if no suitable device is found.
	Select(accelerators []AcceleratorInfo, workload Workload) int
}

// RoundRobin cycles through devices in order, skipping devices that lack
// sufficient memory for the workload.
type RoundRobin struct {
	mu      sync.Mutex
	counter int
}

// Select picks the next device in round-robin order that has enough memory.
func (rr *RoundRobin) Select(accelerators []AcceleratorInfo, workload Workload) int {
	n := len(accelerators)
	if n == 0 {
		return -1
	}

	rr.mu.Lock()
	start := rr.counter
	rr.counter++
	rr.mu.Unlock()

	for i := 0; i < n; i++ {
		idx := (start + i) % n
		if accelerators[idx].AvailableMemory >= workload.MemoryRequired {
			return idx
		}
	}
	return -1
}

// LoadBalanced assigns workloads to the device with the lowest combined
// utilization and queue depth, filtering out devices without enough memory.
type LoadBalanced struct{}

// Select picks the device with the lowest load score that has enough memory.
func (lb *LoadBalanced) Select(accelerators []AcceleratorInfo, workload Workload) int {
	best := -1
	bestScore := float64(1<<63 - 1)

	for i, acc := range accelerators {
		if acc.AvailableMemory < workload.MemoryRequired {
			continue
		}
		// Score combines utilization and queue depth (queue depth scaled to [0,1] range).
		score := acc.Utilization + float64(acc.QueueDepth)*0.1
		if score < bestScore {
			bestScore = score
			best = i
		}
	}
	return best
}

// Priority prefers devices of specific types in a defined order. Within the
// same device type, it falls back to load-balanced selection.
type Priority struct {
	// Order lists device types from most preferred to least preferred.
	Order []DeviceType
}

// Select picks the first device matching the highest-priority type that has
// enough memory and lowest load.
func (p *Priority) Select(accelerators []AcceleratorInfo, workload Workload) int {
	lb := &LoadBalanced{}

	for _, dt := range p.Order {
		// Build a filtered view of matching devices.
		var filtered []AcceleratorInfo
		var indices []int
		for i, acc := range accelerators {
			if acc.Type == dt {
				filtered = append(filtered, acc)
				indices = append(indices, i)
			}
		}
		if len(filtered) == 0 {
			continue
		}
		idx := lb.Select(filtered, workload)
		if idx >= 0 {
			return indices[idx]
		}
	}
	return -1
}

// Scheduler manages workload scheduling across multiple accelerators.
type Scheduler struct {
	mu           sync.RWMutex
	accelerators []AcceleratorInfo
	strategy     SchedulingStrategy
	// assignments tracks which workloads are on which devices.
	assignments map[string]int
}

// NewScheduler creates a scheduler with the given accelerators and strategy.
func NewScheduler(accelerators []AcceleratorInfo, strategy SchedulingStrategy) *Scheduler {
	accs := make([]AcceleratorInfo, len(accelerators))
	copy(accs, accelerators)
	return &Scheduler{
		accelerators: accs,
		strategy:     strategy,
		assignments:  make(map[string]int),
	}
}

// Schedule picks the best device for the given workload and records the assignment.
func (s *Scheduler) Schedule(workload Workload) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	idx := s.strategy.Select(s.accelerators, workload)
	if idx < 0 {
		return -1, errors.New("no suitable device found for workload")
	}

	deviceID := s.accelerators[idx].DeviceID
	s.assignments[workload.ID] = deviceID
	s.accelerators[idx].QueueDepth++
	return deviceID, nil
}

// UpdateUtilization sets the utilization for the device with the given ID.
func (s *Scheduler) UpdateUtilization(deviceID int, utilization float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i := range s.accelerators {
		if s.accelerators[i].DeviceID == deviceID {
			s.accelerators[i].Utilization = utilization
			return
		}
	}
}

// AutoMigrate suggests migrations from devices above the threshold to the
// least loaded device. The threshold is a utilization value (0.0-1.0).
func (s *Scheduler) AutoMigrate(threshold float64) []Migration {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Find the least loaded device.
	minUtil := float64(2.0)
	minIdx := -1
	for i, acc := range s.accelerators {
		if acc.Utilization < minUtil {
			minUtil = acc.Utilization
			minIdx = i
		}
	}
	if minIdx < 0 {
		return nil
	}

	target := s.accelerators[minIdx]

	// If the target is also above threshold, no useful migration is possible.
	if target.Utilization >= threshold {
		return nil
	}

	var migrations []Migration
	for wID, dID := range s.assignments {
		if dID == target.DeviceID {
			continue
		}
		// Find the source device utilization.
		for _, acc := range s.accelerators {
			if acc.DeviceID == dID && acc.Utilization > threshold {
				migrations = append(migrations, Migration{
					WorkloadID: wID,
					FromDevice: dID,
					ToDevice:   target.DeviceID,
					Reason: fmt.Sprintf(
						"device %d utilization %.2f exceeds threshold %.2f; migrating to device %d (utilization %.2f)",
						dID, acc.Utilization, threshold, target.DeviceID, target.Utilization,
					),
				})
				break
			}
		}
	}
	return migrations
}
