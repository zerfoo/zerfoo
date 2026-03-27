package inference

// DeviceType represents a compute device for expert placement.
type DeviceType int

const (
	// CPU indicates the expert should run on the CPU.
	CPU DeviceType = iota
	// GPU indicates the expert should run on the GPU.
	GPU
)

func (d DeviceType) String() string {
	switch d {
	case GPU:
		return "GPU"
	default:
		return "CPU"
	}
}

// ExpertPlacement describes the device assignment for a single expert.
type ExpertPlacement struct {
	ExpertID int
	Device   DeviceType
	Reason   string
}

// PlacementOption configures an [ExpertPlacementPolicy].
type PlacementOption func(*ExpertPlacementPolicy)

// WithThreshold sets the routing frequency threshold above which an expert
// is placed on GPU. The default threshold is 0.5.
func WithThreshold(t float64) PlacementOption {
	return func(p *ExpertPlacementPolicy) {
		p.threshold = t
	}
}

// ExpertPlacementPolicy decides which MoE experts run on GPU vs CPU based
// on routing frequency statistics. Shared experts (frequency 1.0) are always
// placed on GPU. Routed experts with frequency >= threshold go to GPU;
// the rest go to CPU.
type ExpertPlacementPolicy struct {
	threshold  float64
	numExperts int
	placements []ExpertPlacement
}

// NewExpertPlacementPolicy creates a policy for numExperts experts.
// The default threshold is 0.5; use [WithThreshold] to override.
func NewExpertPlacementPolicy(numExperts int, opts ...PlacementOption) *ExpertPlacementPolicy {
	p := &ExpertPlacementPolicy{
		threshold:  0.5,
		numExperts: numExperts,
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

// Assign computes device placements for all experts based on routingStats,
// which maps expert ID to activation frequency in [0.0, 1.0].
// Experts not present in routingStats are assigned to CPU.
func (p *ExpertPlacementPolicy) Assign(routingStats map[int]float64) []ExpertPlacement {
	p.placements = make([]ExpertPlacement, p.numExperts)
	for i := range p.numExperts {
		freq, ok := routingStats[i]
		switch {
		case ok && freq == 1.0:
			p.placements[i] = ExpertPlacement{
				ExpertID: i,
				Device:   GPU,
				Reason:   "shared expert (always active)",
			}
		case ok && freq >= p.threshold:
			p.placements[i] = ExpertPlacement{
				ExpertID: i,
				Device:   GPU,
				Reason:   "routing frequency above threshold",
			}
		default:
			p.placements[i] = ExpertPlacement{
				ExpertID: i,
				Device:   CPU,
				Reason:   "routing frequency below threshold",
			}
		}
	}
	return p.placements
}

// DeviceMap returns the current device assignments as a map from expert ID
// to [DeviceType]. Returns nil if [Assign] has not been called.
func (p *ExpertPlacementPolicy) DeviceMap() map[int]DeviceType {
	if p.placements == nil {
		return nil
	}
	m := make(map[int]DeviceType, len(p.placements))
	for _, pl := range p.placements {
		m[pl.ExpertID] = pl.Device
	}
	return m
}
