package training

// EarlyStopConfig configures smoothed early stopping behavior.
type EarlyStopConfig struct {
	// Patience is the number of epochs without improvement before stopping.
	Patience int
	// Alpha is the EMA smoothing factor (0 < alpha < 1). Default: 0.1.
	Alpha float64
	// MinDelta is the minimum improvement to count as progress.
	MinDelta float64
	// Mode is "min" for loss (lower is better) or "max" for accuracy (higher is better).
	Mode string
}

// EarlyStopping tracks a smoothed metric via exponential moving average
// and signals when training should stop due to lack of improvement.
type EarlyStopping struct {
	config          EarlyStopConfig
	smoothedMetric  float64
	bestSmoothed    float64
	epochsNoImprove int
	initialized     bool
}

// NewEarlyStopping creates a new EarlyStopping instance with the given config.
// If Alpha is zero, it defaults to 0.1. If Mode is empty, it defaults to "min".
func NewEarlyStopping(config EarlyStopConfig) *EarlyStopping {
	if config.Alpha == 0 {
		config.Alpha = 0.1
	}
	if config.Mode == "" {
		config.Mode = "min"
	}
	return &EarlyStopping{config: config}
}

// Step records a new metric value and returns true if training should stop.
// On the first call, it initializes the smoothed metric to the raw value.
// On subsequent calls, it applies EMA smoothing and checks for improvement.
func (es *EarlyStopping) Step(metric float64) bool {
	if !es.initialized {
		es.smoothedMetric = metric
		es.bestSmoothed = metric
		es.initialized = true
		return false
	}

	es.smoothedMetric = es.config.Alpha*metric + (1-es.config.Alpha)*es.smoothedMetric

	improved := false
	switch es.config.Mode {
	case "max":
		improved = es.smoothedMetric-es.bestSmoothed > es.config.MinDelta
	default: // "min"
		improved = es.bestSmoothed-es.smoothedMetric > es.config.MinDelta
	}

	if improved {
		es.bestSmoothed = es.smoothedMetric
		es.epochsNoImprove = 0
	} else {
		es.epochsNoImprove++
	}

	return es.epochsNoImprove >= es.config.Patience
}

// Reset clears all state so the instance can be reused.
func (es *EarlyStopping) Reset() {
	es.smoothedMetric = 0
	es.bestSmoothed = 0
	es.epochsNoImprove = 0
	es.initialized = false
}

// BestMetric returns the best smoothed metric observed so far.
func (es *EarlyStopping) BestMetric() float64 {
	return es.bestSmoothed
}
