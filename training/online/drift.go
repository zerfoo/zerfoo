package online

import (
	"math"
	"sync"
	"time"
)

// DriftAlert is emitted when the current window Sharpe ratio falls below
// the rolling mean minus one standard deviation.
type DriftAlert struct {
	// Timestamp is when the alert was raised.
	Timestamp time.Time

	// CurrentSharpe is the Sharpe ratio of the current window.
	CurrentSharpe float64

	// MeanSharpe is the rolling mean of historical Sharpe ratios.
	MeanSharpe float64

	// Threshold is mean - 1 sigma; the alert fires when CurrentSharpe < Threshold.
	Threshold float64

	// WindowSize is the number of P&L observations in the rolling window.
	WindowSize int
}

// DriftConfig holds parameters for the DriftDetector.
type DriftConfig struct {
	// WindowSize is the number of daily P&L observations in the rolling window.
	// Defaults to 30 if zero.
	WindowSize int
}

// DriftDetector computes a rolling Sharpe ratio over a configurable window
// of daily P&L observations and raises an alert when the current Sharpe ratio
// drops below the historical mean minus one standard deviation.
type DriftDetector struct {
	mu         sync.Mutex
	windowSize int
	pnls       []float64
	sharpes    []float64
}

// NewDriftDetector creates a new DriftDetector with the given configuration.
func NewDriftDetector(cfg DriftConfig) *DriftDetector {
	ws := cfg.WindowSize
	if ws <= 0 {
		ws = 30
	}
	return &DriftDetector{
		windowSize: ws,
	}
}

// Observe records a daily P&L value and returns a DriftAlert if the current
// window Sharpe ratio has dropped below the rolling mean minus one sigma.
// Returns nil when there is no alert or insufficient data.
func (d *DriftDetector) Observe(date time.Time, pnl float64) *DriftAlert {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.pnls = append(d.pnls, pnl)

	// Need at least windowSize observations to compute Sharpe.
	if len(d.pnls) < d.windowSize {
		return nil
	}

	// Trim to keep at most windowSize P&L values.
	if len(d.pnls) > d.windowSize {
		d.pnls = d.pnls[len(d.pnls)-d.windowSize:]
	}

	// Compute returns from P&L series.
	returns := make([]float64, len(d.pnls)-1)
	for i := 1; i < len(d.pnls); i++ {
		returns[i-1] = d.pnls[i] - d.pnls[i-1]
	}

	sharpe := annualizedSharpe(returns)
	d.sharpes = append(d.sharpes, sharpe)

	// Cap sharpes history to 10x the window size to bound memory usage
	// while retaining enough history for meaningful statistics.
	maxSharpes := d.windowSize * 10
	if len(d.sharpes) > maxSharpes {
		d.sharpes = d.sharpes[len(d.sharpes)-maxSharpes:]
	}

	// Need at least 3 historical Sharpe values to compute a meaningful
	// standard deviation for the threshold.
	if len(d.sharpes) < 3 {
		return nil
	}

	meanS, stdS := meanStd(d.sharpes[:len(d.sharpes)-1])
	threshold := meanS - stdS

	if sharpe < threshold {
		return &DriftAlert{
			Timestamp:     date,
			CurrentSharpe: sharpe,
			MeanSharpe:    meanS,
			Threshold:     threshold,
			WindowSize:    d.windowSize,
		}
	}
	return nil
}

// CurrentSharpe returns the Sharpe ratio of the current window.
// Returns 0 if there is insufficient data.
func (d *DriftDetector) CurrentSharpe() float64 {
	d.mu.Lock()
	defer d.mu.Unlock()

	if len(d.pnls) < 2 {
		return 0
	}

	returns := make([]float64, len(d.pnls)-1)
	for i := 1; i < len(d.pnls); i++ {
		returns[i-1] = d.pnls[i] - d.pnls[i-1]
	}
	return annualizedSharpe(returns)
}

// Window returns a copy of the current P&L window.
func (d *DriftDetector) Window() []float64 {
	d.mu.Lock()
	defer d.mu.Unlock()

	out := make([]float64, len(d.pnls))
	copy(out, d.pnls)
	return out
}

// annualizedSharpe computes mean(returns) / std(returns) * sqrt(252).
func annualizedSharpe(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	mean, std := meanStd(returns)
	if std == 0 {
		return 0
	}
	return (mean / std) * math.Sqrt(252.0)
}

// meanStd computes the population mean and standard deviation of xs.
func meanStd(xs []float64) (float64, float64) {
	n := float64(len(xs))
	if n == 0 {
		return 0, 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean := sum / n

	var sumSq float64
	for _, x := range xs {
		d := x - mean
		sumSq += d * d
	}
	std := math.Sqrt(sumSq / n)
	return mean, std
}
