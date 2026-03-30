package timeseries

import (
	its "github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// NewTestForecaster creates a FoundationForecaster with a stub graph for use
// in tests outside this package. numVars is the expected number of input
// variates and horizon is the maximum forecast horizon.
func NewTestForecaster(numVars, horizon int) (*FoundationForecaster, error) {
	cfg := &its.TiRexConfig{
		NumLayers: 1,
		InputDim:  numVars,
		HiddenDim: 4,
		Horizon:   horizon,
		NumVars:   numVars,
	}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	return newFoundationForecasterFromConfig(cfg, engine)
}
