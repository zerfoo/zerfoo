package timeseries

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func newTestEngine() (compute.Engine[float32], numeric.Arithmetic[float32]) {
	ops := numeric.Float32Ops{}
	return compute.NewCPUEngine[float32](ops), ops
}
