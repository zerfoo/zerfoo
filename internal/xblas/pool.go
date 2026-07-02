package xblas

import "github.com/zerfoo/zerfoo/internal/workerpool"

var defaultPool *workerpool.Pool

// InitPool creates the shared worker pool used by parallel GEMV routines.
// Call once from the engine constructor. n is the number of workers.
func InitPool(n int) {
	if defaultPool != nil {
		return
	}
	defaultPool = workerpool.New(n)
}

// ShutdownPool closes the shared worker pool. Safe to call if not initialized.
func ShutdownPool() {
	if defaultPool != nil {
		defaultPool.Close()
		defaultPool = nil
	}
}
