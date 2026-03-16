//go:build !unix

package inference

import (
	"fmt"
	"io"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model"
)

// loadZMFWithMmap is not supported on non-unix platforms.
func loadZMFWithMmap(
	_ compute.Engine[float32],
	_ string,
	_ []model.BuildOption,
) (*model.Model[float32], io.Closer, error) {
	return nil, nil, fmt.Errorf("mmap loading is not supported on this platform")
}
