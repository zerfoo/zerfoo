package attention

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// QKNorm applies a form of normalization to Query (Q) and Key (K) tensors
// to stabilize attention score scales, similar to RMSNorm.
// It normalizes Q and K independently by their respective RMS values.
func QKNorm[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], q, k *tensor.TensorNumeric[T], epsilon float64) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], error) {
	if q == nil {
		return nil, nil, fmt.Errorf("query tensor (q) cannot be nil")
	}
	if k == nil {
		return nil, nil, fmt.Errorf("key tensor (k) cannot be nil")
	}
	if !q.ShapeEquals(k) {
		return nil, nil, fmt.Errorf("query and key tensors must have the same shape: q %v, k %v", q.Shape(), k.Shape())
	}

	// Normalize Q
	qData := q.Data()
	sumSqQ := float64(0.0)
	for _, v := range qData {
		sumSqQ += float64(v * v)
	}
	rmsQ := math.Sqrt(sumSqQ/float64(q.Size()) + epsilon)

	normalizedQData := make([]T, q.Size())
	for i, v := range qData {
		normalizedQData[i] = T(float64(v) / rmsQ)
	}
	normalizedQ, err := tensor.New[T](q.Shape(), normalizedQData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create normalized Q tensor: %w", err)
	}

	// Normalize K
	kData := k.Data()
	sumSqK := float64(0.0)
	for _, v := range kData {
		sumSqK += float64(v * v)
	}
	rmsK := math.Sqrt(sumSqK/float64(k.Size()) + epsilon)

	normalizedKData := make([]T, k.Size())
	for i, v := range kData {
		normalizedKData[i] = T(float64(v) / rmsK)
	}
	normalizedK, err := tensor.New[T](k.Shape(), normalizedKData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create normalized K tensor: %w", err)
	}

	return normalizedQ, normalizedK, nil
}
