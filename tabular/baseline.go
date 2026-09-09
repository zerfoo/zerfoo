package tabular

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// NewMajorityClassifier fits a constant class-prior baseline using training
// labels only. Its argmax is the majority class (lowest ID breaks ties).
// Zero weights and log-prior biases allow the same GGUF deployment contract.
func NewMajorityClassifier[T tensor.Float](ctx context.Context, dataset *Dataset, engine compute.Engine[T]) (*Classifier[T], error) {
	if dataset == nil {
		return nil, fmt.Errorf("tabular: baseline requires dataset")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m := dataset.Manifest()
	_, labels, err := dataset.Partition("train")
	if err != nil {
		return nil, err
	}
	model, err := NewClassifier(ClassifierConfig{InputDim: len(m.Options.Features), ClassCount: len(m.Labels), Labels: m.Labels}, engine)
	if err != nil {
		return nil, err
	}
	counts := make([]int, len(m.Labels))
	for _, label := range labels {
		counts[label]++
	}
	bias := make([]T, len(counts))
	for i, count := range counts {
		if count == 0 {
			return nil, fmt.Errorf("tabular: baseline training class absent")
		}
		bias[i] = T(math.Log(float64(count) / float64(len(labels))))
	}
	model.params[0].Value.SetData(make([]T, model.config.InputDim*model.config.ClassCount))
	model.params[1].Value.SetData(bias)
	return model, nil
}
