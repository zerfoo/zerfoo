package tabular

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestMajorityClassifierUsesTrainingPriors(t *testing.T) {
	dataset := classifierFixture(t)
	_, labels, err := dataset.Partition("train")
	if err != nil {
		t.Fatal(err)
	}
	counts := make([]int, 3)
	for _, label := range labels {
		counts[label]++
	}
	model, err := NewMajorityClassifier(context.Background(), dataset, compute.NewCPUEngine(numeric.Float32Ops{}))
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range [][]float64{{0, 0, 0, 0}, {100, -100, 12, 4}} {
		prediction, err := model.Predict(context.Background(), row)
		if err != nil {
			t.Fatal(err)
		}
		if prediction.ClassID != 0 {
			t.Fatal("balanced tie must select lowest class")
		}
		for i, p := range prediction.Probabilities {
			if math.Abs(p-float64(counts[i])/float64(len(labels))) > 1e-6 {
				t.Fatal("incorrect training prior")
			}
		}
	}
}
