package tabular

import (
	"math"
	"testing"
)

func TestEvaluationFrozenConfusionMatrix(t *testing.T) {
	truth := []int{0, 0, 0, 1, 1, 2, 2, 2}
	predicted := []int{0, 0, 1, 1, 2, 0, 2, 2}
	predictions := make([]Prediction, len(truth))
	for i, class := range predicted {
		probabilities := make([]float64, 3)
		probabilities[class] = 1
		predictions[i] = Prediction{ClassID: class, Probabilities: probabilities}
	}
	report, err := EvaluatePredictions(predictions, truth, 3)
	if err != nil {
		t.Fatal(err)
	}
	if report.Samples != 8 || math.Abs(report.Accuracy-0.625) > 1e-12 || math.Abs(report.MacroF1-0.6111111111111112) > 1e-12 {
		t.Fatalf("metric fixture mismatch: %+v", report)
	}
	expectedCE := -3 * math.Log(1e-7) / 8
	if math.Abs(report.CrossEntropy-expectedCE) > 1e-12 {
		t.Fatalf("cross entropy %g want %g", report.CrossEntropy, expectedCE)
	}
	predictions[0].ClassID = 2
	if _, err := EvaluatePredictions(predictions, truth, 3); err == nil {
		t.Fatal("inconsistent predictions accepted")
	}
}
