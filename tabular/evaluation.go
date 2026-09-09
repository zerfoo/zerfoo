package tabular

import (
	"context"
	"fmt"
	"math"
)

// Evaluation reports scalar metrics over exactly the supplied prediction rows.
type Evaluation struct {
	Samples         int     `json:"samples"`
	Accuracy        float64 `json:"accuracy"`
	MacroF1         float64 `json:"macro_f1"`
	CrossEntropy    float64 `json:"cross_entropy"`
	ConfusionMatrix [][]int `json:"confusion_matrix"`
}

// EvaluatePredictions uses true-class rows and predicted-class columns. Missing
// class F1 denominators contribute zero. Unknown/out-of-range predictions fail.
func EvaluatePredictions(predictions []Prediction, targets []int, classes int) (Evaluation, error) {
	if classes < 2 || len(predictions) == 0 || len(predictions) != len(targets) {
		return Evaluation{}, fmt.Errorf("tabular: invalid evaluation shape")
	}
	result := Evaluation{Samples: len(targets), ConfusionMatrix: make([][]int, classes)}
	for i := range result.ConfusionMatrix {
		result.ConfusionMatrix[i] = make([]int, classes)
	}
	correct := 0
	for i, p := range predictions {
		if targets[i] < 0 || targets[i] >= classes || p.ClassID < 0 || p.ClassID >= classes || len(p.Probabilities) != classes {
			return Evaluation{}, fmt.Errorf("tabular: invalid evaluation row %d", i)
		}
		sum := 0.0
		best := 0
		for j, value := range p.Probabilities {
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
				return Evaluation{}, fmt.Errorf("tabular: invalid probability at row %d", i)
			}
			sum += value
			if value > p.Probabilities[best] {
				best = j
			}
		}
		if math.Abs(sum-1) > 1e-5 || p.ClassID != best {
			return Evaluation{}, fmt.Errorf("tabular: inconsistent probability/class at row %d", i)
		}
		result.ConfusionMatrix[targets[i]][p.ClassID]++
		if p.ClassID == targets[i] {
			correct++
		}
		result.CrossEntropy -= math.Log(math.Max(1e-7, p.Probabilities[targets[i]]))
	}
	result.Accuracy = float64(correct) / float64(result.Samples)
	result.CrossEntropy /= float64(result.Samples)
	for class := 0; class < classes; class++ {
		truth, predicted := 0, 0
		for other := 0; other < classes; other++ {
			truth += result.ConfusionMatrix[class][other]
			predicted += result.ConfusionMatrix[other][class]
		}
		if truth+predicted > 0 {
			result.MacroF1 += 2 * float64(result.ConfusionMatrix[class][class]) / float64(truth+predicted) / float64(classes)
		}
	}
	return result, nil
}

// Evaluate scores supplied standardized rows without modifying the classifier.
func (c *Classifier[T]) Evaluate(ctx context.Context, rows [][]float64, targets []int) (Evaluation, error) {
	predictions, err := c.PredictBatch(ctx, rows)
	if err != nil {
		return Evaluation{}, err
	}
	return EvaluatePredictions(predictions, targets, c.config.ClassCount)
}
