package metrics

import (
	"math"
	"sort"
)

// Metrics holds evaluation metrics for model performance
type Metrics struct {
	PearsonCorrelation  float64
	SpearmanCorrelation float64
	MSE                 float64
	RMSE                float64
	MAE                 float64
}

// CalculateMetrics computes evaluation metrics for predictions vs targets
func CalculateMetrics(predictions, targets []float64) *Metrics {
	if len(predictions) != len(targets) || len(predictions) == 0 {
		return nil
	}

	pearson := PearsonCorrelation(predictions, targets)
	spearman := SpearmanCorrelation(predictions, targets)
	mse := calculateMSE(predictions, targets)
	rmse := math.Sqrt(mse)
	mae := calculateMAE(predictions, targets)

	return &Metrics{
		PearsonCorrelation:  pearson,
		SpearmanCorrelation: spearman,
		MSE:                 mse,
		RMSE:                rmse,
		MAE:                 mae,
	}
}

// PearsonCorrelation calculates the Pearson correlation coefficient between two slices
func PearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return math.NaN()
	}

	n := float64(len(x))
	
	// Calculate means
	var sumX, sumY float64
	for i := 0; i < len(x); i++ {
		sumX += x[i]
		sumY += y[i]
	}
	meanX := sumX / n
	meanY := sumY / n

	// Calculate numerator and denominators
	var numerator, sumXX, sumYY float64
	for i := 0; i < len(x); i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		sumXX += dx * dx
		sumYY += dy * dy
	}

	denominator := math.Sqrt(sumXX * sumYY)
	if denominator == 0 {
		return math.NaN()
	}

	return numerator / denominator
}

// SpearmanCorrelation calculates the Spearman rank correlation coefficient
func SpearmanCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return math.NaN()
	}

	// Convert to ranks
	ranksX := calculateRanks(x)
	ranksY := calculateRanks(y)

	// Calculate Pearson correlation on ranks
	return PearsonCorrelation(ranksX, ranksY)
}

// calculateRanks converts values to their ranks
func calculateRanks(values []float64) []float64 {
	n := len(values)
	ranks := make([]float64, n)
	
	// Create sorted indices
	type indexValue struct {
		index int
		value float64
	}
	
	sorted := make([]indexValue, n)
	for i, v := range values {
		sorted[i] = indexValue{index: i, value: v}
	}
	
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].value < sorted[j].value
	})
	
	// Assign ranks handling ties by averaging
	i := 0
	for i < n {
		j := i
		currentValue := sorted[i].value
		for j < n && sorted[j].value == currentValue {
			j++
		}
		
		// Average rank for tied values
		avgRank := float64(i+j-1) / 2.0 + 1.0
		
		for k := i; k < j; k++ {
			ranks[sorted[k].index] = avgRank
		}
		
		i = j
	}
	
	return ranks
}

// calculateMSE computes Mean Squared Error
func calculateMSE(predictions, targets []float64) float64 {
	sum := 0.0
	for i := range predictions {
		diff := predictions[i] - targets[i]
		sum += diff * diff
	}
	return sum / float64(len(predictions))
}

// calculateMAE computes Mean Absolute Error
func calculateMAE(predictions, targets []float64) float64 {
	sum := 0.0
	for i := range predictions {
		sum += math.Abs(predictions[i] - targets[i])
	}
	return sum / float64(len(predictions))
}