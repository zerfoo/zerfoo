package metrics

import (
	"fmt"
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

// CalculateEraMetrics calculates per-era correlations and summary statistics
func CalculateEraMetrics(predictions, targets []float64, eraLabels []int) ([]float64, float64, float64) {
	if len(predictions) != len(targets) || len(predictions) != len(eraLabels) || len(predictions) == 0 {
		return []float64{}, 0.0, 0.0
	}

	// Group by era
	eraData := make(map[int]struct {
		preds   []float64
		targets []float64
	})

	for i, era := range eraLabels {
		if _, exists := eraData[era]; !exists {
			eraData[era] = struct {
				preds   []float64
				targets []float64
			}{
				preds:   []float64{},
				targets: []float64{},
			}
		}
		data := eraData[era]
		data.preds = append(data.preds, predictions[i])
		data.targets = append(data.targets, targets[i])
		eraData[era] = data
	}

	// Calculate correlation for each era
	var eraCorrelations []float64
	for _, data := range eraData {
		corr := PearsonCorrelation(data.preds, data.targets)
		if !math.IsNaN(corr) {
			eraCorrelations = append(eraCorrelations, corr)
		}
	}

	if len(eraCorrelations) == 0 {
		return []float64{}, 0.0, 0.0
	}

	// Calculate mean and standard deviation
	var sum, sumSq float64
	for _, corr := range eraCorrelations {
		sum += corr
		sumSq += corr * corr
	}

	mean := sum / float64(len(eraCorrelations))
	variance := sumSq/float64(len(eraCorrelations)) - mean*mean
	stdDev := math.Sqrt(math.Max(0, variance))

	return eraCorrelations, mean, stdDev
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

// NumeraiMetrics holds Numerai-specific validation metrics
type NumeraiMetrics struct {
	// Core Numerai metrics
	PearsonCorrelation  float64
	SpearmanCorrelation float64
	
	// Performance metrics specific to Numerai
	SharpeRatio         float64
	MaxDrawdown         float64
	CalmarRatio         float64
	
	// Era-based analysis (key for Numerai)
	EraCorrelations     []float64
	MeanEraCorrelation  float64
	StdEraCorrelation   float64
	
	// Risk metrics for Numerai trading
	Volatility          float64
	DownsideDeviation   float64
	HitRate             float64  // % of correct directional predictions
}

// CalculateNumeraiMetrics computes comprehensive Numerai-specific metrics
func CalculateNumeraiMetrics(
	predictions, targets []float64,
	eraLabels []int,
) *NumeraiMetrics {
	if len(predictions) != len(targets) || len(predictions) == 0 {
		return &NumeraiMetrics{}
	}
	
	metrics := &NumeraiMetrics{}
	
	// Core correlations (most important for Numerai)
	metrics.PearsonCorrelation = PearsonCorrelation(predictions, targets)
	metrics.SpearmanCorrelation = SpearmanCorrelation(predictions, targets)
	
	// Era-based analysis if era labels provided (critical for Numerai)
	if len(eraLabels) == len(predictions) {
		metrics.EraCorrelations, metrics.MeanEraCorrelation, metrics.StdEraCorrelation = CalculateEraMetrics(predictions, targets, eraLabels)
	}
	
	// Financial metrics using Numerai-style returns
	returns := computeNumeraiReturns(predictions, targets)
	metrics.SharpeRatio = calculateSharpeRatio(returns)
	metrics.MaxDrawdown = calculateMaxDrawdown(returns)
	metrics.CalmarRatio = calculateCalmarRatio(returns, metrics.MaxDrawdown)
	metrics.Volatility = calculateStandardDeviation(returns)
	metrics.DownsideDeviation = calculateDownsideDeviation(returns)
	metrics.HitRate = calculateHitRate(predictions, targets)
	
	return metrics
}

// computeNumeraiReturns computes returns specific to Numerai scoring
func computeNumeraiReturns(predictions, targets []float64) []float64 {
	returns := make([]float64, len(predictions))
	for i := range predictions {
		// Numerai-style return: prediction * target
		returns[i] = predictions[i] * targets[i]
	}
	return returns
}

// calculateSharpeRatio computes the Sharpe ratio
func calculateSharpeRatio(returns []float64) float64 {
	if len(returns) == 0 {
		return 0.0
	}
	
	meanReturn := calculateMean(returns)
	stdReturn := calculateStandardDeviation(returns)
	
	if stdReturn == 0 {
		return 0.0
	}
	
	// Annualized Sharpe ratio for daily returns
	return meanReturn / stdReturn * math.Sqrt(252)
}

// calculateMaxDrawdown computes the maximum drawdown
func calculateMaxDrawdown(returns []float64) float64 {
	if len(returns) == 0 {
		return 0.0
	}
	
	cumReturns := make([]float64, len(returns))
	cumReturns[0] = returns[0]
	for i := 1; i < len(returns); i++ {
		cumReturns[i] = cumReturns[i-1] + returns[i]
	}
	
	maxDrawdown := 0.0
	peak := cumReturns[0]
	
	for _, value := range cumReturns {
		if value > peak {
			peak = value
		}
		
		drawdown := peak - value
		if drawdown > maxDrawdown {
			maxDrawdown = drawdown
		}
	}
	
	return maxDrawdown
}

// calculateCalmarRatio computes the Calmar ratio
func calculateCalmarRatio(returns []float64, maxDrawdown float64) float64 {
	if maxDrawdown == 0 {
		return 0.0
	}
	
	annualizedReturn := calculateMean(returns) * 252
	return annualizedReturn / maxDrawdown
}

// calculateDownsideDeviation computes downside deviation
func calculateDownsideDeviation(returns []float64) float64 {
	if len(returns) == 0 {
		return 0.0
	}
	
	meanReturn := calculateMean(returns)
	sumSquares := 0.0
	count := 0
	
	for _, r := range returns {
		if r < meanReturn {
			diff := r - meanReturn
			sumSquares += diff * diff
			count++
		}
	}
	
	if count == 0 {
		return 0.0
	}
	
	return math.Sqrt(sumSquares / float64(count))
}

// calculateHitRate computes hit rate (directional accuracy)
func calculateHitRate(predictions, targets []float64) float64 {
	if len(predictions) != len(targets) || len(predictions) == 0 {
		return 0.0
	}
	
	correct := 0
	for i := range predictions {
		if (predictions[i] > 0 && targets[i] > 0) || (predictions[i] <= 0 && targets[i] <= 0) {
			correct++
		}
	}
	
	return float64(correct) / float64(len(predictions))
}

// calculateMean computes the mean of a slice
func calculateMean(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// calculateStandardDeviation computes standard deviation
func calculateStandardDeviation(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	meanVal := calculateMean(data)
	sumSquares := 0.0
	
	for _, x := range data {
		diff := x - meanVal
		sumSquares += diff * diff
	}
	
	return math.Sqrt(sumSquares / float64(len(data)))
}

// PrintNumeraiMetrics prints Numerai metrics in a readable format
func (m *NumeraiMetrics) PrintNumeraiMetrics() {
	fmt.Printf("\n=== Numerai Model Performance ===\n")
	fmt.Printf("Core Metrics:\n")
	fmt.Printf("  Pearson Correlation:  %8.6f\n", m.PearsonCorrelation)
	fmt.Printf("  Spearman Correlation: %8.6f\n", m.SpearmanCorrelation)
	
	fmt.Printf("\nRisk-Adjusted Performance:\n")
	fmt.Printf("  Sharpe Ratio:         %8.4f\n", m.SharpeRatio)
	fmt.Printf("  Calmar Ratio:         %8.4f\n", m.CalmarRatio)
	
	fmt.Printf("\nRisk Metrics:\n")
	fmt.Printf("  Volatility:           %8.6f\n", m.Volatility)
	fmt.Printf("  Max Drawdown:         %8.6f\n", m.MaxDrawdown)
	fmt.Printf("  Downside Deviation:   %8.6f\n", m.DownsideDeviation)
	
	fmt.Printf("\nAccuracy:\n")
	fmt.Printf("  Hit Rate:             %8.2f%%\n", m.HitRate*100)
	
	if len(m.EraCorrelations) > 0 {
		fmt.Printf("\nEra-Based Analysis (Critical for Numerai):\n")
		fmt.Printf("  Era Count:            %8d\n", len(m.EraCorrelations))
		fmt.Printf("  Mean Era Correlation: %8.6f\n", m.MeanEraCorrelation)
		fmt.Printf("  Std Era Correlation:  %8.6f\n", m.StdEraCorrelation)
	}
}