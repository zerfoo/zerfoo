package numerai

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/tensor"
)

func TestNewRiskManager(t *testing.T) {
	config := DefaultRiskConfig()
	outputDir := "/tmp/test_risk_output"
	rm := NewRiskManager(config, outputDir)
	
	if rm == nil {
		t.Fatal("NewRiskManager returned nil")
	}
	if rm.config != config {
		t.Error("RiskManager config not set correctly")
	}
	if rm.outputDir != outputDir {
		t.Error("RiskManager outputDir not set correctly")
	}
}

func TestDefaultRiskConfig(t *testing.T) {
	config := DefaultRiskConfig()
	
	if config.MaxVariance != 0.04 {
		t.Errorf("Expected MaxVariance 0.04, got %f", config.MaxVariance)
	}
	if config.MinCorrelation != 0.01 {
		t.Errorf("Expected MinCorrelation 0.01, got %f", config.MinCorrelation)
	}
	if config.WeeklyReviewDay != 1 {
		t.Errorf("Expected WeeklyReviewDay 1 (Monday), got %d", config.WeeklyReviewDay)
	}
}

func TestRiskLevel(t *testing.T) {
	testCases := []struct {
		level    RiskLevel
		expected string
	}{
		{RiskLow, "LOW"},
		{RiskMedium, "MEDIUM"},
		{RiskHigh, "HIGH"},
		{RiskCritical, "CRITICAL"},
	}
	
	for _, tc := range testCases {
		if tc.level.String() != tc.expected {
			t.Errorf("RiskLevel(%d).String() = %s, expected %s", tc.level, tc.level.String(), tc.expected)
		}
	}
}

func TestRiskManagerComputeRiskMetrics(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Create test data
	predData := []float32{0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.4}
	predictions, err := tensor.New[float32]([]int{10, 1}, predData)
	if err != nil {
		t.Fatalf("Failed to create predictions tensor: %v", err)
	}
	
	targetData := []float32{0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.7, 0.9, 0.5}
	targets, err := tensor.New[float32]([]int{10, 1}, targetData)
	if err != nil {
		t.Fatalf("Failed to create targets tensor: %v", err)
	}
	
	// Create dummy features
	featData := make([]float32, 10*5) // 10 samples, 5 features
	for i := range featData {
		featData[i] = float32(i) * 0.01
	}
	features, err := tensor.New[float32]([]int{10, 5}, featData)
	if err != nil {
		t.Fatalf("Failed to create features tensor: %v", err)
	}
	
	eras := []string{"era1", "era1", "era1", "era2", "era2", "era2", "era3", "era3", "era3", "era3"}
	
	metrics, err := rm.ComputeRiskMetrics(predictions, targets, features, eras, "test_model")
	if err != nil {
		t.Fatalf("ComputeRiskMetrics failed: %v", err)
	}
	
	// Check that metrics were computed
	if metrics.ModelName != "test_model" {
		t.Errorf("Expected model name 'test_model', got %s", metrics.ModelName)
	}
	
	if metrics.Variance < 0 {
		t.Errorf("Variance should be non-negative, got %f", metrics.Variance)
	}
	
	if metrics.Correlation < -1 || metrics.Correlation > 1 {
		t.Errorf("Correlation should be in [-1, 1], got %f", metrics.Correlation)
	}
	
	if metrics.Concentration < 0 || metrics.Concentration > 1 {
		t.Errorf("Concentration should be in [0, 1], got %f", metrics.Concentration)
	}
	
	if len(metrics.FeatureExposures) != 5 {
		t.Errorf("Expected 5 feature exposures, got %d", len(metrics.FeatureExposures))
	}
	
	if len(metrics.EraExposures) != 3 {
		t.Errorf("Expected 3 era exposures, got %d", len(metrics.EraExposures))
	}
	
	if metrics.OverallRiskScore < 0 || metrics.OverallRiskScore > 1 {
		t.Errorf("Overall risk score should be in [0, 1], got %f", metrics.OverallRiskScore)
	}
}

func TestMonitorRisks(t *testing.T) {
	config := DefaultRiskConfig()
	rm := NewRiskManager(config, "/tmp/test_output")
	
	// Create high-risk metrics
	metrics := &RiskMetrics{
		Variance:         0.08, // Above critical threshold (0.06)
		Correlation:      -0.01, // Below critical threshold (-0.005)
		Concentration:    0.12,  // Above max threshold (0.1)
		MaxExposure:      0.07,  // Above max threshold (0.05)
		MaxDrawdown:      0.25,  // Above max threshold (0.2)
		Sharpe:          0.3,   // Below min threshold (0.5)
		PredictionDrift:  0.03,  // Above max threshold (0.02)
		OverallRiskScore: 0.75,
	}
	
	alerts := rm.MonitorRisks(metrics)
	
	if len(alerts) == 0 {
		t.Error("Expected alerts for high-risk metrics")
	}
	
	// Check specific alerts
	varianceAlert := false
	correlationAlert := false
	for _, alert := range alerts {
		if alert.Type == "variance" && alert.Level == RiskCritical {
			varianceAlert = true
		}
		if alert.Type == "correlation" && alert.Level == RiskCritical {
			correlationAlert = true
		}
	}
	
	if !varianceAlert {
		t.Error("Expected critical variance alert")
	}
	if !correlationAlert {
		t.Error("Expected critical correlation alert")
	}
}

func TestMonitorRisksWithCooldown(t *testing.T) {
	config := DefaultRiskConfig()
	config.AlertCooldownHours = 1 // 1 hour cooldown
	rm := NewRiskManager(config, "/tmp/test_output")
	
	metrics := &RiskMetrics{
		Variance: 0.08, // Above critical threshold
	}
	
	// First alert should be generated
	alerts1 := rm.MonitorRisks(metrics)
	if len(alerts1) == 0 {
		t.Error("Expected first alert")
	}
	
	// Second alert immediately should be filtered by cooldown
	alerts2 := rm.MonitorRisks(metrics)
	if len(alerts2) > 0 {
		t.Error("Expected second alert to be filtered by cooldown")
	}
}

func TestComputeCorrelation(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Perfect correlation
	predData := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	targetData := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	
	predictions, _ := tensor.New[float32]([]int{5, 1}, predData)
	targets, _ := tensor.New[float32]([]int{5, 1}, targetData)
	
	correlation, err := rm.computeCorrelation(predictions, targets)
	if err != nil {
		t.Fatalf("computeCorrelation failed: %v", err)
	}
	
	if correlation < 0.99 {
		t.Errorf("Expected near-perfect correlation, got %f", correlation)
	}
	
	// Zero correlation (constant predictions)
	constPredData := []float32{0.5, 0.5, 0.5, 0.5, 0.5}
	constPredictions, _ := tensor.New[float32]([]int{5, 1}, constPredData)
	
	zeroCorr, err := rm.computeCorrelation(constPredictions, targets)
	if err != nil {
		t.Fatalf("computeCorrelation with constant predictions failed: %v", err)
	}
	
	if zeroCorr != 0.0 {
		t.Errorf("Expected zero correlation for constant predictions, got %f", zeroCorr)
	}
}

func TestComputeConcentration(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Uniform distribution (low concentration)
	uniformData := []float32{0.2, 0.2, 0.2, 0.2, 0.2}
	uniform, _ := tensor.New[float32]([]int{5, 1}, uniformData)
	
	uniformConc := rm.computeConcentration(uniform)
	if uniformConc > 0.1 {
		t.Errorf("Expected low concentration for uniform data, got %f", uniformConc)
	}
	
	// Highly concentrated distribution
	concentratedData := []float32{0.01, 0.01, 0.01, 0.01, 0.96}
	concentrated, _ := tensor.New[float32]([]int{5, 1}, concentratedData)
	
	concConc := rm.computeConcentration(concentrated)
	if concConc <= uniformConc {
		t.Error("Concentrated data should have higher concentration than uniform")
	}
}

func TestComputeVolatility(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Constant returns (zero volatility)
	constantReturns := []float64{0.01, 0.01, 0.01, 0.01, 0.01}
	constVol := rm.computeVolatility(constantReturns)
	if constVol > 1e-10 {
		t.Errorf("Expected near-zero volatility for constant returns, got %f", constVol)
	}
	
	// Variable returns (positive volatility)
	variableReturns := []float64{-0.05, 0.02, 0.08, -0.03, 0.01}
	varVol := rm.computeVolatility(variableReturns)
	if varVol <= 0 {
		t.Errorf("Expected positive volatility for variable returns, got %f", varVol)
	}
	if varVol <= constVol {
		t.Error("Variable returns should have higher volatility than constant")
	}
}

func TestComputeSharpe(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Positive returns with low volatility (high Sharpe)
	goodReturns := []float64{0.02, 0.025, 0.018, 0.022, 0.021}
	goodSharpe := rm.computeSharpe(goodReturns)
	if goodSharpe <= 0 {
		t.Errorf("Expected positive Sharpe ratio for good returns, got %f", goodSharpe)
	}
	
	// Negative returns (negative Sharpe)
	badReturns := []float64{-0.02, -0.01, -0.03, -0.015, -0.025}
	badSharpe := rm.computeSharpe(badReturns)
	if badSharpe >= 0 {
		t.Errorf("Expected negative Sharpe ratio for negative returns, got %f", badSharpe)
	}
	
	if badSharpe >= goodSharpe {
		t.Error("Good returns should have better Sharpe ratio than bad returns")
	}
}

func TestComputeMaxDrawdown(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Returns with significant drawdown
	returnsWithDrawdown := []float64{0.1, 0.05, -0.2, -0.1, 0.05, 0.1}
	drawdown := rm.computeMaxDrawdown(returnsWithDrawdown)
	if drawdown <= 0 {
		t.Errorf("Expected positive drawdown, got %f", drawdown)
	}
	
	// Only positive returns (minimal drawdown)
	positiveReturns := []float64{0.01, 0.02, 0.01, 0.03, 0.02}
	posDrawdown := rm.computeMaxDrawdown(positiveReturns)
	if posDrawdown >= drawdown {
		t.Error("Positive returns should have lower drawdown than mixed returns")
	}
}

func TestGenerateWeeklyReview(t *testing.T) {
	// Create temporary output directory
	tempDir, err := os.MkdirTemp("", "risk_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	}()
	
	rm := NewRiskManager(DefaultRiskConfig(), tempDir)
	
	// Add some sample metrics to history
	sampleMetrics := RiskMetrics{
		Variance:         0.02,
		Correlation:      0.05,
		Concentration:    0.03,
		Sharpe:          0.8,
		MaxDrawdown:     0.1,
		Volatility:      0.15,
		OverallRiskScore: 0.3,
		ModelName:       "test_model",
		Timestamp:       time.Now(),
	}
	rm.riskHistory = append(rm.riskHistory, sampleMetrics)
	
	review, err := rm.GenerateWeeklyReview("test_model")
	if err != nil {
		t.Fatalf("GenerateWeeklyReview failed: %v", err)
	}
	
	if review.ModelName != "test_model" {
		t.Errorf("Expected model name 'test_model', got %s", review.ModelName)
	}
	
	if review.PerformanceGrade == "" {
		t.Error("Expected performance grade to be assigned")
	}
	
	if review.RiskGrade == "" {
		t.Error("Expected risk grade to be assigned")
	}
	
	if review.TrendAnalysis == "" {
		t.Error("Expected trend analysis to be assigned")
	}
	
	// Check that review was saved
	reviewPath := filepath.Join(tempDir, "reviews", fmt.Sprintf("review_%s.json", review.WeekOf.Format("2006_01_02")))
	if _, err := os.Stat(reviewPath); os.IsNotExist(err) {
		t.Error("Expected review file to be saved")
	}
}

func TestAssignPerformanceGrade(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	testCases := []struct {
		sharpe   float64
		expected string
	}{
		{1.2, "A"},
		{0.8, "B"},
		{0.5, "C"},
		{0.3, "D"},
		{0.1, "F"},
	}
	
	for _, tc := range testCases {
		metrics := RiskMetrics{Sharpe: tc.sharpe}
		grade := rm.assignPerformanceGrade(metrics)
		if grade != tc.expected {
			t.Errorf("Sharpe %.1f: expected grade %s, got %s", tc.sharpe, tc.expected, grade)
		}
	}
}

func TestAssignRiskGrade(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	testCases := []struct {
		riskScore float64
		expected  string
	}{
		{0.1, "A"},
		{0.3, "B"},
		{0.5, "C"},
		{0.7, "D"},
		{0.9, "F"},
	}
	
	for _, tc := range testCases {
		metrics := RiskMetrics{OverallRiskScore: tc.riskScore}
		grade := rm.assignRiskGrade(metrics)
		if grade != tc.expected {
			t.Errorf("Risk score %.1f: expected grade %s, got %s", tc.riskScore, tc.expected, grade)
		}
	}
}

func TestAnalyzeTrend(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	testCases := []struct {
		riskChange float64
		expected   string
	}{
		{-0.1, "improving"},
		{-0.02, "stable"},
		{0.02, "stable"},
		{0.1, "degrading"},
	}
	
	for _, tc := range testCases {
		weekOverWeek := RiskMetrics{OverallRiskScore: tc.riskChange}
		trend := rm.analyzeTrend(weekOverWeek)
		if trend != tc.expected {
			t.Errorf("Risk change %.2f: expected trend %s, got %s", tc.riskChange, tc.expected, trend)
		}
	}
}

func TestGenerateRecommendations(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Create review with high-risk metrics
	review := &WeeklyReview{
		Summary: RiskMetrics{
			Variance:    0.05, // Above threshold
			Correlation: 0.005, // Below threshold
		},
		ActiveAlerts: []RiskAlert{{}, {}, {}, {}}, // 4 alerts (>3 threshold)
	}
	
	recommendations := rm.generateRecommendations(review)
	
	if len(recommendations) == 0 {
		t.Error("Expected recommendations for high-risk metrics")
	}
	
	// Check for specific recommendations
	found := map[string]bool{
		"Reduce Prediction Variance": false,
		"Improve Model Correlation":  false,
		"Review Risk Monitoring Process": false,
	}
	
	for _, rec := range recommendations {
		if _, exists := found[rec.Title]; exists {
			found[rec.Title] = true
		}
	}
	
	for title, wasFound := range found {
		if !wasFound {
			t.Errorf("Expected recommendation: %s", title)
		}
	}
}

func TestGenerateActionItems(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	review := &WeeklyReview{
		Recommendations: []Recommendation{
			{
				Title:    "High Priority Item",
				Priority: "high",
				Description: "This is urgent",
				Timeline: "immediate",
			},
			{
				Title:    "Low Priority Item",
				Priority: "low",
				Description: "This can wait",
				Timeline: "this_month",
			},
		},
	}
	
	actionItems := rm.generateActionItems(review)
	
	// Should only create action items for high priority recommendations
	if len(actionItems) != 1 {
		t.Errorf("Expected 1 action item for high priority recommendation, got %d", len(actionItems))
	}
	
	if len(actionItems) > 0 {
		item := actionItems[0]
		if item.Title != "High Priority Item" {
			t.Errorf("Expected action item title 'High Priority Item', got %s", item.Title)
		}
		if item.Status != "pending" {
			t.Errorf("Expected action item status 'pending', got %s", item.Status)
		}
	}
}

func TestValidateRiskInputs(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Valid inputs
	predData := []float32{0.5, 0.6}
	predictions, _ := tensor.New[float32]([]int{2, 1}, predData)
	targets, _ := tensor.New[float32]([]int{2, 1}, predData)
	features, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	
	err := rm.validateRiskInputs(predictions, targets, features, []string{"era1", "era2"})
	if err != nil {
		t.Errorf("Expected valid inputs to pass validation, got error: %v", err)
	}
	
	// Nil predictions
	err = rm.validateRiskInputs(nil, targets, features, []string{"era1", "era2"})
	if err == nil {
		t.Error("Expected error for nil predictions")
	}
	
	// Mismatched sample counts
	wrongTargets, _ := tensor.New[float32]([]int{3, 1}, []float32{1, 2, 3})
	err = rm.validateRiskInputs(predictions, wrongTargets, features, []string{"era1", "era2"})
	if err == nil {
		t.Error("Expected error for mismatched sample counts")
	}
	
	// Wrong era count
	err = rm.validateRiskInputs(predictions, targets, features, []string{"era1"})
	if err == nil {
		t.Error("Expected error for wrong era count")
	}
}

func TestGetWeekStart(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Test with a known Wednesday (2024-01-03)
	wednesday := time.Date(2024, 1, 3, 15, 30, 45, 0, time.UTC)
	weekStart := rm.getWeekStart(wednesday)
	
	// Should return Monday (2024-01-01) at midnight
	expectedStart := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	
	if !weekStart.Equal(expectedStart) {
		t.Errorf("Expected week start %v, got %v", expectedStart, weekStart)
	}
	
	// Test with Monday itself
	monday := time.Date(2024, 1, 1, 10, 0, 0, 0, time.UTC)
	mondayStart := rm.getWeekStart(monday)
	expectedMondayStart := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	
	if !mondayStart.Equal(expectedMondayStart) {
		t.Errorf("Expected Monday start %v, got %v", expectedMondayStart, mondayStart)
	}
}

func TestComputeDueDate(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	now := time.Now()
	
	testCases := []struct {
		timeline string
		days     int
	}{
		{"immediate", 1},
		{"this_week", 7},
		{"this_month", 30}, // Approximately
		{"unknown", 7},     // Default
	}
	
	for _, tc := range testCases {
		dueDate := rm.computeDueDate(tc.timeline)
		daysDiff := int(dueDate.Sub(now).Hours() / 24)
		
		// Allow some tolerance for month calculation
		tolerance := 1
		if tc.timeline == "this_month" {
			tolerance = 5
		}
		
		if abs(daysDiff-tc.days) > tolerance {
			t.Errorf("Timeline %s: expected ~%d days, got %d", tc.timeline, tc.days, daysDiff)
		}
	}
}

func TestRiskManagerString(t *testing.T) {
	rm := NewRiskManager(DefaultRiskConfig(), "/tmp/test_output")
	
	// Add some data to test string representation
	rm.alerts = append(rm.alerts, RiskAlert{})
	rm.riskHistory = append(rm.riskHistory, RiskMetrics{})
	rm.reviewHistory = append(rm.reviewHistory, WeeklyReview{})
	
	result := rm.String()
	if result == "" {
		t.Error("String() should return non-empty result")
	}
	
	if len(result) < 20 {
		t.Error("String() should return meaningful description")
	}
}

// Helper function
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}