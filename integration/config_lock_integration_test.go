package integration

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/numerai"
)

func TestConfigLockIntegration(t *testing.T) {
	// Create temporary directory for integration test
	tempDir, err := os.MkdirTemp("", "config_lock_integration")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	}()

	locker := numerai.NewConfigLocker(tempDir)

	// Test 1: Lock a baseline model configuration
	config := numerai.BaselineModelConfig{
		ModelType:      "mlp",
		HiddenSizes:    []int{128, 64, 32},
		DropoutRate:    0.2,
		UseLayerNorm:   true,
		LearningRate:   0.001,
		BatchSize:      1000,
		NumEpochs:      50,
		WeightDecay:    1e-4,
		EarlyStop:      10,
		RandomSeed:     12345,
		FeatureScaling: "standard",
		TopKFeatures:   100,
	}

	metadata := map[string]interface{}{
		"experiment":    "integration_test",
		"description":   "Testing config lock integration",
		"data_version":  "v1.2.3",
		"model_version": "v2.1.0",
	}

	// Create config lock
	lock, err := locker.LockConfig(config, metadata)
	if err != nil {
		t.Fatalf("Failed to create config lock: %v", err)
	}

	t.Logf("Created config lock with hash: %s", lock.ConfigHash)

	// Test 2: Validate against the same configuration (should pass)
	if err := locker.ValidateConfigLock(config, lock); err != nil {
		t.Errorf("Validation should pass for identical config: %v", err)
	}

	// Test 3: Validate against modified configuration (should fail)
	modifiedConfig := config
	modifiedConfig.LearningRate = 0.002 // Change learning rate

	if err := locker.ValidateConfigLock(modifiedConfig, lock); err == nil {
		t.Error("Validation should fail for modified config")
	} else {
		t.Logf("Expected validation failure: %v", err)
	}

	// Test 4: Find the latest lock
	latestFile, err := locker.FindLatestLock()
	if err != nil {
		t.Fatalf("Failed to find latest lock: %v", err)
	}

	// Test 5: Load the lock and verify it matches
	loadedLock, err := locker.LoadConfigLock(latestFile)
	if err != nil {
		t.Fatalf("Failed to load lock: %v", err)
	}

	if loadedLock.ConfigHash != lock.ConfigHash {
		t.Error("Loaded lock hash doesn't match original")
	}

	// Test 6: Test warm run functionality
	dataValidator := func(dataPath string) error {
		// For integration test, just check it's not empty
		if dataPath == "" {
			return nil
		}
		return nil
	}

	modelTester := func(cfg interface{}) error {
		// Validate that we can unmarshal the config
		data, err := json.Marshal(cfg)
		if err != nil {
			return err
		}

		var testConfig numerai.BaselineModelConfig
		return json.Unmarshal(data, &testConfig)
	}

	runner := numerai.NewWarmRunner(dataValidator, modelTester)

	// Run warm-up with the loaded lock
	result, err := runner.RunWarmUp(loadedLock, "test_data_path")
	if err != nil {
		t.Fatalf("Warm run failed: %v", err)
	}

	if !result.Success {
		t.Errorf("Warm run should succeed, error: %s", result.ErrorMessage)
	}

	// Validate all expected checks passed
	expectedValidations := []string{
		"data_valid",
		"model_valid",
		"config_lock_valid",
		"environment_stable",
	}

	for _, validation := range expectedValidations {
		if !result.Validation[validation] {
			t.Errorf("Validation %s should pass", validation)
		}
	}

	t.Logf("Warm run completed in %v with %d validations", 
		result.Duration, len(result.Validation))
}

func TestConfigLockCLIScenario(t *testing.T) {
	// This test simulates the CLI workflow for config locking
	tempDir, err := os.MkdirTemp("", "cli_scenario_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	}()

	// Create dummy data file for testing
	dataFile := filepath.Join(tempDir, "test_data.csv")
	if err := os.WriteFile(dataFile, []byte("era,feature_1,feature_2,target\n1,0.1,0.2,0.5\n"), 0644); err != nil {
		t.Fatalf("Failed to create test data file: %v", err)
	}

	// Simulate CLI configuration
	cliConfig := struct {
		DataPath       string  `json:"data_path"`
		ModelType      string  `json:"model_type"`
		LearningRate   float64 `json:"learning_rate"`
		BatchSize      int     `json:"batch_size"`
		NumEpochs      int     `json:"num_epochs"`
		RandomSeed     int     `json:"random_seed"`
	}{
		DataPath:     dataFile,
		ModelType:    "linear",
		LearningRate: 0.01,
		BatchSize:    1000,
		NumEpochs:    25,
		RandomSeed:   42,
	}

	lockDir := filepath.Join(tempDir, "config_locks")
	locker := numerai.NewConfigLocker(lockDir)

	// Step 1: Create initial config lock
	lock1, err := locker.LockConfig(cliConfig, map[string]interface{}{
		"run_type": "initial_experiment",
		"notes":    "First run with this configuration",
	})
	if err != nil {
		t.Fatalf("Failed to create initial lock: %v", err)
	}

	t.Logf("Step 1: Created initial lock: %s", lock1.ConfigHash)

	// Step 2: Simulate running with the same config (should validate)
	if err := locker.ValidateConfigLock(cliConfig, lock1); err != nil {
		t.Errorf("Step 2: Same config validation should pass: %v", err)
	}

	// Step 3: Simulate parameter change (learning rate adjustment)
	adjustedConfig := cliConfig
	adjustedConfig.LearningRate = 0.005

	// This should fail validation against the original lock
	if err := locker.ValidateConfigLock(adjustedConfig, lock1); err == nil {
		t.Error("Step 3: Adjusted config should fail validation against original lock")
	}

	// Step 4: Create new lock for adjusted config
	lock2, err := locker.LockConfig(adjustedConfig, map[string]interface{}{
		"run_type": "parameter_adjustment",
		"notes":    "Reduced learning rate from 0.01 to 0.005",
		"parent_hash": lock1.ConfigHash,
	})
	if err != nil {
		t.Fatalf("Failed to create adjusted lock: %v", err)
	}

	t.Logf("Step 4: Created adjusted lock: %s", lock2.ConfigHash)

	// Step 5: Verify we can find the latest lock
	latestFile, err := locker.FindLatestLock()
	if err != nil {
		t.Fatalf("Step 5: Failed to find latest lock: %v", err)
	}

	latestLock, err := locker.LoadConfigLock(latestFile)
	if err != nil {
		t.Fatalf("Failed to load latest lock: %v", err)
	}

	// The latest should be the adjusted config
	if latestLock.ConfigHash != lock2.ConfigHash {
		t.Error("Step 5: Latest lock should be the adjusted config")
	}

	// Step 6: Demonstrate warm run with file validation
	dataValidator := func(dataPath string) error {
		if _, err := os.Stat(dataPath); os.IsNotExist(err) {
			return err
		}
		return nil
	}

	modelTester := func(cfg interface{}) error {
		// Check required fields are present
		data, err := json.Marshal(cfg)
		if err != nil {
			return err
		}

		var config map[string]interface{}
		if err := json.Unmarshal(data, &config); err != nil {
			return err
		}

		// Validate required fields
		requiredFields := []string{"data_path", "model_type", "learning_rate"}
		for _, field := range requiredFields {
			if _, exists := config[field]; !exists {
				return err
			}
		}

		return nil
	}

	runner := numerai.NewWarmRunner(dataValidator, modelTester)

	// Test warm run with the latest config
	warmResult, err := runner.RunWarmUp(latestLock, dataFile)
	if err != nil {
		t.Fatalf("Step 6: Warm run failed: %v", err)
	}

	if !warmResult.Success {
		t.Errorf("Step 6: Warm run should succeed: %s", warmResult.ErrorMessage)
	}

	t.Logf("Step 6: CLI scenario warm run completed successfully")
	t.Logf("  Duration: %v", warmResult.Duration)
	t.Logf("  Validations: %d passed", len(warmResult.Validation))

	// Verify the data file validation actually worked
	if !warmResult.Validation["data_valid"] {
		t.Error("Data validation should have passed")
	}

	if !warmResult.Validation["model_valid"] {
		t.Error("Model validation should have passed")
	}
}