package numerai

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestConfigLocker_LockConfig(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "config_lock_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	}()
	
	locker := NewConfigLocker(tempDir)
	
	// Test configuration
	testConfig := BaselineModelConfig{
		ModelType:    "linear",
		LearningRate: 0.01,
		BatchSize:    1000,
		NumEpochs:    10,
		RandomSeed:   42,
	}
	
	metadata := map[string]interface{}{
		"test_run": true,
		"version":  "1.0.0",
	}
	
	// Lock the configuration
	lock, err := locker.LockConfig(testConfig, metadata)
	if err != nil {
		t.Fatalf("Failed to lock config: %v", err)
	}
	
	// Validate lock properties
	if lock.Version == "" {
		t.Error("Lock version should not be empty")
	}
	
	if lock.ConfigHash == "" {
		t.Error("Config hash should not be empty")
	}
	
	if lock.Timestamp.IsZero() {
		t.Error("Timestamp should not be zero")
	}
	
	// Validate metadata
	if lock.Metadata["test_run"] != true {
		t.Error("Metadata not preserved correctly")
	}
	
	// Validate environment capture
	if lock.Environment == nil {
		t.Error("Environment should be captured")
	}
	
	// Check that lock file was created
	entries, err := os.ReadDir(tempDir)
	if err != nil {
		t.Fatalf("Failed to read temp dir: %v", err)
	}
	
	if len(entries) != 1 {
		t.Errorf("Expected 1 lock file, got %d", len(entries))
	}
}

func TestConfigLocker_ValidateConfigLock(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "config_validation_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	}()
	
	locker := NewConfigLocker(tempDir)
	
	originalConfig := BaselineModelConfig{
		ModelType:    "linear",
		LearningRate: 0.01,
		BatchSize:    1000,
		NumEpochs:    10,
		RandomSeed:   42,
	}
	
	// Create lock
	lock, err := locker.LockConfig(originalConfig, nil)
	if err != nil {
		t.Fatalf("Failed to lock config: %v", err)
	}
	
	// Test validation with same config (should pass)
	if err := locker.ValidateConfigLock(originalConfig, lock); err != nil {
		t.Errorf("Validation should pass for identical config: %v", err)
	}
	
	// Test validation with different config (should fail)
	differentConfig := originalConfig
	differentConfig.LearningRate = 0.02
	
	if err := locker.ValidateConfigLock(differentConfig, lock); err == nil {
		t.Error("Validation should fail for different config")
	}
}

func TestConfigLocker_LoadConfigLock(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "config_load_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	}()
	
	locker := NewConfigLocker(tempDir)
	
	testConfig := BaselineModelConfig{
		ModelType:    "mlp",
		HiddenSizes:  []int{64, 32},
		LearningRate: 0.001,
		BatchSize:    500,
		NumEpochs:    20,
		RandomSeed:   123,
	}
	
	metadata := map[string]interface{}{
		"experiment": "test_load",
		"notes":      "Testing configuration loading",
	}
	
	// Create and save lock
	originalLock, err := locker.LockConfig(testConfig, metadata)
	if err != nil {
		t.Fatalf("Failed to create lock: %v", err)
	}
	
	// Find and load the lock file
	lockFile, err := locker.FindLatestLock()
	if err != nil {
		t.Fatalf("Failed to find lock file: %v", err)
	}
	
	loadedLock, err := locker.LoadConfigLock(lockFile)
	if err != nil {
		t.Fatalf("Failed to load lock: %v", err)
	}
	
	// Validate loaded lock matches original
	if loadedLock.ConfigHash != originalLock.ConfigHash {
		t.Error("Loaded lock hash doesn't match original")
	}
	
	if loadedLock.Version != originalLock.Version {
		t.Error("Loaded lock version doesn't match original")
	}
	
	// Validate metadata preservation
	if loadedLock.Metadata["experiment"] != "test_load" {
		t.Error("Metadata not preserved during load")
	}
}

func TestConfigLocker_FindLatestLock(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "find_latest_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	}()
	
	locker := NewConfigLocker(tempDir)
	
	// Test with no lock files
	_, err = locker.FindLatestLock()
	if err == nil {
		t.Error("Should return error when no lock files exist")
	}
	
	// Create multiple lock files with different timestamps
	configs := []BaselineModelConfig{
		{ModelType: "linear", RandomSeed: 1},
		{ModelType: "mlp", RandomSeed: 2},
		{ModelType: "ensemble", RandomSeed: 3},
	}
	
	var lockFiles []string
	for i, config := range configs {
		lock, err := locker.LockConfig(config, map[string]interface{}{"order": i})
		if err != nil {
			t.Fatalf("Failed to create lock %d: %v", i, err)
		}
		
		// Add small delay to ensure different timestamps
		time.Sleep(10 * time.Millisecond)
		
		// Find the created lock file
		entries, err := os.ReadDir(tempDir)
		if err != nil {
			t.Fatalf("Failed to read temp dir: %v", err)
		}
		
		for _, entry := range entries {
			name := entry.Name()
			fullPath := filepath.Join(tempDir, name)
			
			// Check if this is a new file
			isNew := true
			for _, existing := range lockFiles {
				if existing == fullPath {
					isNew = false
					break
				}
			}
			
			if isNew && filepath.Ext(name) == ".json" {
				lockFiles = append(lockFiles, fullPath)
				break
			}
		}
		
		_ = lock // Use the lock variable
	}
	
	// Find latest lock
	latestFile, err := locker.FindLatestLock()
	if err != nil {
		t.Fatalf("Failed to find latest lock: %v", err)
	}
	
	// Load and verify it's the latest one
	latestLock, err := locker.LoadConfigLock(latestFile)
	if err != nil {
		t.Fatalf("Failed to load latest lock: %v", err)
	}
	
	// The latest should have order = 2 (last one created)
	if latestLock.Metadata["order"] != float64(2) { // JSON unmarshaling makes it float64
		t.Errorf("Expected latest lock to have order 2, got %v", latestLock.Metadata["order"])
	}
}

func TestWarmRunner_RunWarmUp(t *testing.T) {
	// Create validation functions
	dataValidator := func(dataPath string) error {
		if dataPath == "" {
			return nil
		}
		return nil // Always pass for test
	}
	
	modelTester := func(config interface{}) error {
		// Validate that config is a BaselineModelConfig
		if _, ok := config.(BaselineModelConfig); !ok {
			// Try to unmarshal from interface{}
			data, err := json.Marshal(config)
			if err != nil {
				return err
			}
			
			var testConfig BaselineModelConfig
			if err := json.Unmarshal(data, &testConfig); err != nil {
				return err
			}
		}
		return nil
	}
	
	runner := NewWarmRunner(dataValidator, modelTester)
	
	// Create a test config lock
	tempDir, err := os.MkdirTemp("", "warm_run_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	}()
	
	locker := NewConfigLocker(tempDir)
	testConfig := BaselineModelConfig{
		ModelType:    "linear",
		LearningRate: 0.01,
		BatchSize:    1000,
		NumEpochs:    5,
		RandomSeed:   42,
	}
	
	lock, err := locker.LockConfig(testConfig, map[string]interface{}{
		"warm_run_test": true,
	})
	if err != nil {
		t.Fatalf("Failed to create config lock: %v", err)
	}
	
	// Run warm-up
	result, err := runner.RunWarmUp(lock, "test_data_path")
	if err != nil {
		t.Fatalf("Warm run failed: %v", err)
	}
	
	// Validate result
	if !result.Success {
		t.Errorf("Warm run should succeed, error: %s", result.ErrorMessage)
	}
	
	if result.Duration <= 0 {
		t.Error("Duration should be positive")
	}
	
	// Check validations
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
	
	// Validate config lock reference
	if result.ConfigLock == nil {
		t.Error("Config lock should be included in result")
	}
	
	if result.ConfigLock.ConfigHash != lock.ConfigHash {
		t.Error("Config lock hash should match")
	}
}

func TestWarmRunner_RunWarmUp_WithFailures(t *testing.T) {
	// Create validation functions that fail
	dataValidator := func(dataPath string) error {
		return nil // Don't fail data validation in this test
	}
	
	modelTester := func(config interface{}) error {
		return fmt.Errorf("model validation intentionally failed")
	}
	
	runner := NewWarmRunner(dataValidator, modelTester)
	
	// Create a test config lock
	lock := &ConfigLock{
		Version:    "1.0.0",
		Timestamp:  time.Now(),
		ConfigHash: "test_hash",
		Config: BaselineModelConfig{
			ModelType: "linear",
		},
	}
	
	// Run warm-up (should fail)
	result, err := runner.RunWarmUp(lock, "test_data_path")
	if err != nil {
		t.Fatalf("Warm run should return result even on failure: %v", err)
	}
	
	// Validate failure result
	if result.Success {
		t.Error("Warm run should fail when model validation fails")
	}
	
	if result.ErrorMessage == "" {
		t.Error("Error message should be set on failure")
	}
	
	if result.Duration <= 0 {
		t.Error("Duration should still be recorded on failure")
	}
}