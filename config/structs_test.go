package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEngineConfig_LoadAndValidate(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "engine.json", `{
		"device": "cuda",
		"memory_limit_mb": 4096,
		"log_level": "info"
	}`)

	cfg, err := Load[EngineConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Device != "cuda" {
		t.Errorf("Device = %q, want %q", cfg.Device, "cuda")
	}
	if cfg.MemoryLimitMB != 4096 {
		t.Errorf("MemoryLimitMB = %d, want 4096", cfg.MemoryLimitMB)
	}
	if cfg.LogLevel != "info" {
		t.Errorf("LogLevel = %q, want %q", cfg.LogLevel, "info")
	}

	errs := Validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %v", errs)
	}
}

func TestEngineConfig_RequiredMissing(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "engine.json", `{}`)

	cfg, err := Load[EngineConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	errs := Validate(cfg)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
}

func TestEngineConfig_EnvOverride(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "engine.json", `{"device":"cpu"}`)

	t.Setenv("ZF_DEVICE", "cuda")
	t.Setenv("ZF_MEMORY_LIMIT_MB", "8192")

	cfg, err := LoadWithEnv[EngineConfig](path, "ZF")
	if err != nil {
		t.Fatalf("LoadWithEnv: %v", err)
	}

	if cfg.Device != "cuda" {
		t.Errorf("Device = %q, want %q", cfg.Device, "cuda")
	}
	if cfg.MemoryLimitMB != 8192 {
		t.Errorf("MemoryLimitMB = %d, want 8192", cfg.MemoryLimitMB)
	}
}

func TestTrainingConfig_LoadAndValidate(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "train.json", `{
		"batch_size": 32,
		"learning_rate": "0.001",
		"optimizer": "adam",
		"epochs": 10,
		"checkpoint_interval": 5
	}`)

	cfg, err := Load[TrainingConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.BatchSize != 32 {
		t.Errorf("BatchSize = %d, want 32", cfg.BatchSize)
	}
	if cfg.LearningRate != "0.001" {
		t.Errorf("LearningRate = %q, want %q", cfg.LearningRate, "0.001")
	}
	if cfg.Optimizer != "adam" {
		t.Errorf("Optimizer = %q, want %q", cfg.Optimizer, "adam")
	}
	if cfg.Epochs != 10 {
		t.Errorf("Epochs = %d, want 10", cfg.Epochs)
	}

	errs := Validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %v", errs)
	}
}

func TestTrainingConfig_RequiredMissing(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "train.json", `{}`)

	cfg, err := Load[TrainingConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	errs := Validate(cfg)
	if len(errs) != 3 {
		t.Fatalf("expected 3 errors, got %d: %v", len(errs), errs)
	}
}

func TestDistributedConfig_LoadAndValidate(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "dist.json", `{
		"coordinator_address": "localhost:50051",
		"timeout_seconds": 30,
		"tls_enabled": true
	}`)

	cfg, err := Load[DistributedConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.CoordinatorAddress != "localhost:50051" {
		t.Errorf("CoordinatorAddress = %q, want %q", cfg.CoordinatorAddress, "localhost:50051")
	}
	if cfg.TimeoutSeconds != 30 {
		t.Errorf("TimeoutSeconds = %d, want 30", cfg.TimeoutSeconds)
	}
	if !cfg.TLSEnabled {
		t.Error("TLSEnabled = false, want true")
	}

	errs := Validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %v", errs)
	}
}

func TestDistributedConfig_RequiredMissing(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "dist.json", `{}`)

	cfg, err := Load[DistributedConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	errs := Validate(cfg)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
}

func TestDistributedConfig_EnvOverride(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "dist.json", `{"coordinator_address":"file-addr"}`)

	t.Setenv("DIST_COORDINATOR_ADDRESS", "env-addr")
	t.Setenv("DIST_TIMEOUT_SECONDS", "60")
	t.Setenv("DIST_TLS_ENABLED", "true")

	cfg, err := LoadWithEnv[DistributedConfig](path, "DIST")
	if err != nil {
		t.Fatalf("LoadWithEnv: %v", err)
	}

	if cfg.CoordinatorAddress != "env-addr" {
		t.Errorf("CoordinatorAddress = %q, want %q", cfg.CoordinatorAddress, "env-addr")
	}
	if cfg.TimeoutSeconds != 60 {
		t.Errorf("TimeoutSeconds = %d, want 60", cfg.TimeoutSeconds)
	}
	if !cfg.TLSEnabled {
		t.Error("TLSEnabled = false, want true")
	}
}

func TestAllConfigs_FromSameFile(t *testing.T) {
	dir := t.TempDir()
	content := `{
		"device": "cpu",
		"memory_limit_mb": 2048,
		"log_level": "debug",
		"batch_size": 64,
		"learning_rate": "0.01",
		"optimizer": "sgd",
		"epochs": 5,
		"checkpoint_interval": 2,
		"coordinator_address": "10.0.0.1:50051",
		"timeout_seconds": 15,
		"tls_enabled": false
	}`
	path := filepath.Join(dir, "all.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	eng, err := Load[EngineConfig](path)
	if err != nil {
		t.Fatalf("Load EngineConfig: %v", err)
	}
	if eng.Device != "cpu" {
		t.Errorf("EngineConfig.Device = %q, want %q", eng.Device, "cpu")
	}

	train, err := Load[TrainingConfig](path)
	if err != nil {
		t.Fatalf("Load TrainingConfig: %v", err)
	}
	if train.BatchSize != 64 {
		t.Errorf("TrainingConfig.BatchSize = %d, want 64", train.BatchSize)
	}

	dist, err := Load[DistributedConfig](path)
	if err != nil {
		t.Fatalf("Load DistributedConfig: %v", err)
	}
	if dist.CoordinatorAddress != "10.0.0.1:50051" {
		t.Errorf("DistributedConfig.CoordinatorAddress = %q, want %q", dist.CoordinatorAddress, "10.0.0.1:50051")
	}
}
