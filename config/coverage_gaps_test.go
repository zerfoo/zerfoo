package config

import (
	"testing"
)

// ---------------------------------------------------------------------------
// LoadWithEnv: Load fails (missing file propagation)
// ---------------------------------------------------------------------------

func TestLoadWithEnv_LoadFails(t *testing.T) {
	_, err := LoadWithEnv[envConfig]("/nonexistent/path.json", "APP")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

// ---------------------------------------------------------------------------
// applyEnvOverrides: invalid bool env var
// ---------------------------------------------------------------------------

func TestLoadWithEnv_InvalidBoolEnv(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "cfg.json", `{"host":"localhost"}`)

	t.Setenv("APP_DEBUG", "not-a-bool")

	_, err := LoadWithEnv[envConfig](path, "APP")
	if err == nil {
		t.Error("expected error for non-boolean DEBUG env var")
	}
}
