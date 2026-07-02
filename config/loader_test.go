package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type simpleConfig struct {
	Name    string `json:"name"`
	Port    int    `json:"port"`
	Verbose bool   `json:"verbose"`
}

type requiredConfig struct {
	Host string `json:"host" validate:"required"`
	Port int    `json:"port" validate:"required"`
}

type envConfig struct {
	Host    string `env:"HOST"     json:"host"`
	Port    int    `env:"PORT"     json:"port"`
	Debug   bool   `env:"DEBUG"    json:"debug"`
	Timeout int    `json:"timeout"`
}

func writeJSON(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoad_ValidJSON(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "cfg.json", `{"name":"test","port":8080,"verbose":true}`)

	cfg, err := Load[simpleConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Name != "test" {
		t.Errorf("Name = %q, want %q", cfg.Name, "test")
	}
	if cfg.Port != 8080 {
		t.Errorf("Port = %d, want 8080", cfg.Port)
	}
	if !cfg.Verbose {
		t.Error("Verbose = false, want true")
	}
}

func TestLoad_MissingFile(t *testing.T) {
	_, err := Load[simpleConfig]("/nonexistent/path.json")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoad_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "bad.json", `{invalid`)

	_, err := Load[simpleConfig](path)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestLoad_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "empty.json", `{}`)

	cfg, err := Load[simpleConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Name != "" {
		t.Errorf("Name = %q, want empty", cfg.Name)
	}
}

func TestValidate_RequiredPresent(t *testing.T) {
	cfg := requiredConfig{Host: "localhost", Port: 8080}
	errs := Validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %v", errs)
	}
}

func TestValidate_RequiredMissing(t *testing.T) {
	cfg := requiredConfig{}
	errs := Validate(cfg)
	if len(errs) != 2 {
		t.Fatalf("expected 2 errors, got %d: %v", len(errs), errs)
	}

	joined := strings.Join(errs, "; ")
	if !strings.Contains(joined, "Host") {
		t.Errorf("expected Host error, got %q", joined)
	}
	if !strings.Contains(joined, "Port") {
		t.Errorf("expected Port error, got %q", joined)
	}
}

func TestValidate_RequiredPartial(t *testing.T) {
	cfg := requiredConfig{Host: "localhost"}
	errs := Validate(cfg)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if !strings.Contains(errs[0], "Port") {
		t.Errorf("expected Port error, got %q", errs[0])
	}
}

func TestLoadWithEnv_OverridesString(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "cfg.json", `{"host":"file-host","port":80}`)

	t.Setenv("MYAPP_HOST", "env-host")

	cfg, err := LoadWithEnv[envConfig](path, "MYAPP")
	if err != nil {
		t.Fatalf("LoadWithEnv: %v", err)
	}

	if cfg.Host != "env-host" {
		t.Errorf("Host = %q, want %q", cfg.Host, "env-host")
	}
	if cfg.Port != 80 {
		t.Errorf("Port = %d, want 80 (not overridden)", cfg.Port)
	}
}

func TestLoadWithEnv_OverridesInt(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "cfg.json", `{"host":"localhost","port":80}`)

	t.Setenv("APP_PORT", "9090")

	cfg, err := LoadWithEnv[envConfig](path, "APP")
	if err != nil {
		t.Fatalf("LoadWithEnv: %v", err)
	}

	if cfg.Port != 9090 {
		t.Errorf("Port = %d, want 9090", cfg.Port)
	}
}

func TestLoadWithEnv_OverridesBool(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "cfg.json", `{"host":"localhost"}`)

	t.Setenv("APP_DEBUG", "true")

	cfg, err := LoadWithEnv[envConfig](path, "APP")
	if err != nil {
		t.Fatalf("LoadWithEnv: %v", err)
	}

	if !cfg.Debug {
		t.Error("Debug = false, want true")
	}
}

func TestLoadWithEnv_NoEnvVars(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "cfg.json", `{"host":"from-file","port":3000}`)

	cfg, err := LoadWithEnv[envConfig](path, "NOEXIST")
	if err != nil {
		t.Fatalf("LoadWithEnv: %v", err)
	}

	if cfg.Host != "from-file" {
		t.Errorf("Host = %q, want %q", cfg.Host, "from-file")
	}
}

func TestLoadWithEnv_InvalidIntEnv(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "cfg.json", `{"host":"localhost"}`)

	t.Setenv("APP_PORT", "not-a-number")

	_, err := LoadWithEnv[envConfig](path, "APP")
	if err == nil {
		t.Error("expected error for non-numeric PORT env var")
	}
}

func TestValidate_NoValidateTags(t *testing.T) {
	cfg := simpleConfig{Name: "x"}
	errs := Validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected 0 errors for struct without validate tags, got %v", errs)
	}
}

func TestLoadAndValidate(t *testing.T) {
	dir := t.TempDir()
	path := writeJSON(t, dir, "cfg.json", `{"host":"localhost","port":8080}`)

	cfg, err := Load[requiredConfig](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	errs := Validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected 0 validation errors, got %v", errs)
	}
}
