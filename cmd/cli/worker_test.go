package cli

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/distributed"
)

// fakeWorkerNode is a workerNode stand-in that records the config it was
// built with (via the closure in the test) and never touches the network,
// so tests can assert on distributed.WorkerNodeConfig.TLS without standing
// up a real gRPC server/coordinator pair.
type fakeWorkerNode struct{}

func (fakeWorkerNode) Start(context.Context) error { return nil }
func (fakeWorkerNode) Close(context.Context) error { return nil }

func TestWorkerCommand_Name(t *testing.T) {
	cmd := NewWorkerCommand(nil)
	if got := cmd.Name(); got != "worker" {
		t.Errorf("Name() = %q, want %q", got, "worker")
	}
}

func TestWorkerCommand_Description(t *testing.T) {
	cmd := NewWorkerCommand(nil)
	if got := cmd.Description(); got == "" {
		t.Error("Description() should not be empty")
	}
}

func TestWorkerCommand_Usage(t *testing.T) {
	cmd := NewWorkerCommand(nil)
	if got := cmd.Usage(); got == "" {
		t.Error("Usage() should not be empty")
	}
}

func TestWorkerCommand_Examples(t *testing.T) {
	cmd := NewWorkerCommand(nil)
	if got := cmd.Examples(); len(got) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestWorkerCommand_MissingCoordinatorAddress(t *testing.T) {
	cmd := NewWorkerCommand(nil)
	err := cmd.Run(context.Background(), []string{"--worker-address", "localhost:9001"})
	if err == nil {
		t.Fatal("expected error for missing --coordinator-address")
	}
}

func TestWorkerCommand_MissingWorkerAddress(t *testing.T) {
	cmd := NewWorkerCommand(nil)
	err := cmd.Run(context.Background(), []string{"--coordinator-address", "localhost:9000"})
	if err == nil {
		t.Fatal("expected error for missing --worker-address")
	}
}

func TestWorkerCommand_UnknownFlag(t *testing.T) {
	cmd := NewWorkerCommand(nil)
	err := cmd.Run(context.Background(), []string{"--unknown"})
	if err == nil {
		t.Fatal("expected error for unknown flag")
	}
}

func TestWorkerCommand_FlagRequiresValue(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"coordinator-address", []string{"--coordinator-address"}},
		{"worker-address", []string{"--worker-address"}},
		{"worker-id", []string{"--worker-id"}},
		{"world-size", []string{"--world-size"}},
		{"tls-cert", []string{"--tls-cert"}},
		{"tls-key", []string{"--tls-key"}},
		{"tls-ca", []string{"--tls-ca"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewWorkerCommand(nil)
			if err := cmd.Run(context.Background(), tt.args); err == nil {
				t.Errorf("expected error for %s without value", tt.name)
			}
		})
	}
}

func TestWorkerCommand_InvalidWorldSize(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{"non-numeric", "abc"},
		{"zero", "0"},
		{"negative", "-1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewWorkerCommand(nil)
			err := cmd.Run(context.Background(), []string{
				"--coordinator-address", "localhost:9000",
				"--worker-address", "localhost:9001",
				"--world-size", tt.value,
			})
			if err == nil {
				t.Errorf("expected error for --world-size %s", tt.value)
			}
		})
	}
}

func TestParsePositiveInt(t *testing.T) {
	tests := []struct {
		input string
		want  int
		err   bool
	}{
		{"1", 1, false},
		{"42", 42, false},
		{"100", 100, false},
		{"0", 0, true},
		{"abc", 0, true},
		{"-1", 0, true},
		{"", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := parsePositiveInt(tt.input)
			if (err != nil) != tt.err {
				t.Errorf("parsePositiveInt(%q) error = %v, wantErr %v", tt.input, err, tt.err)
			}
			if got != tt.want {
				t.Errorf("parsePositiveInt(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

// TestWorkerCommand_Run_TLSFlagsPopulateWorkerNodeConfig is T140.2 coverage:
// starting the worker CLI with --tls-cert/--tls-key/--tls-ca populated must
// actually result in a TLS-configured distributed.WorkerNodeConfig being
// handed to the worker node constructor, not just parsed and discarded. It
// substitutes a fakeWorkerNode (via the newWorkerNode hook) so the assertion
// is on the config Run() builds, without needing a live coordinator/gRPC
// server.
func TestWorkerCommand_Run_TLSFlagsPopulateWorkerNodeConfig(t *testing.T) {
	dir := t.TempDir()
	certPath := filepath.Join(dir, "cert.pem")
	keyPath := filepath.Join(dir, "key.pem")
	caPath := filepath.Join(dir, "ca.pem")

	var gotCfg distributed.WorkerNodeConfig
	cmd := NewWorkerCommand(nil)
	cmd.newWorkerNode = func(cfg distributed.WorkerNodeConfig) workerNode {
		gotCfg = cfg
		return fakeWorkerNode{}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Run blocks on <-ctx.Done() after Start(); pre-cancel so it returns immediately.

	if err := cmd.Run(ctx, []string{
		"--coordinator-address", "127.0.0.1:9000",
		"--worker-address", "127.0.0.1:9001",
		"--tls-cert", certPath,
		"--tls-key", keyPath,
		"--tls-ca", caPath,
	}); err != nil {
		t.Fatalf("Run() error = %v, want nil", err)
	}

	if gotCfg.TLS == nil {
		t.Fatal("WorkerNodeConfig.TLS is nil; want it populated from --tls-cert/--tls-key/--tls-ca")
	}
	if gotCfg.TLS.CertPath != certPath {
		t.Errorf("TLS.CertPath = %q, want %q", gotCfg.TLS.CertPath, certPath)
	}
	if gotCfg.TLS.KeyPath != keyPath {
		t.Errorf("TLS.KeyPath = %q, want %q", gotCfg.TLS.KeyPath, keyPath)
	}
	if gotCfg.TLS.CACertPath != caPath {
		t.Errorf("TLS.CACertPath = %q, want %q", gotCfg.TLS.CACertPath, caPath)
	}
}

// TestWorkerCommand_Run_NoTLSFlags_LeavesTLSNil documents the dev-loopback
// path: omitting all --tls-* flags must leave WorkerNodeConfig.TLS nil (the
// loopback-without-TLS behavior worker_node.go's Start() has always allowed).
func TestWorkerCommand_Run_NoTLSFlags_LeavesTLSNil(t *testing.T) {
	var gotCfg distributed.WorkerNodeConfig
	cmd := NewWorkerCommand(nil)
	cmd.newWorkerNode = func(cfg distributed.WorkerNodeConfig) workerNode {
		gotCfg = cfg
		return fakeWorkerNode{}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := cmd.Run(ctx, []string{
		"--coordinator-address", "127.0.0.1:9000",
		"--worker-address", "127.0.0.1:9001",
	}); err != nil {
		t.Fatalf("Run() error = %v, want nil", err)
	}

	if gotCfg.TLS != nil {
		t.Errorf("TLS = %+v, want nil when no --tls-* flags are given", gotCfg.TLS)
	}
}

// TestWorkerCommand_Run_PartialTLSFlags_Errors ensures a partial --tls-*
// flag combination (which would build a TLSConfig that mismatches what
// tlsconfig.go's ServerCredentials/ClientCredentials expect) is rejected up
// front rather than silently starting with a broken TLS configuration.
func TestWorkerCommand_Run_PartialTLSFlags_Errors(t *testing.T) {
	dir := t.TempDir()
	certPath := filepath.Join(dir, "cert.pem")
	keyPath := filepath.Join(dir, "key.pem")
	caPath := filepath.Join(dir, "ca.pem")

	tests := []struct {
		name string
		args []string
	}{
		{"cert only", []string{"--tls-cert", certPath}},
		{"key only", []string{"--tls-key", keyPath}},
		{"ca only", []string{"--tls-ca", caPath}},
		{"cert and key, no ca", []string{"--tls-cert", certPath, "--tls-key", keyPath}},
		{"cert and ca, no key", []string{"--tls-cert", certPath, "--tls-ca", caPath}},
		{"key and ca, no cert", []string{"--tls-key", keyPath, "--tls-ca", caPath}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewWorkerCommand(nil)
			cmd.newWorkerNode = func(cfg distributed.WorkerNodeConfig) workerNode {
				t.Fatal("newWorkerNode should not be called when TLS flags are incomplete")
				return fakeWorkerNode{}
			}

			args := append([]string{
				"--coordinator-address", "127.0.0.1:9000",
				"--worker-address", "127.0.0.1:9001",
			}, tt.args...)

			if err := cmd.Run(context.Background(), args); err == nil {
				t.Error("expected error for partial --tls-* flags")
			}
		})
	}
}

func TestBuildTLSConfig(t *testing.T) {
	t.Run("all empty returns nil config and nil error", func(t *testing.T) {
		cfg, err := buildTLSConfig("", "", "")
		if err != nil {
			t.Fatalf("buildTLSConfig() error = %v, want nil", err)
		}
		if cfg != nil {
			t.Errorf("buildTLSConfig() = %+v, want nil", cfg)
		}
	})

	t.Run("all three set builds a populated config", func(t *testing.T) {
		cfg, err := buildTLSConfig("cert.pem", "key.pem", "ca.pem")
		if err != nil {
			t.Fatalf("buildTLSConfig() error = %v, want nil", err)
		}
		if cfg == nil {
			t.Fatal("buildTLSConfig() = nil, want a populated *distributed.TLSConfig")
		}
		if cfg.CertPath != "cert.pem" || cfg.KeyPath != "key.pem" || cfg.CACertPath != "ca.pem" {
			t.Errorf("buildTLSConfig() = %+v, want CertPath=cert.pem KeyPath=key.pem CACertPath=ca.pem", cfg)
		}
	})

	t.Run("partial set errors", func(t *testing.T) {
		if _, err := buildTLSConfig("cert.pem", "", ""); err == nil {
			t.Error("expected error for cert-only")
		}
	})
}

func TestWorkerCommand_Interface(t *testing.T) {
	var _ Command = (*WorkerCommand)(nil)
}
