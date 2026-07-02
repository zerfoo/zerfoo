package cli

import (
	"context"
	"testing"
)

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

func TestWorkerCommand_Interface(t *testing.T) {
	var _ Command = (*WorkerCommand)(nil)
}
