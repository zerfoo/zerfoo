package cli

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/training"
)

func TestCLI(t *testing.T) {
	// Create CLI
	cliApp := NewCLI()
	
	// Register commands
	predictCmd := NewPredictCommand(model.Float32ModelRegistry)
	cliApp.RegisterCommand(predictCmd)
	
	trainCmd := NewTrainCommand(model.Float32ModelRegistry, training.Float32Registry)
	cliApp.RegisterCommand(trainCmd)
	
	tokenizeCmd := NewTokenizeCommand()
	cliApp.RegisterCommand(tokenizeCmd)
	
	// Test that commands are registered
	commands := cliApp.registry.List()
	expectedCommands := []string{"predict", "train", "tokenize"}
	
	if len(commands) != len(expectedCommands) {
		t.Errorf("Expected %d commands, got %d", len(expectedCommands), len(commands))
	}
	
	for _, expected := range expectedCommands {
		found := false
		for _, cmd := range commands {
			if cmd == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected command '%s' not found", expected)
		}
	}
}

func TestTokenizeCommand(t *testing.T) {
	cmd := NewTokenizeCommand()
	ctx := context.Background()
	
	// Test successful tokenization
	args := []string{"--text", "Hello world"}
	err := cmd.Run(ctx, args)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	
	// Test missing text argument
	err = cmd.Run(ctx, []string{})
	if err == nil {
		t.Error("Expected error for missing text argument")
	}
}

func TestTrainCommandDeprecation(t *testing.T) {
	cmd := NewTrainCommand(model.Float32ModelRegistry, training.Float32Registry)
	ctx := context.Background()
	
	// Train command should return deprecation error
	err := cmd.Run(ctx, []string{})
	if err == nil {
		t.Error("Expected deprecation error")
	}
	
	expectedMsg := "training functionality has been moved to domain-specific applications - use audacity for Numerai training"
	if err.Error() != expectedMsg {
		t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
	}
}