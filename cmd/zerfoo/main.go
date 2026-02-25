package main

import (
	"context"
	"log"
	"os"

	"github.com/zerfoo/zerfoo/cmd/cli"
	"github.com/zerfoo/zerfoo/model"
)

func main() {
	ctx := context.Background()

	// Create CLI application
	cliApp := cli.NewCLI()

	// Setup registries
	modelRegistry := model.Float32ModelRegistry

	// Register commands
	predictCmd := cli.NewPredictCommand(modelRegistry, func(f float64) float32 { return float32(f) }, func(v float32) float64 { return float64(v) })
	cliApp.RegisterCommand(predictCmd)

	tokenizeCmd := cli.NewTokenizeCommand()
	cliApp.RegisterCommand(tokenizeCmd)

	// Run CLI
	if err := cliApp.Run(ctx, os.Args[1:]); err != nil {
		log.Printf("CLI execution failed: %v", err)
		os.Exit(1)
	}
}
