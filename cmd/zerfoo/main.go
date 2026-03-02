package main

import (
	"context"
	"log"
	"os"

	"github.com/zerfoo/zerfoo/cmd/cli"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/shutdown"
)

func main() {
	if err := run(); err != nil {
		log.Printf("CLI execution failed: %v", err)
		os.Exit(1)
	}
}

func run() error {
	// Create shutdown coordinator and signal-aware context.
	coord := shutdown.New()
	ctx, cancel := cli.SignalContext(context.Background(), coord)
	defer cancel()

	// Create CLI application
	cliApp := cli.NewCLI()

	// Setup registries
	modelRegistry := model.Float32ModelRegistry

	// Register commands
	predictCmd := cli.NewPredictCommand(modelRegistry, func(f float64) float32 { return float32(f) }, func(v float32) float64 { return float64(v) })
	cliApp.RegisterCommand(predictCmd)

	tokenizeCmd := cli.NewTokenizeCommand()
	cliApp.RegisterCommand(tokenizeCmd)

	workerCmd := cli.NewWorkerCommand(coord)
	cliApp.RegisterCommand(workerCmd)

	// Run CLI
	return cliApp.Run(ctx, os.Args[1:])
}
