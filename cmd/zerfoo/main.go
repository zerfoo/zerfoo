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
	predictCmd := cli.NewPredictCommand(modelRegistry)
	cliApp.RegisterCommand(predictCmd)

	tokenizeCmd := cli.NewTokenizeCommand()
	cliApp.RegisterCommand(tokenizeCmd)

	// Run CLI
	if err := cliApp.Run(ctx, os.Args[1:]); err != nil {
		log.Printf("CLI execution failed: %v", err)
		os.Exit(1)
	}
}
