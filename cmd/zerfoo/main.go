package main

import (
	"context"
	"log"
	"os"

	"github.com/zerfoo/zerfoo/cmd/cli"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/training"
)

func main() {
	ctx := context.Background()
	
	// Create CLI application
	cliApp := cli.NewCLI()
	
	// Setup registries
	modelRegistry := model.Float32ModelRegistry
	trainingRegistry := training.Float32Registry
	
	// Register commands
	predictCmd := cli.NewPredictCommand(modelRegistry)
	cliApp.RegisterCommand(predictCmd)
	
	trainCmd := cli.NewTrainCommand(modelRegistry, trainingRegistry)
	cliApp.RegisterCommand(trainCmd)
	
	tokenizeCmd := cli.NewTokenizeCommand()
	cliApp.RegisterCommand(tokenizeCmd)
	
	// Run CLI
	if err := cliApp.Run(ctx, os.Args[1:]); err != nil {
		log.Printf("CLI execution failed: %v", err)
		os.Exit(1)
	}
}