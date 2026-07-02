package main

import (
	"context"
	"log"
	"os"

	"github.com/zerfoo/zerfoo/cmd/cli"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/serve/shutdown"
)

// version is set at build time via -ldflags "-X main.version=...".
var version string

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

	pullCmd := cli.NewPullCommand(nil, os.Stdout)
	cliApp.RegisterCommand(pullCmd)

	listCmd := cli.NewListCommand(nil, os.Stdout)
	cliApp.RegisterCommand(listCmd)

	rmCmd := cli.NewRmCommand(nil, os.Stdout)
	cliApp.RegisterCommand(rmCmd)

	runCmd := cli.NewRunCommand(os.Stdin, os.Stdout)
	cliApp.RegisterCommand(runCmd)

	serveCmd := cli.NewServeCommand(coord, os.Stdout)
	cliApp.RegisterCommand(serveCmd)

	versionCmd := cli.NewVersionCommand(version, os.Stdout)
	cliApp.RegisterCommand(versionCmd)

	automlCmd := cli.NewAutoMLCommand(os.Stdout)
	cliApp.RegisterCommand(automlCmd)

	trainCmd := cli.NewTrainCommand(os.Stdout)
	cliApp.RegisterCommand(trainCmd)

	guardCmd := cli.NewGuardCommand(os.Stdout)
	cliApp.RegisterCommand(guardCmd)

	sentimentCmd := cli.NewSentimentCommand(os.Stdout)
	cliApp.RegisterCommand(sentimentCmd)

	finetuneSentimentCmd := cli.NewFineTuneSentimentCommand(os.Stdout)
	cliApp.RegisterCommand(finetuneSentimentCmd)

	transmlaCmd := cli.NewTransMLACommand(os.Stdout)
	cliApp.RegisterCommand(transmlaCmd)

	eagleTrainCmd := cli.NewEagleTrainCommand(os.Stdout)
	cliApp.RegisterCommand(eagleTrainCmd)

	transcribeCmd := cli.NewTranscribeCommand(os.Stdout)
	cliApp.RegisterCommand(transcribeCmd)

	// Run CLI
	return cliApp.Run(ctx, os.Args[1:])
}
