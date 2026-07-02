// Package cli provides the command-line interface framework for Zerfoo. (Stability: stable)
//
// # Architecture
//
// The package is built around three abstractions:
//
//   - [Command] — the interface every CLI command implements (Name, Description,
//     Run, Usage, Examples).
//   - [CommandRegistry] — a name-keyed store of Command values. Use [NewCommandRegistry]
//     to create one, then call Register and Get.
//   - [CLI] — the top-level dispatcher. It owns a CommandRegistry, matches the
//     first positional argument to a registered command, and delegates execution.
//
// # Built-in commands
//
// The following commands ship with Zerfoo:
//
//   - run       — interactive prompt-response generation ([RunCommand])
//   - serve     — start an OpenAI-compatible HTTP server ([ServeCommand])
//   - pull      — download and cache a model from a registry ([PullCommand])
//   - list      — list locally cached models ([ListCommand])
//   - rm        — remove a cached model ([RmCommand])
//   - worker    — start a distributed training worker ([WorkerCommand])
//   - predict   — batch model inference on CSV/JSON data ([PredictCommand])
//   - tokenize  — tokenize text with the Zerfoo tokenizer ([TokenizeCommand])
//
// # Adding a new command
//
// Implement the [Command] interface and register it with the CLI:
//
//	type MyCommand struct{}
//
//	func (c *MyCommand) Name() string        { return "mycommand" }
//	func (c *MyCommand) Description() string { return "Do something useful" }
//	func (c *MyCommand) Usage() string       { return "mycommand [OPTIONS]" }
//	func (c *MyCommand) Examples() []string  { return nil }
//
//	func (c *MyCommand) Run(ctx context.Context, args []string) error {
//	    // Command logic here.
//	    return nil
//	}
//
// Then register it during CLI setup:
//
//	app := cli.NewCLI()
//	app.RegisterCommand(&MyCommand{})
//
// # Configuration
//
// [BaseConfig] provides common options (verbose, output path, format, config
// file) that command-specific configs can embed. See [PredictCommandConfig]
// for an example of extending BaseConfig with domain-specific fields.
//
// # Signal handling
//
// [SignalContext] creates a context that cancels on SIGINT/SIGTERM and
// optionally triggers a [shutdown.Coordinator] for graceful shutdown.
// Long-running commands (serve, worker) use this to clean up on exit.
// Stability: stable
package cli
