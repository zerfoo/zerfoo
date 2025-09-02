// Package cli provides a generic command-line interface framework for Zerfoo.
// This framework uses the plugin registry system to enable extensible, configurable CLI tools.
package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training"
)

// Command represents a generic CLI command with pluggable functionality.
type Command interface {
	// Name returns the command name
	Name() string

	// Description returns the command description
	Description() string

	// Run executes the command with the given arguments
	Run(ctx context.Context, args []string) error

	// Usage returns usage information
	Usage() string

	// Examples returns usage examples
	Examples() []string
}

// CommandRegistry manages available CLI commands.
type CommandRegistry struct {
	commands map[string]Command
}

// NewCommandRegistry creates a new command registry.
func NewCommandRegistry() *CommandRegistry {
	return &CommandRegistry{
		commands: make(map[string]Command),
	}
}

// Register adds a command to the registry.
func (r *CommandRegistry) Register(cmd Command) {
	r.commands[cmd.Name()] = cmd
}

// Get retrieves a command by name.
func (r *CommandRegistry) Get(name string) (Command, bool) {
	cmd, exists := r.commands[name]
	return cmd, exists
}

// List returns all registered command names.
func (r *CommandRegistry) List() []string {
	names := make([]string, 0, len(r.commands))
	for name := range r.commands {
		names = append(names, name)
	}
	return names
}

// BaseConfig provides common configuration options for CLI commands.
type BaseConfig struct {
	// Common options
	Verbose   bool   `json:"verbose"`
	Output    string `json:"output"`
	Format    string `json:"format"`    // "json", "yaml", "csv", "text"
	ConfigFile string `json:"config_file"`
	
	// Plugin configuration
	Plugins   map[string]interface{} `json:"plugins"`
	
	// Extension point for command-specific configuration
	Extensions map[string]interface{} `json:"extensions"`
}

// PredictCommand implements model prediction using the plugin system.
type PredictCommand[T tensor.Numeric] struct {
	modelRegistry    *model.ModelRegistry[T]
	defaultConfig    *PredictCommandConfig
}

// PredictCommandConfig configures model prediction.
type PredictCommandConfig struct {
	BaseConfig
	
	// Model configuration
	ModelPath     string `json:"model_path"`
	ModelProvider string `json:"model_provider"` // Registry key for model provider
	ModelConfig   map[string]interface{} `json:"model_config"`
	
	// Data configuration
	DataPath       string   `json:"data_path"`
	DataProvider   string   `json:"data_provider"`   // Registry key for data provider
	FeatureColumns []string `json:"feature_columns"`
	IDColumn       string   `json:"id_column"`
	GroupColumn    string   `json:"group_column"`
	
	// Prediction configuration
	BatchSize     int  `json:"batch_size"`
	IncludeProbs  bool `json:"include_probs"`
	Overwrite     bool `json:"overwrite"`
}

// NewPredictCommand creates a new predict command.
func NewPredictCommand[T tensor.Numeric](registry *model.ModelRegistry[T]) *PredictCommand[T] {
	return &PredictCommand[T]{
		modelRegistry: registry,
		defaultConfig: &PredictCommandConfig{
			BaseConfig: BaseConfig{
				Format:  "csv",
				Plugins: make(map[string]interface{}),
				Extensions: make(map[string]interface{}),
			},
			ModelProvider: "standard",
			DataProvider:  "csv",
			IDColumn:      "id",
			BatchSize:     10000,
		},
	}
}

// Name implements Command.Name
func (c *PredictCommand[T]) Name() string {
	return "predict"
}

// Description implements Command.Description
func (c *PredictCommand[T]) Description() string {
	return "Perform model inference on data using configurable model and data providers"
}

// Run implements Command.Run
func (c *PredictCommand[T]) Run(ctx context.Context, args []string) error {
	config, err := c.parseArgs(args)
	if err != nil {
		return fmt.Errorf("failed to parse arguments: %w", err)
	}
	
	if config.Verbose {
		fmt.Printf("Running prediction with config: %+v\n", config)
	}
	
	// Create model provider (placeholder - in production would use this)
	_, err = c.modelRegistry.GetModelProvider(ctx, config.ModelProvider, config.ModelConfig)
	if err != nil {
		return fmt.Errorf("failed to get model provider '%s': %w", config.ModelProvider, err)
	}
	
	// Load model
	modelLoader, err := c.modelRegistry.GetModelLoader(ctx, "zmf", nil)
	if err != nil {
		return fmt.Errorf("failed to get model loader: %w", err)
	}
	
	modelInstance, err := modelLoader.LoadFromPath(ctx, config.ModelPath)
	if err != nil {
		return fmt.Errorf("failed to load model from %s: %w", config.ModelPath, err)
	}
	
	if config.Verbose {
		metadata := modelInstance.GetMetadata()
		fmt.Printf("Loaded model: %s (version: %s, parameters: %d)\n", 
			metadata.Name, metadata.Version, metadata.Parameters)
	}
	
	// Run prediction
	result, err := c.runPrediction(ctx, config, modelInstance)
	if err != nil {
		return fmt.Errorf("prediction failed: %w", err)
	}
	
	// Save results
	if err := c.saveResults(config, result); err != nil {
		return fmt.Errorf("failed to save results: %w", err)
	}
	
	if config.Verbose {
		fmt.Printf("Prediction completed successfully. Results saved to: %s\n", config.Output)
	}
	
	return nil
}

// Usage implements Command.Usage
func (c *PredictCommand[T]) Usage() string {
	return `predict [OPTIONS]
  
Perform model inference using configurable providers.

OPTIONS:
  --model-path <path>       Path to model file (required)
  --data-path <path>        Path to input data (required)  
  --output <path>           Output path for predictions (required)
  --model-provider <name>   Model provider name (default: standard)
  --data-provider <name>    Data provider name (default: csv)
  --batch-size <int>        Prediction batch size (default: 10000)
  --format <format>         Output format: csv, json (default: csv)
  --include-probs           Include prediction probabilities
  --id-col <name>           ID column name (default: id)
  --group-col <name>        Optional grouping column name
  --verbose                 Verbose output
  --overwrite              Overwrite existing output
  --config <path>          Load configuration from file`
}

// Examples implements Command.Examples
func (c *PredictCommand[T]) Examples() []string {
	return []string{
		"predict --model-path model.zmf --data-path data.csv --output predictions.csv",
		"predict --model-path model.zmf --data-path data.csv --output pred.json --format json --include-probs",
		"predict --config predict_config.json --verbose",
	}
}

// TrainCommand implements model training using the plugin system.
type TrainCommand[T tensor.Numeric] struct {
	modelRegistry    *model.ModelRegistry[T]
	trainingRegistry *training.PluginRegistry[T]
	defaultConfig    *TrainCommandConfig
}

// TrainCommandConfig configures model training.
type TrainCommandConfig struct {
	BaseConfig
	
	// Data configuration
	DataPath      string `json:"data_path"`
	DataProvider  string `json:"data_provider"`
	
	// Model configuration
	ModelType     string `json:"model_type"`
	ModelProvider string `json:"model_provider"`
	ModelConfig   map[string]interface{} `json:"model_config"`
	
	// Training configuration
	TrainingWorkflow string `json:"training_workflow"`
	TrainingConfig   map[string]interface{} `json:"training_config"`
	
	// Output configuration
	OutputPath    string `json:"output_path"`
	SaveInterval  int    `json:"save_interval"`
}

// NewTrainCommand creates a new train command.
func NewTrainCommand[T tensor.Numeric](
	modelRegistry *model.ModelRegistry[T],
	trainingRegistry *training.PluginRegistry[T],
) *TrainCommand[T] {
	return &TrainCommand[T]{
		modelRegistry:    modelRegistry,
		trainingRegistry: trainingRegistry,
		defaultConfig: &TrainCommandConfig{
			BaseConfig: BaseConfig{
				Format:     "json",
				Plugins:    make(map[string]interface{}),
				Extensions: make(map[string]interface{}),
			},
			ModelProvider:    "standard",
			DataProvider:     "csv", 
			TrainingWorkflow: "standard",
		},
	}
}

// Name implements Command.Name
func (c *TrainCommand[T]) Name() string {
	return "train"
}

// Description implements Command.Description
func (c *TrainCommand[T]) Description() string {
	return "Train models using configurable training workflows and data providers"
}

// Run implements Command.Run
func (c *TrainCommand[T]) Run(ctx context.Context, args []string) error {
	return fmt.Errorf("training functionality has been moved to domain-specific applications - use audacity for Numerai training")
}

// Usage implements Command.Usage
func (c *TrainCommand[T]) Usage() string {
	return `train [OPTIONS]

DEPRECATED: Training functionality has been moved to domain-specific applications.
For Numerai tournament training, please use the audacity project.`
}

// Examples implements Command.Examples  
func (c *TrainCommand[T]) Examples() []string {
	return []string{
		"# Training has been moved to audacity",
		"cd ../audacity && go run cmd/numerai-train/main.go [options]",
	}
}

// Helper methods for PredictCommand

func (c *PredictCommand[T]) parseArgs(args []string) (*PredictCommandConfig, error) {
	config := *c.defaultConfig // Copy defaults
	
	// Simple argument parsing (in a real implementation, use flag or cobra)
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model-path":
			if i+1 >= len(args) {
				return nil, fmt.Errorf("--model-path requires a value")
			}
			config.ModelPath = args[i+1]
			i++
		case "--data-path":
			if i+1 >= len(args) {
				return nil, fmt.Errorf("--data-path requires a value")
			}
			config.DataPath = args[i+1]
			i++
		case "--output":
			if i+1 >= len(args) {
				return nil, fmt.Errorf("--output requires a value")
			}
			config.Output = args[i+1]
			i++
		case "--verbose":
			config.Verbose = true
		case "--overwrite":
			config.Overwrite = true
		case "--include-probs":
			config.IncludeProbs = true
		case "--config":
			if i+1 >= len(args) {
				return nil, fmt.Errorf("--config requires a value")
			}
			if err := c.loadConfig(args[i+1], &config); err != nil {
				return nil, fmt.Errorf("failed to load config: %w", err)
			}
			i++
		}
	}
	
	// Validate required parameters
	if config.ModelPath == "" {
		return nil, fmt.Errorf("--model-path is required")
	}
	if config.DataPath == "" {
		return nil, fmt.Errorf("--data-path is required")
	}
	if config.Output == "" {
		return nil, fmt.Errorf("--output is required")
	}
	
	return &config, nil
}

func (c *PredictCommand[T]) loadConfig(path string, config *PredictCommandConfig) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	
	return json.Unmarshal(data, config)
}

func (c *PredictCommand[T]) runPrediction(ctx context.Context, config *PredictCommandConfig, modelInstance model.ModelInstance[T]) (*PredictionResult, error) {
	startTime := time.Now()
	
	result := &PredictionResult{
		ModelPath:  config.ModelPath,
		DataPath:   config.DataPath,
		OutputPath: config.Output,
		Timestamp:  startTime,
		Config:     config,
		Success:    false,
	}
	
	// Placeholder implementation - in a real system, this would:
	// 1. Load data using the configured data provider
	// 2. Process data in batches
	// 3. Run model inference
	// 4. Collect predictions and statistics
	
	result.NumSamples = 10000    // Placeholder
	result.NumFeatures = 100     // Placeholder
	result.PredictionStats = map[string]float64{
		"mean": 0.5,
		"std":  0.1,
		"min":  0.0,
		"max":  1.0,
	}
	result.Duration = time.Since(startTime)
	result.Success = true
	
	return result, nil
}

func (c *PredictCommand[T]) saveResults(config *PredictCommandConfig, result *PredictionResult) error {
	// Create output directory
	outputDir := filepath.Dir(config.Output)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Check if output exists and overwrite flag
	if _, err := os.Stat(config.Output); err == nil && !config.Overwrite {
		return fmt.Errorf("output file exists and overwrite not enabled: %s", config.Output)
	}
	
	// Save results based on format
	switch strings.ToLower(config.Format) {
	case "json":
		return c.saveJSONResults(config, result)
	case "csv":
		return c.saveCSVResults(config, result)
	default:
		return fmt.Errorf("unsupported output format: %s", config.Format)
	}
}

func (c *PredictCommand[T]) saveJSONResults(config *PredictCommandConfig, result *PredictionResult) error {
	// Placeholder JSON output
	output := map[string]interface{}{
		"predictions": []map[string]interface{}{
			{"id": "sample_1", "prediction": 0.75},
			{"id": "sample_2", "prediction": 0.25},
		},
		"metadata": result,
	}
	
	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(config.Output, data, 0644)
}

func (c *PredictCommand[T]) saveCSVResults(config *PredictCommandConfig, result *PredictionResult) error {
	// Placeholder CSV output
	content := fmt.Sprintf("%s,prediction\n", config.IDColumn)
	content += "sample_1,0.75\n"
	content += "sample_2,0.25\n"
	
	return os.WriteFile(config.Output, []byte(content), 0644)
}

// PredictionResult contains prediction results and metadata (reused from original).
type PredictionResult struct {
	ModelPath       string                 `json:"model_path"`
	DataPath        string                 `json:"data_path"`
	OutputPath      string                 `json:"output_path"`
	Timestamp       time.Time              `json:"timestamp"`
	Config          *PredictCommandConfig  `json:"config"`
	NumSamples      int                    `json:"num_samples"`
	NumFeatures     int                    `json:"num_features"`
	PredictionStats map[string]float64     `json:"prediction_stats"`
	Duration        time.Duration          `json:"duration"`
	Success         bool                   `json:"success"`
	ErrorMessage    string                 `json:"error_message,omitempty"`
}

// TokenizeCommand implements text tokenization.
type TokenizeCommand struct {
	defaultText string
}

// NewTokenizeCommand creates a new tokenize command.
func NewTokenizeCommand() *TokenizeCommand {
	return &TokenizeCommand{}
}

// Name implements Command.Name
func (c *TokenizeCommand) Name() string {
	return "tokenize"
}

// Description implements Command.Description
func (c *TokenizeCommand) Description() string {
	return "Tokenize text using the Zerfoo tokenizer"
}

// Run implements Command.Run
func (c *TokenizeCommand) Run(ctx context.Context, args []string) error {
	// Simple argument parsing - in production would use proper flag parsing
	var text string
	
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--text":
			if i+1 >= len(args) {
				return fmt.Errorf("--text requires a value")
			}
			text = args[i+1]
			i++
		}
	}
	
	if text == "" {
		return fmt.Errorf("please provide text to tokenize using the --text flag")
	}
	
	// Use tokenizer (placeholder - would need proper tokenizer implementation)
	// For now, simple word tokenization
	words := strings.Fields(text)
	tokenIDs := make([]int, len(words))
	for i := range words {
		tokenIDs[i] = i + 1 // Simple sequential IDs
	}
	
	fmt.Printf("Token IDs for '%s': %v\n", text, tokenIDs)
	return nil
}

// Usage implements Command.Usage
func (c *TokenizeCommand) Usage() string {
	return `tokenize [OPTIONS]

Tokenize text using the Zerfoo tokenizer.

OPTIONS:
  --text <string>    Text to tokenize (required)`
}

// Examples implements Command.Examples
func (c *TokenizeCommand) Examples() []string {
	return []string{
		`tokenize --text "Hello world"`,
		`tokenize --text "The quick brown fox jumps over the lazy dog"`,
	}
}

// CLI provides the main command-line interface.
type CLI struct {
	registry *CommandRegistry
}

// NewCLI creates a new CLI instance.
func NewCLI() *CLI {
	return &CLI{
		registry: NewCommandRegistry(),
	}
}

// RegisterCommand adds a command to the CLI.
func (c *CLI) RegisterCommand(cmd Command) {
	c.registry.Register(cmd)
}

// Run executes a command based on arguments.
func (c *CLI) Run(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return c.printUsage()
	}
	
	cmdName := args[0]
	cmd, exists := c.registry.Get(cmdName)
	if !exists {
		return fmt.Errorf("unknown command: %s\n\nUse 'help' to see available commands", cmdName)
	}
	
	return cmd.Run(ctx, args[1:])
}

func (c *CLI) printUsage() error {
	fmt.Printf("Zerfoo CLI - Generic Machine Learning Framework\n\n")
	fmt.Printf("USAGE:\n")
	fmt.Printf("  zerfoo <command> [options]\n\n")
	fmt.Printf("AVAILABLE COMMANDS:\n")
	
	for _, name := range c.registry.List() {
		cmd, _ := c.registry.Get(name)
		fmt.Printf("  %-12s %s\n", name, cmd.Description())
	}
	
	fmt.Printf("\nUse 'zerfoo <command> --help' for more information about a command.\n")
	return nil
}