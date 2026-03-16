package cli

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/ztensor/tensor"
	tokenizer "github.com/zerfoo/ztoken"
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

// List returns all registered command names in sorted order.
func (r *CommandRegistry) List() []string {
	names := make([]string, 0, len(r.commands))
	for name := range r.commands {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// BaseConfig provides common configuration options for CLI commands.
type BaseConfig struct {
	// Common options
	Verbose    bool   `json:"verbose"`
	Output     string `json:"output"`
	Format     string `json:"format"` // "json", "yaml", "csv", "text"
	ConfigFile string `json:"configFile"`

	// Plugin configuration
	Plugins map[string]interface{} `json:"plugins"`

	// Extension point for command-specific configuration
	Extensions map[string]interface{} `json:"extensions"`
}

// PredictCommand implements model prediction using the plugin system.
type PredictCommand[T tensor.Numeric] struct {
	modelRegistry *model.ModelRegistry[T]
	fromFloat64   func(float64) T
	toFloat64     func(T) float64
	defaultConfig *PredictCommandConfig
}

// PredictCommandConfig configures model prediction.
type PredictCommandConfig struct {
	BaseConfig

	// Model configuration
	ModelPath     string                 `json:"modelPath"`
	ModelProvider string                 `json:"modelProvider"` // Registry key for model provider
	ModelConfig   map[string]interface{} `json:"modelConfig"`

	// Data configuration
	DataPath       string   `json:"dataPath"`
	DataProvider   string   `json:"dataProvider"` // Registry key for data provider
	FeatureColumns []string `json:"featureColumns"`
	IDColumn       string   `json:"idColumn"`
	GroupColumn    string   `json:"groupColumn"`

	// Prediction configuration
	BatchSize    int  `json:"batchSize"`
	IncludeProbs bool `json:"includeProbs"`
	Overwrite    bool `json:"overwrite"`
}

// NewPredictCommand creates a new predict command.
// fromFloat64 converts a float64 CSV value to type T.
// toFloat64 converts a prediction value of type T back to float64 for output.
func NewPredictCommand[T tensor.Numeric](registry *model.ModelRegistry[T], fromFloat64 func(float64) T, toFloat64 func(T) float64) *PredictCommand[T] {
	return &PredictCommand[T]{
		modelRegistry: registry,
		fromFloat64:   fromFloat64,
		toFloat64:     toFloat64,
		defaultConfig: &PredictCommandConfig{
			BaseConfig: BaseConfig{
				Format:     "csv",
				Plugins:    make(map[string]interface{}),
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
	modelLoader, err := c.modelRegistry.GetModelLoader(ctx, "gguf", nil)
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
		"predict --model-path model.gguf --data-path data.csv --output predictions.csv",
		"predict --model-path model.gguf --data-path data.csv --output pred.json --format json --include-probs",
		"predict --config predict_config.json --verbose",
	}
}

// Helper methods for PredictCommand

func (c *PredictCommand[T]) parseArgs(args []string) (*PredictCommandConfig, error) {
	config := *c.defaultConfig // Copy defaults

	// Simple argument parsing (in a real implementation, use flag or cobra)
	for i := 0; i < len(args); i++ {
		arg := args[i]
		var eqVal string
		var hasEq bool
		if flag, val, ok := splitFlag(arg); ok {
			arg = flag
			eqVal = val
			hasEq = true
		}
		nextVal := func(flagName string) (string, error) {
			if hasEq {
				return eqVal, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", flagName)
			}
			i++
			return args[i], nil
		}
		switch arg {
		case "--model-path":
			v, err := nextVal("--model-path")
			if err != nil {
				return nil, err
			}
			config.ModelPath = v
		case "--data-path":
			v, err := nextVal("--data-path")
			if err != nil {
				return nil, err
			}
			config.DataPath = v
		case "--output":
			v, err := nextVal("--output")
			if err != nil {
				return nil, err
			}
			config.Output = v
		case "--verbose":
			config.Verbose = true
		case "--overwrite":
			config.Overwrite = true
		case "--include-probs":
			config.IncludeProbs = true
		case "--config":
			v, err := nextVal("--config")
			if err != nil {
				return nil, err
			}
			if err := c.loadConfig(v, &config); err != nil {
				return nil, fmt.Errorf("failed to load config: %w", err)
			}
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
	// The path is provided by the user of the CLI tool.
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

	// Read CSV data
	ids, features, numFeatures, err := c.readCSVData(config)
	if err != nil {
		return result, fmt.Errorf("failed to read data: %w", err)
	}
	result.NumSamples = len(ids)
	result.NumFeatures = numFeatures

	// Convert features to tensor of type T and run model forward
	data := make([]T, len(features))
	for i, f := range features {
		data[i] = c.fromFloat64(f)
	}

	inputTensor, err := tensor.New[T]([]int{len(ids), numFeatures}, data)
	if err != nil {
		return result, fmt.Errorf("failed to create input tensor: %w", err)
	}

	output, err := modelInstance.Forward(ctx, inputTensor)
	if err != nil {
		return result, fmt.Errorf("model forward failed: %w", err)
	}

	// Extract predictions as float64 values
	outputData := output.Data()
	predictions := make([]float64, len(outputData))
	for i, v := range outputData {
		predictions[i] = c.toFloat64(v)
	}

	result.Predictions = predictions
	result.IDs = ids
	result.Duration = time.Since(startTime)
	result.Success = true

	return result, nil
}

// readCSVData reads a CSV file and returns sample IDs, flattened features, and
// the number of feature columns.
func (c *PredictCommand[T]) readCSVData(config *PredictCommandConfig) (ids []string, features []float64, numFeatures int, err error) {
	file, err := os.Open(config.DataPath)
	if err != nil {
		return nil, nil, 0, err
	}
	defer file.Close() //nolint:errcheck

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, nil, 0, fmt.Errorf("failed to read CSV header: %w", err)
	}

	// Determine which columns are the ID and which are features
	idIdx := -1
	featureIdxs := make([]int, 0)
	for i, col := range header {
		col = strings.TrimSpace(col)
		if col == config.IDColumn {
			idIdx = i
			continue
		}
		if len(config.FeatureColumns) > 0 {
			for _, fc := range config.FeatureColumns {
				if col == fc {
					featureIdxs = append(featureIdxs, i)
					break
				}
			}
		} else {
			// Auto-detect: all non-ID columns are features
			featureIdxs = append(featureIdxs, i)
		}
	}

	numFeatures = len(featureIdxs)
	if numFeatures == 0 {
		return nil, nil, 0, fmt.Errorf("no feature columns found in CSV")
	}

	// Read rows
	for {
		record, readErr := reader.Read()
		if readErr != nil {
			break // EOF or error
		}

		// Extract ID
		sampleID := ""
		if idIdx >= 0 && idIdx < len(record) {
			sampleID = record[idIdx]
		} else {
			sampleID = fmt.Sprintf("row_%d", len(ids))
		}
		ids = append(ids, sampleID)

		// Extract features
		for _, fi := range featureIdxs {
			if fi < len(record) {
				val, parseErr := strconv.ParseFloat(strings.TrimSpace(record[fi]), 64)
				if parseErr != nil {
					val = 0.0
				}
				features = append(features, val)
			} else {
				features = append(features, 0.0)
			}
		}
	}

	return ids, features, numFeatures, nil
}

func (c *PredictCommand[T]) saveResults(config *PredictCommandConfig, result *PredictionResult) error {
	// Create output directory
	outputDir := filepath.Dir(config.Output)
	if err := os.MkdirAll(outputDir, 0750); err != nil {
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
	type predictionRow struct {
		ID         string  `json:"id"`
		Prediction float64 `json:"prediction"`
	}

	rows := make([]predictionRow, len(result.IDs))
	for i, id := range result.IDs {
		pred := 0.0
		if i < len(result.Predictions) {
			pred = result.Predictions[i]
		}
		rows[i] = predictionRow{ID: id, Prediction: pred}
	}

	data, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(config.Output, data, 0600)
}

func (c *PredictCommand[T]) saveCSVResults(config *PredictCommandConfig, result *PredictionResult) error {
	file, err := os.Create(config.Output)
	if err != nil {
		return err
	}
	defer file.Close() //nolint:errcheck

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.Write([]string{config.IDColumn, "prediction"}); err != nil {
		return err
	}

	for i, id := range result.IDs {
		pred := 0.0
		if i < len(result.Predictions) {
			pred = result.Predictions[i]
		}
		if err := writer.Write([]string{id, strconv.FormatFloat(pred, 'f', 6, 64)}); err != nil {
			return err
		}
	}

	return nil
}

// PredictionResult contains prediction results and metadata.
type PredictionResult struct {
	ModelPath   string                `json:"modelPath"`
	DataPath    string                `json:"dataPath"`
	OutputPath  string                `json:"outputPath"`
	Timestamp   time.Time             `json:"timestamp"`
	Config      *PredictCommandConfig `json:"config"`
	NumSamples  int                   `json:"numSamples"`
	NumFeatures int                   `json:"numFeatures"`
	Predictions []float64             `json:"predictions,omitempty"`
	IDs         []string              `json:"ids,omitempty"`
	Duration    time.Duration         `json:"duration"`
	Success     bool                  `json:"success"`
}

// TokenizeCommand implements text tokenization.
type TokenizeCommand struct {
	tok *tokenizer.WhitespaceTokenizer
}

// NewTokenizeCommand creates a new tokenize command.
func NewTokenizeCommand() *TokenizeCommand {
	return &TokenizeCommand{
		tok: tokenizer.NewWhitespaceTokenizer(),
	}
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
func (c *TokenizeCommand) Run(_ context.Context, args []string) error {
	var text, vocabPath string

	for i := 0; i < len(args); i++ {
		arg := args[i]
		var eqVal string
		var hasEq bool
		if flag, val, ok := splitFlag(arg); ok {
			arg = flag
			eqVal = val
			hasEq = true
		}
		nextVal := func(flagName string) (string, error) {
			if hasEq {
				return eqVal, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", flagName)
			}
			i++
			return args[i], nil
		}
		switch arg {
		case "--text":
			v, err := nextVal("--text")
			if err != nil {
				return err
			}
			text = v
		case "--vocab":
			v, err := nextVal("--vocab")
			if err != nil {
				return err
			}
			vocabPath = v
		}
	}

	if text == "" {
		return fmt.Errorf("please provide text to tokenize using the --text flag")
	}

	// Load vocabulary from file if provided
	if vocabPath != "" {
		if err := c.loadVocab(vocabPath); err != nil {
			return fmt.Errorf("failed to load vocabulary: %w", err)
		}
	}

	tokenIDs, err := c.tok.Encode(text)
	if err != nil {
		return fmt.Errorf("tokenization failed: %w", err)
	}
	fmt.Printf("Token IDs for '%s': %v\n", text, tokenIDs)
	return nil
}

// loadVocab loads a vocabulary file (one token per line) into the tokenizer.
func (c *TokenizeCommand) loadVocab(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close() //nolint:errcheck

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		token := strings.TrimSpace(scanner.Text())
		if token != "" {
			c.tok.AddToken(token)
		}
	}
	return scanner.Err()
}

// Tok returns the underlying tokenizer for testing.
func (c *TokenizeCommand) Tok() *tokenizer.WhitespaceTokenizer {
	return c.tok
}

// Usage implements Command.Usage
func (c *TokenizeCommand) Usage() string {
	return `tokenize [OPTIONS]

Tokenize text using the Zerfoo tokenizer.

OPTIONS:
  --text <string>    Text to tokenize (required)
  --vocab <path>     Path to vocabulary file (one token per line)`
}

// Examples implements Command.Examples
func (c *TokenizeCommand) Examples() []string {
	return []string{
		`tokenize --text "Hello world"`,
		`tokenize --text "The quick brown fox jumps over the lazy dog"`,
	}
}

// splitFlag checks whether arg contains an "=" (e.g. "--flag=value") and, if
// so, returns the flag name and value separately. When no "=" is present it
// returns the original arg and an empty string with ok=false.
func splitFlag(arg string) (flag, value string, ok bool) {
	if idx := strings.Index(arg, "="); idx >= 0 {
		return arg[:idx], arg[idx+1:], true
	}
	return arg, "", false
}

// CLI provides the main command-line interface.
type CLI struct {
	registry *CommandRegistry
	out      io.Writer
}

// NewCLI creates a new CLI instance.
func NewCLI() *CLI {
	return &CLI{
		registry: NewCommandRegistry(),
		out:      os.Stdout,
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

	if args[0] == "--help" || args[0] == "-h" {
		return c.printUsage()
	}

	cmdName := args[0]
	cmd, exists := c.registry.Get(cmdName)
	if !exists {
		return fmt.Errorf("unknown command: %s\n\nUse 'help' to see available commands", cmdName)
	}

	for _, arg := range args[1:] {
		if arg == "--help" || arg == "-h" {
			return c.printCommandHelp(cmd)
		}
	}

	return cmd.Run(ctx, args[1:])
}

func (c *CLI) printUsage() error {
	fmt.Fprintf(c.out, "Zerfoo CLI - Generic Machine Learning Framework\n\n")
	fmt.Fprintf(c.out, "USAGE:\n")
	fmt.Fprintf(c.out, "  zerfoo <command> [options]\n\n")
	fmt.Fprintf(c.out, "AVAILABLE COMMANDS:\n")

	for _, name := range c.registry.List() {
		cmd, _ := c.registry.Get(name)
		fmt.Fprintf(c.out, "  %-12s %s\n", name, cmd.Description())
	}

	fmt.Fprintf(c.out, "\nUse 'zerfoo <command> --help' for more information about a command.\n")
	return nil
}

func (c *CLI) printCommandHelp(cmd Command) error {
	fmt.Fprintf(c.out, "USAGE:\n  %s\n", cmd.Usage())

	examples := cmd.Examples()
	if len(examples) > 0 {
		fmt.Fprintf(c.out, "\nEXAMPLES:\n")
		for _, ex := range examples {
			fmt.Fprintf(c.out, "  %s\n", ex)
		}
	}

	return nil
}
