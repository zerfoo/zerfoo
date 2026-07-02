package cli

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

func float32From(f float64) float32 { return float32(f) }
func float32To(v float32) float64   { return float64(v) }

func TestCLI(t *testing.T) {
	cliApp := NewCLI()

	predictCmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	cliApp.RegisterCommand(predictCmd)

	tokenizeCmd := NewTokenizeCommand()
	cliApp.RegisterCommand(tokenizeCmd)

	commands := cliApp.registry.List()
	if len(commands) != 2 {
		t.Errorf("Expected 2 commands, got %d", len(commands))
	}

	for _, name := range []string{"predict", "tokenize"} {
		if _, ok := cliApp.registry.Get(name); !ok {
			t.Errorf("Expected command %q to be registered", name)
		}
	}
}

func TestTokenizeCommand_NoVocab(t *testing.T) {
	cmd := NewTokenizeCommand()
	ctx := context.Background()

	// Without a vocab file, all words map to <unk>
	err := cmd.Run(ctx, []string{"--text", "Hello world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the tokenizer returns <unk> IDs for unknown words
	ids, encErr := cmd.Tok().Encode("Hello world")
	if encErr != nil {
		t.Fatalf("Encode error: %v", encErr)
	}
	if len(ids) != 2 {
		t.Fatalf("expected 2 token IDs, got %d", len(ids))
	}
	for i, id := range ids {
		if id != 0 {
			t.Errorf("token %d: expected <unk> ID=0, got %d", i, id)
		}
	}
}

func TestTokenizeCommand_WithVocab(t *testing.T) {
	dir := t.TempDir()
	vocabFile := filepath.Join(dir, "vocab.txt")
	err := os.WriteFile(vocabFile, []byte("hello\nworld\nfoo\nbar\n"), 0600)
	if err != nil {
		t.Fatalf("failed to write vocab file: %v", err)
	}

	cmd := NewTokenizeCommand()
	ctx := context.Background()

	err = cmd.Run(ctx, []string{"--vocab", vocabFile, "--text", "hello world baz"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the tokenizer loaded the vocabulary and produces correct IDs
	tok := cmd.Tok()
	ids, encErr := tok.Encode("hello world baz")
	if encErr != nil {
		t.Fatalf("Encode error: %v", encErr)
	}
	if len(ids) != 3 {
		t.Fatalf("expected 3 token IDs, got %d", len(ids))
	}
	// "hello" added after special tokens (<unk>=0, <s>=1, </s>=2, <pad>=3) -> ID 4
	if ids[0] != 4 {
		t.Errorf("expected hello ID=4, got %d", ids[0])
	}
	// "world" -> ID 5
	if ids[1] != 5 {
		t.Errorf("expected world ID=5, got %d", ids[1])
	}
	// "baz" is OOV -> ID 0 (<unk>)
	if ids[2] != 0 {
		t.Errorf("expected baz ID=0 (<unk>), got %d", ids[2])
	}
}

func TestTokenizeCommand_MissingText(t *testing.T) {
	cmd := NewTokenizeCommand()
	err := cmd.Run(context.Background(), []string{})
	if err == nil {
		t.Error("expected error for missing text argument")
	}
}

func TestTokenizeCommand_MissingVocabFile(t *testing.T) {
	cmd := NewTokenizeCommand()
	err := cmd.Run(context.Background(), []string{"--vocab", "/nonexistent/vocab.txt", "--text", "hello"})
	if err == nil {
		t.Error("expected error for missing vocab file")
	}
}

func TestPredictCommand_MissingArgs(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	ctx := context.Background()

	err := cmd.Run(ctx, []string{})
	if err == nil {
		t.Error("expected error for missing arguments")
	}

	// Missing data-path
	err = cmd.Run(ctx, []string{"--model-path", "m.zmf", "--output", "o.csv"})
	if err == nil {
		t.Error("expected error for missing data-path")
	}

	// Missing output
	err = cmd.Run(ctx, []string{"--model-path", "m.zmf", "--data-path", "d.csv"})
	if err == nil {
		t.Error("expected error for missing output")
	}
}

// --- Mock ModelInstance for testing ---

type mockModelInstance struct {
	forwardErr bool
	output     *tensor.TensorNumeric[float32]
	metadata   model.ModelMetadata
}

func (m *mockModelInstance) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if m.forwardErr {
		return nil, context.Canceled
	}
	return m.output, nil
}

func (m *mockModelInstance) Backward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) error {
	return nil
}

func (m *mockModelInstance) GetGraph() *graph.Graph[float32]         { return nil }
func (m *mockModelInstance) GetMetadata() model.ModelMetadata        { return m.metadata }
func (m *mockModelInstance) Parameters() []*graph.Parameter[float32] { return nil }
func (m *mockModelInstance) SetTrainingMode(_ bool)                  {}
func (m *mockModelInstance) IsTraining() bool                        { return false }

// --- PredictCommand accessor tests ---

func TestPredictCommand_Description(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	desc := cmd.Description()
	if desc == "" {
		t.Error("Description returned empty string")
	}
}

func TestPredictCommand_Usage(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	usage := cmd.Usage()
	if !strings.Contains(usage, "--model-path") {
		t.Error("Usage should mention --model-path")
	}
}

func TestPredictCommand_Examples(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	examples := cmd.Examples()
	if len(examples) == 0 {
		t.Error("Examples should not be empty")
	}
}

// --- TokenizeCommand accessor tests ---

func TestTokenizeCommand_Description(t *testing.T) {
	cmd := NewTokenizeCommand()
	desc := cmd.Description()
	if desc == "" {
		t.Error("Description returned empty string")
	}
}

func TestTokenizeCommand_Usage(t *testing.T) {
	cmd := NewTokenizeCommand()
	usage := cmd.Usage()
	if !strings.Contains(usage, "--text") {
		t.Error("Usage should mention --text")
	}
}

func TestTokenizeCommand_Examples(t *testing.T) {
	cmd := NewTokenizeCommand()
	examples := cmd.Examples()
	if len(examples) == 0 {
		t.Error("Examples should not be empty")
	}
}

// --- TokenizeCommand edge cases ---

func TestTokenizeCommand_MissingTextValue(t *testing.T) {
	cmd := NewTokenizeCommand()
	err := cmd.Run(context.Background(), []string{"--text"})
	if err == nil {
		t.Error("expected error for --text without value")
	}
}

func TestTokenizeCommand_MissingVocabValue(t *testing.T) {
	cmd := NewTokenizeCommand()
	err := cmd.Run(context.Background(), []string{"--vocab"})
	if err == nil {
		t.Error("expected error for --vocab without value")
	}
}

// --- parseArgs tests ---

func TestParseArgs_AllFlags(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config, err := cmd.parseArgs([]string{
		"--model-path", "m.zmf",
		"--data-path", "d.csv",
		"--output", "o.csv",
		"--verbose",
		"--overwrite",
		"--include-probs",
	})
	if err != nil {
		t.Fatalf("parseArgs failed: %v", err)
	}
	if !config.Verbose {
		t.Error("expected Verbose=true")
	}
	if !config.Overwrite {
		t.Error("expected Overwrite=true")
	}
	if !config.IncludeProbs {
		t.Error("expected IncludeProbs=true")
	}
	if config.ModelPath != "m.zmf" {
		t.Errorf("ModelPath = %q, want m.zmf", config.ModelPath)
	}
}

func TestParseArgs_MissingFlagValues(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)

	tests := []struct {
		name string
		args []string
	}{
		{"model-path", []string{"--model-path"}},
		{"data-path", []string{"--model-path", "m", "--data-path"}},
		{"output", []string{"--model-path", "m", "--data-path", "d", "--output"}},
		{"config", []string{"--model-path", "m", "--data-path", "d", "--output", "o", "--config"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := cmd.parseArgs(tt.args)
			if err == nil {
				t.Errorf("expected error for missing --%s value", tt.name)
			}
		})
	}
}

func TestParseArgs_WithConfigFile(t *testing.T) {
	dir := t.TempDir()
	configFile := filepath.Join(dir, "config.json")
	cfgData, _ := json.Marshal(PredictCommandConfig{
		BaseConfig: BaseConfig{Verbose: true},
		ModelPath:  "from-config.zmf",
		DataPath:   "from-config.csv",
	})
	if err := os.WriteFile(configFile, cfgData, 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	// --config first, then override output with flag
	config, err := cmd.parseArgs([]string{
		"--config", configFile,
		"--output", "o.csv",
	})
	if err != nil {
		t.Fatalf("parseArgs failed: %v", err)
	}
	if !config.Verbose {
		t.Error("expected Verbose=true from config file")
	}
	if config.ModelPath != "from-config.zmf" {
		t.Errorf("ModelPath = %q, want from-config.zmf", config.ModelPath)
	}
}

func TestParseArgs_BadConfigFile(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	_, err := cmd.parseArgs([]string{
		"--model-path", "m.zmf",
		"--data-path", "d.csv",
		"--output", "o.csv",
		"--config", "/nonexistent/config.json",
	})
	if err == nil {
		t.Error("expected error for nonexistent config file")
	}
}

// --- loadConfig tests ---

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	configFile := filepath.Join(dir, "config.json")

	original := PredictCommandConfig{
		BaseConfig: BaseConfig{
			Verbose: true,
			Format:  "json",
		},
		ModelPath: "test.zmf",
		BatchSize: 500,
	}
	data, _ := json.Marshal(original)
	if err := os.WriteFile(configFile, data, 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	var config PredictCommandConfig
	if err := cmd.loadConfig(configFile, &config); err != nil {
		t.Fatalf("loadConfig failed: %v", err)
	}

	if config.ModelPath != "test.zmf" {
		t.Errorf("ModelPath = %q, want test.zmf", config.ModelPath)
	}
	if config.BatchSize != 500 {
		t.Errorf("BatchSize = %d, want 500", config.BatchSize)
	}
}

func TestLoadConfig_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	configFile := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(configFile, []byte("{invalid"), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	var config PredictCommandConfig
	if err := cmd.loadConfig(configFile, &config); err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestLoadConfig_FileNotFound(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	var config PredictCommandConfig
	if err := cmd.loadConfig("/nonexistent.json", &config); err == nil {
		t.Error("expected error for nonexistent file")
	}
}

// --- readCSVData tests ---

func TestReadCSVData_Basic(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "id,f1,f2\na,1.0,2.0\nb,3.0,4.0\n"
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{
		BaseConfig: BaseConfig{},
	}
	config.DataPath = csvFile
	config.IDColumn = "id"

	ids, features, numFeatures, err := cmd.readCSVData(config)
	if err != nil {
		t.Fatalf("readCSVData failed: %v", err)
	}
	if len(ids) != 2 {
		t.Errorf("len(ids) = %d, want 2", len(ids))
	}
	if ids[0] != "a" || ids[1] != "b" {
		t.Errorf("ids = %v, want [a b]", ids)
	}
	if numFeatures != 2 {
		t.Errorf("numFeatures = %d, want 2", numFeatures)
	}
	want := []float64{1.0, 2.0, 3.0, 4.0}
	for i, f := range features {
		if math.Abs(f-want[i]) > 1e-9 {
			t.Errorf("features[%d] = %v, want %v", i, f, want[i])
		}
	}
}

func TestReadCSVData_ExplicitFeatureColumns(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "id,f1,f2,f3\na,1.0,2.0,3.0\n"
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{}
	config.DataPath = csvFile
	config.IDColumn = "id"
	config.FeatureColumns = []string{"f1", "f3"}

	ids, features, numFeatures, err := cmd.readCSVData(config)
	if err != nil {
		t.Fatalf("readCSVData failed: %v", err)
	}
	if len(ids) != 1 {
		t.Errorf("len(ids) = %d, want 1", len(ids))
	}
	if numFeatures != 2 {
		t.Errorf("numFeatures = %d, want 2", numFeatures)
	}
	// Should pick f1 (1.0) and f3 (3.0)
	if features[0] != 1.0 || features[1] != 3.0 {
		t.Errorf("features = %v, want [1 3]", features)
	}
}

func TestReadCSVData_NoIDColumn(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "f1,f2\n1.0,2.0\n3.0,4.0\n"
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{}
	config.DataPath = csvFile
	config.IDColumn = "id" // Not in CSV

	ids, _, _, err := cmd.readCSVData(config)
	if err != nil {
		t.Fatalf("readCSVData failed: %v", err)
	}
	// Should auto-generate row IDs
	if ids[0] != "row_0" || ids[1] != "row_1" {
		t.Errorf("ids = %v, want [row_0 row_1]", ids)
	}
}

func TestReadCSVData_NoFeatureColumns(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "id\na\nb\n"
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{}
	config.DataPath = csvFile
	config.IDColumn = "id"

	_, _, _, err := cmd.readCSVData(config)
	if err == nil {
		t.Error("expected error for no feature columns")
	}
}

func TestReadCSVData_FileNotFound(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{}
	config.DataPath = "/nonexistent/data.csv"

	_, _, _, err := cmd.readCSVData(config)
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestReadCSVData_InvalidValues(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "id,f1,f2\na,bad,2.0\n"
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{}
	config.DataPath = csvFile
	config.IDColumn = "id"

	_, features, _, err := cmd.readCSVData(config)
	if err != nil {
		t.Fatalf("readCSVData should not fail on bad values: %v", err)
	}
	// "bad" should be parsed as 0.0
	if features[0] != 0.0 {
		t.Errorf("features[0] = %v, want 0 (fallback for non-numeric)", features[0])
	}
}

// --- saveResults tests ---

func TestSaveResults_CSV(t *testing.T) {
	dir := t.TempDir()
	outputFile := filepath.Join(dir, "out.csv")

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{
		BaseConfig: BaseConfig{
			Output: outputFile,
			Format: "csv",
		},
		IDColumn: "id",
	}
	result := &PredictionResult{
		IDs:         []string{"a", "b"},
		Predictions: []float64{0.5, 0.9},
	}

	if err := cmd.saveResults(config, result); err != nil {
		t.Fatalf("saveResults failed: %v", err)
	}

	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "id,prediction") {
		t.Error("CSV should contain header")
	}
	if !strings.Contains(content, "a,") {
		t.Error("CSV should contain row a")
	}
}

func TestSaveResults_JSON(t *testing.T) {
	dir := t.TempDir()
	outputFile := filepath.Join(dir, "out.json")

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{
		BaseConfig: BaseConfig{
			Output: outputFile,
			Format: "json",
		},
	}
	result := &PredictionResult{
		IDs:         []string{"x"},
		Predictions: []float64{0.42},
	}

	if err := cmd.saveResults(config, result); err != nil {
		t.Fatalf("saveResults failed: %v", err)
	}

	data, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}
	if !strings.Contains(string(data), `"id"`) {
		t.Error("JSON should contain id field")
	}
}

func TestSaveResults_UnsupportedFormat(t *testing.T) {
	dir := t.TempDir()
	outputFile := filepath.Join(dir, "out.txt")

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{
		BaseConfig: BaseConfig{
			Output: outputFile,
			Format: "xml",
		},
	}
	result := &PredictionResult{}

	if err := cmd.saveResults(config, result); err == nil {
		t.Error("expected error for unsupported format")
	}
}

func TestSaveResults_ExistsNoOverwrite(t *testing.T) {
	dir := t.TempDir()
	outputFile := filepath.Join(dir, "out.csv")
	if err := os.WriteFile(outputFile, []byte("existing"), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{
		BaseConfig: BaseConfig{
			Output: outputFile,
			Format: "csv",
		},
		Overwrite: false,
	}
	result := &PredictionResult{}

	if err := cmd.saveResults(config, result); err == nil {
		t.Error("expected error when file exists and overwrite=false")
	}
}

func TestSaveResults_ExistsWithOverwrite(t *testing.T) {
	dir := t.TempDir()
	outputFile := filepath.Join(dir, "out.csv")
	if err := os.WriteFile(outputFile, []byte("existing"), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{
		BaseConfig: BaseConfig{
			Output: outputFile,
			Format: "csv",
		},
		IDColumn:  "id",
		Overwrite: true,
	}
	result := &PredictionResult{
		IDs:         []string{"a"},
		Predictions: []float64{1.0},
	}

	if err := cmd.saveResults(config, result); err != nil {
		t.Fatalf("saveResults with overwrite should succeed: %v", err)
	}
}

func TestSaveResults_PredictionsShorterThanIDs(t *testing.T) {
	dir := t.TempDir()

	// Test JSON path: IDs > Predictions (should default to 0.0)
	jsonFile := filepath.Join(dir, "out.json")
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{
		BaseConfig: BaseConfig{Output: jsonFile, Format: "json"},
	}
	result := &PredictionResult{
		IDs:         []string{"a", "b"},
		Predictions: []float64{0.5}, // Only 1 prediction for 2 IDs
	}
	if err := cmd.saveResults(config, result); err != nil {
		t.Fatalf("saveResults failed: %v", err)
	}

	// Test CSV path similarly
	csvFile := filepath.Join(dir, "out.csv")
	config2 := &PredictCommandConfig{
		BaseConfig: BaseConfig{Output: csvFile, Format: "csv"},
		IDColumn:   "id",
	}
	if err := cmd.saveResults(config2, result); err != nil {
		t.Fatalf("saveResults failed: %v", err)
	}
}

// --- runPrediction tests ---

func TestRunPrediction(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "id,f1,f2\na,1.0,2.0\nb,3.0,4.0\n"
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	outputTensor, _ := tensor.New[float32]([]int{2, 1}, []float32{0.5, 0.9})
	mock := &mockModelInstance{
		output:   outputTensor,
		metadata: model.ModelMetadata{Name: "test", Version: "1.0"},
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{
		IDColumn: "id",
	}
	config.DataPath = csvFile

	result, err := cmd.runPrediction(context.Background(), config, mock)
	if err != nil {
		t.Fatalf("runPrediction failed: %v", err)
	}
	if !result.Success {
		t.Error("expected Success=true")
	}
	if result.NumSamples != 2 {
		t.Errorf("NumSamples = %d, want 2", result.NumSamples)
	}
	if result.NumFeatures != 2 {
		t.Errorf("NumFeatures = %d, want 2", result.NumFeatures)
	}
	if len(result.Predictions) != 2 {
		t.Errorf("len(Predictions) = %d, want 2", len(result.Predictions))
	}
}

func TestRunPrediction_ForwardError(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "id,f1\na,1.0\n"
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	mock := &mockModelInstance{forwardErr: true}
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{IDColumn: "id"}
	config.DataPath = csvFile

	result, err := cmd.runPrediction(context.Background(), config, mock)
	if err == nil {
		t.Error("expected error from forward")
	}
	if result.Success {
		t.Error("expected Success=false on error")
	}
}

func TestRunPrediction_ReadDataError(t *testing.T) {
	mock := &mockModelInstance{}
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{IDColumn: "id"}
	config.DataPath = "/nonexistent.csv"

	result, err := cmd.runPrediction(context.Background(), config, mock)
	if err == nil {
		t.Error("expected error from missing data file")
	}
	if result.Success {
		t.Error("expected Success=false on error")
	}
}

// --- CLI.Run tests ---

func TestCLI_Run_NoArgs(t *testing.T) {
	cliApp := NewCLI()
	err := cliApp.Run(context.Background(), []string{})
	if err != nil {
		t.Errorf("Run with no args should return nil (print usage), got: %v", err)
	}
}

func TestCLI_Run_UnknownCommand(t *testing.T) {
	cliApp := NewCLI()
	err := cliApp.Run(context.Background(), []string{"nonexistent"})
	if err == nil {
		t.Error("expected error for unknown command")
	}
	if !strings.Contains(err.Error(), "unknown command") {
		t.Errorf("error should mention 'unknown command', got: %v", err)
	}
}

func TestCLI_Run_ValidCommand(t *testing.T) {
	cliApp := NewCLI()
	cliApp.RegisterCommand(NewTokenizeCommand())

	err := cliApp.Run(context.Background(), []string{"tokenize", "--text", "hello"})
	if err != nil {
		t.Errorf("Run valid command failed: %v", err)
	}
}

func TestCLI_PrintUsage_WithCommands(t *testing.T) {
	cliApp := NewCLI()
	cliApp.RegisterCommand(NewTokenizeCommand())
	// printUsage is called by Run with no args
	err := cliApp.Run(context.Background(), []string{})
	if err != nil {
		t.Errorf("printUsage should not return error: %v", err)
	}
}

// --- PredictCommand.Run integration (model provider not found path) ---

func TestPredictCommand_Run_ModelProviderError(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(csvFile, []byte("id,f1\na,1.0\n"), 0600); err != nil {
		t.Fatal(err)
	}

	err := cmd.Run(context.Background(), []string{
		"--model-path", "m.zmf",
		"--data-path", csvFile,
		"--output", filepath.Join(dir, "out.csv"),
	})
	// Should fail at model provider lookup since "standard" is not registered
	if err == nil {
		t.Error("expected error from model provider lookup")
	}
}

// --- Mock model provider and loader for full Run test ---

type mockModelProvider struct{}

func (m *mockModelProvider) CreateModel(_ context.Context, _ model.ModelConfig) (model.ModelInstance[float32], error) {
	return nil, nil
}

func (m *mockModelProvider) CreateFromGraph(_ context.Context, _ *graph.Graph[float32], _ model.ModelConfig) (model.ModelInstance[float32], error) {
	return nil, nil
}

func (m *mockModelProvider) GetCapabilities() model.ModelCapabilities {
	return model.ModelCapabilities{}
}

func (m *mockModelProvider) GetProviderInfo() model.ProviderInfo {
	return model.ProviderInfo{Name: "test"}
}

type mockModelLoader struct {
	instance model.ModelInstance[float32]
	loadErr  bool
}

func (m *mockModelLoader) LoadFromPath(_ context.Context, _ string) (model.ModelInstance[float32], error) {
	if m.loadErr {
		return nil, context.Canceled
	}
	return m.instance, nil
}

func (m *mockModelLoader) LoadFromReader(_ context.Context, _ io.Reader) (model.ModelInstance[float32], error) {
	return nil, nil
}

func (m *mockModelLoader) LoadFromBytes(_ context.Context, _ []byte) (model.ModelInstance[float32], error) {
	return nil, nil
}

func (m *mockModelLoader) SupportsFormat(_ string) bool { return true }
func (m *mockModelLoader) GetLoaderInfo() model.LoaderInfo {
	return model.LoaderInfo{Name: "test"}
}

func TestPredictCommand_Run_FullPath(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(csvFile, []byte("id,f1\na,1.0\nb,2.0\n"), 0600); err != nil {
		t.Fatal(err)
	}
	outputFile := filepath.Join(dir, "out.csv")

	outputTensor, _ := tensor.New[float32]([]int{2, 1}, []float32{0.5, 0.9})
	mockInst := &mockModelInstance{
		output:   outputTensor,
		metadata: model.ModelMetadata{Name: "test-model", Version: "1.0", Parameters: 100},
	}

	reg := model.NewModelRegistry[float32]()
	_ = reg.RegisterModelProvider("standard", func(_ context.Context, _ map[string]any) (model.ModelProvider[float32], error) {
		return &mockModelProvider{}, nil
	})
	_ = reg.RegisterModelLoader("gguf", func(_ context.Context, _ map[string]any) (model.ModelLoader[float32], error) {
		return &mockModelLoader{instance: mockInst}, nil
	})

	cmd := NewPredictCommand(reg, float32From, float32To)
	err := cmd.Run(context.Background(), []string{
		"--model-path", filepath.Join(dir, "model.zmf"),
		"--data-path", csvFile,
		"--output", outputFile,
		"--verbose",
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if _, err := os.Stat(outputFile); err != nil {
		t.Errorf("output file not created: %v", err)
	}
}

func TestPredictCommand_Run_LoadModelError(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(csvFile, []byte("id,f1\na,1.0\n"), 0600); err != nil {
		t.Fatal(err)
	}

	reg := model.NewModelRegistry[float32]()
	_ = reg.RegisterModelProvider("standard", func(_ context.Context, _ map[string]any) (model.ModelProvider[float32], error) {
		return &mockModelProvider{}, nil
	})
	_ = reg.RegisterModelLoader("gguf", func(_ context.Context, _ map[string]any) (model.ModelLoader[float32], error) {
		return &mockModelLoader{loadErr: true}, nil
	})

	cmd := NewPredictCommand(reg, float32From, float32To)
	err := cmd.Run(context.Background(), []string{
		"--model-path", "m.zmf",
		"--data-path", csvFile,
		"--output", filepath.Join(dir, "out.csv"),
	})
	if err == nil {
		t.Error("expected error from model loader")
	}
}

func TestPredictCommand_Run_SaveResultsError(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(csvFile, []byte("id,f1\na,1.0\n"), 0600); err != nil {
		t.Fatal(err)
	}
	// Create a file at the output path, and don't set overwrite
	outputFile := filepath.Join(dir, "out.csv")
	if err := os.WriteFile(outputFile, []byte("existing"), 0600); err != nil {
		t.Fatal(err)
	}

	outputTensor, _ := tensor.New[float32]([]int{1, 1}, []float32{0.5})
	mockInst := &mockModelInstance{
		output:   outputTensor,
		metadata: model.ModelMetadata{Name: "test"},
	}

	reg := model.NewModelRegistry[float32]()
	_ = reg.RegisterModelProvider("standard", func(_ context.Context, _ map[string]any) (model.ModelProvider[float32], error) {
		return &mockModelProvider{}, nil
	})
	_ = reg.RegisterModelLoader("gguf", func(_ context.Context, _ map[string]any) (model.ModelLoader[float32], error) {
		return &mockModelLoader{instance: mockInst}, nil
	})

	cmd := NewPredictCommand(reg, float32From, float32To)
	err := cmd.Run(context.Background(), []string{
		"--model-path", "m.zmf",
		"--data-path", csvFile,
		"--output", outputFile,
	})
	if err == nil {
		t.Error("expected error when output exists and overwrite=false")
	}
}

func TestPredictCommand_Run_ForwardError(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(csvFile, []byte("id,f1\na,1.0\n"), 0600); err != nil {
		t.Fatal(err)
	}

	mockInst := &mockModelInstance{
		forwardErr: true,
		metadata:   model.ModelMetadata{Name: "test"},
	}

	reg := model.NewModelRegistry[float32]()
	_ = reg.RegisterModelProvider("standard", func(_ context.Context, _ map[string]any) (model.ModelProvider[float32], error) {
		return &mockModelProvider{}, nil
	})
	_ = reg.RegisterModelLoader("gguf", func(_ context.Context, _ map[string]any) (model.ModelLoader[float32], error) {
		return &mockModelLoader{instance: mockInst}, nil
	})

	cmd := NewPredictCommand(reg, float32From, float32To)
	err := cmd.Run(context.Background(), []string{
		"--model-path", "m.zmf",
		"--data-path", csvFile,
		"--output", filepath.Join(dir, "out.csv"),
		"--verbose",
	})
	if err == nil {
		t.Error("expected error from forward")
	}
}

// --- readCSVData: header read error ---

func TestReadCSVData_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(csvFile, []byte(""), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config := &PredictCommandConfig{}
	config.DataPath = csvFile
	config.IDColumn = "id"

	_, _, _, err := cmd.readCSVData(config)
	if err == nil {
		t.Error("expected error for empty CSV file (no header)")
	}
}

// --- --help flag tests ---

// newTestCLI creates a CLI that writes to the given buffer instead of stdout.
func newTestCLI(buf *strings.Builder) *CLI {
	c := NewCLI()
	c.out = buf
	return c
}

func TestCLI_GlobalHelp(t *testing.T) {
	var buf strings.Builder
	cliApp := newTestCLI(&buf)
	cliApp.RegisterCommand(NewTokenizeCommand())

	err := cliApp.Run(context.Background(), []string{"--help"})
	if err != nil {
		t.Fatalf("--help returned error: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "AVAILABLE COMMANDS") {
		t.Error("global --help should list available commands")
	}
	if !strings.Contains(out, "tokenize") {
		t.Error("global --help should list the tokenize command")
	}
}

func TestCLI_GlobalHelpShortFlag(t *testing.T) {
	var buf strings.Builder
	cliApp := newTestCLI(&buf)
	cliApp.RegisterCommand(NewTokenizeCommand())

	err := cliApp.Run(context.Background(), []string{"-h"})
	if err != nil {
		t.Fatalf("-h returned error: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "AVAILABLE COMMANDS") {
		t.Error("-h should list available commands")
	}
}

func TestCLI_CommandHelp(t *testing.T) {
	var buf strings.Builder
	cliApp := newTestCLI(&buf)
	cliApp.RegisterCommand(NewTokenizeCommand())

	err := cliApp.Run(context.Background(), []string{"tokenize", "--help"})
	if err != nil {
		t.Fatalf("tokenize --help returned error: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "USAGE:") {
		t.Error("command --help should contain USAGE section")
	}
	if !strings.Contains(out, "--text") {
		t.Error("tokenize --help should mention --text flag")
	}
	if !strings.Contains(out, "EXAMPLES:") {
		t.Error("command --help should contain EXAMPLES section")
	}
}

func TestCLI_CommandHelpShortFlag(t *testing.T) {
	var buf strings.Builder
	cliApp := newTestCLI(&buf)
	cliApp.RegisterCommand(NewTokenizeCommand())

	err := cliApp.Run(context.Background(), []string{"tokenize", "-h"})
	if err != nil {
		t.Fatalf("tokenize -h returned error: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "--text") {
		t.Error("tokenize -h should mention --text flag")
	}
}

func TestCLI_PredictCommandHelp(t *testing.T) {
	var buf strings.Builder
	cliApp := newTestCLI(&buf)
	cliApp.RegisterCommand(NewPredictCommand(model.Float32ModelRegistry, float32From, float32To))

	err := cliApp.Run(context.Background(), []string{"predict", "--help"})
	if err != nil {
		t.Fatalf("predict --help returned error: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "--model-path") {
		t.Error("predict --help should mention --model-path")
	}
	if !strings.Contains(out, "EXAMPLES:") {
		t.Error("predict --help should contain EXAMPLES section")
	}
}

func TestCLI_HelpDoesNotRunCommand(t *testing.T) {
	var buf strings.Builder
	cliApp := newTestCLI(&buf)
	cliApp.RegisterCommand(NewTokenizeCommand())

	// Without --help, tokenize with no --text would error.
	// With --help, it should print help and NOT error.
	err := cliApp.Run(context.Background(), []string{"tokenize", "--help"})
	if err != nil {
		t.Fatalf("--help should not run the command: %v", err)
	}
}

// --- splitFlag tests ---

func TestSplitFlag(t *testing.T) {
	tests := []struct {
		input     string
		wantFlag  string
		wantValue string
		wantOK    bool
	}{
		{"--temperature=0.7", "--temperature", "0.7", true},
		{"--model-path=/tmp/model.gguf", "--model-path", "/tmp/model.gguf", true},
		{"--text=hello world", "--text", "hello world", true},
		{"--verbose", "--verbose", "", false},
		{"--output=", "--output", "", true}, // empty value after =
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			flag, value, ok := splitFlag(tt.input)
			if flag != tt.wantFlag {
				t.Errorf("flag = %q, want %q", flag, tt.wantFlag)
			}
			if value != tt.wantValue {
				t.Errorf("value = %q, want %q", value, tt.wantValue)
			}
			if ok != tt.wantOK {
				t.Errorf("ok = %v, want %v", ok, tt.wantOK)
			}
		})
	}
}

// --- --flag=value syntax tests ---

func TestParseArgs_EqualsSyntax(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config, err := cmd.parseArgs([]string{
		"--model-path=m.zmf",
		"--data-path=d.csv",
		"--output=o.csv",
		"--verbose",
	})
	if err != nil {
		t.Fatalf("parseArgs with = syntax failed: %v", err)
	}
	if config.ModelPath != "m.zmf" {
		t.Errorf("ModelPath = %q, want m.zmf", config.ModelPath)
	}
	if config.DataPath != "d.csv" {
		t.Errorf("DataPath = %q, want d.csv", config.DataPath)
	}
	if config.Output != "o.csv" {
		t.Errorf("Output = %q, want o.csv", config.Output)
	}
	if !config.Verbose {
		t.Error("expected Verbose=true")
	}
}

func TestParseArgs_MixedSyntax(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	config, err := cmd.parseArgs([]string{
		"--model-path=m.zmf",
		"--data-path", "d.csv",
		"--output=o.csv",
	})
	if err != nil {
		t.Fatalf("parseArgs with mixed syntax failed: %v", err)
	}
	if config.ModelPath != "m.zmf" {
		t.Errorf("ModelPath = %q, want m.zmf", config.ModelPath)
	}
	if config.DataPath != "d.csv" {
		t.Errorf("DataPath = %q, want d.csv", config.DataPath)
	}
	if config.Output != "o.csv" {
		t.Errorf("Output = %q, want o.csv", config.Output)
	}
}

func TestTokenizeCommand_EqualsSyntax(t *testing.T) {
	cmd := NewTokenizeCommand()
	err := cmd.Run(context.Background(), []string{"--text=Hello world"})
	if err != nil {
		t.Fatalf("tokenize with --text=value failed: %v", err)
	}
}

func TestTokenizeCommand_EqualsSyntaxWithVocab(t *testing.T) {
	dir := t.TempDir()
	vocabFile := filepath.Join(dir, "vocab.txt")
	if err := os.WriteFile(vocabFile, []byte("hello\nworld\n"), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewTokenizeCommand()
	err := cmd.Run(context.Background(), []string{
		"--vocab=" + vocabFile,
		"--text=hello world",
	})
	if err != nil {
		t.Fatalf("tokenize with --vocab=path --text=value failed: %v", err)
	}
}
