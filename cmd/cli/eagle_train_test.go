package cli

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
	modelgguf "github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestEagleTrainCommand_Name(t *testing.T) {
	cmd := NewEagleTrainCommand(&bytes.Buffer{})
	if got := cmd.Name(); got != "eagle-train" {
		t.Errorf("Name() = %q, want %q", got, "eagle-train")
	}
}

func TestEagleTrainCommand_Description(t *testing.T) {
	cmd := NewEagleTrainCommand(&bytes.Buffer{})
	if got := cmd.Description(); got == "" {
		t.Error("Description() should not be empty")
	}
}

func TestEagleTrainCommand_Usage(t *testing.T) {
	cmd := NewEagleTrainCommand(&bytes.Buffer{})
	usage := cmd.Usage()
	if usage == "" {
		t.Fatal("Usage() should not be empty")
	}
	for _, flag := range []string{"--model", "--corpus", "--output", "--epochs", "--lr", "--max-samples", "--batch-size", "--hidden-dim", "--synthetic"} {
		if !strings.Contains(usage, flag) {
			t.Errorf("Usage() missing flag %s", flag)
		}
	}
}

func TestEagleTrainCommand_Examples(t *testing.T) {
	cmd := NewEagleTrainCommand(&bytes.Buffer{})
	if got := cmd.Examples(); len(got) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestEagleTrainCommand_Interface(t *testing.T) {
	var _ Command = (*EagleTrainCommand)(nil)
}

func TestEagleTrainCommand_ParseArgs_Defaults(t *testing.T) {
	cmd := NewEagleTrainCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{"--synthetic"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.outputPath != "eagle.gguf" {
		t.Errorf("outputPath = %q, want %q", cfg.outputPath, "eagle.gguf")
	}
	if cfg.epochs != 3 {
		t.Errorf("epochs = %d, want 3", cfg.epochs)
	}
	if cfg.lr != 0.001 {
		t.Errorf("lr = %f, want 0.001", cfg.lr)
	}
	if cfg.maxSamples != 10000 {
		t.Errorf("maxSamples = %d, want 10000", cfg.maxSamples)
	}
	if cfg.batchSize != 32 {
		t.Errorf("batchSize = %d, want 32", cfg.batchSize)
	}
	if cfg.hiddenDim != 256 {
		t.Errorf("hiddenDim = %d, want 256", cfg.hiddenDim)
	}
	if !cfg.synthetic {
		t.Error("synthetic should be true")
	}
}

func TestEagleTrainCommand_ParseArgs_AllFlags(t *testing.T) {
	cmd := NewEagleTrainCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{
		"--model", "m.gguf",
		"--corpus", "data.txt",
		"--output", "out.gguf",
		"--epochs", "5",
		"--lr", "0.01",
		"--max-samples", "500",
		"--batch-size", "16",
		"--hidden-dim", "128",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.modelPath != "m.gguf" {
		t.Errorf("modelPath = %q, want %q", cfg.modelPath, "m.gguf")
	}
	if cfg.corpusPath != "data.txt" {
		t.Errorf("corpusPath = %q, want %q", cfg.corpusPath, "data.txt")
	}
	if cfg.outputPath != "out.gguf" {
		t.Errorf("outputPath = %q, want %q", cfg.outputPath, "out.gguf")
	}
	if cfg.epochs != 5 {
		t.Errorf("epochs = %d, want 5", cfg.epochs)
	}
	if cfg.lr != 0.01 {
		t.Errorf("lr = %f, want 0.01", cfg.lr)
	}
	if cfg.maxSamples != 500 {
		t.Errorf("maxSamples = %d, want 500", cfg.maxSamples)
	}
	if cfg.batchSize != 16 {
		t.Errorf("batchSize = %d, want 16", cfg.batchSize)
	}
	if cfg.hiddenDim != 128 {
		t.Errorf("hiddenDim = %d, want 128", cfg.hiddenDim)
	}
}

func TestEagleTrainCommand_ParseArgs_EqualsSyntax(t *testing.T) {
	cmd := NewEagleTrainCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{
		"--synthetic",
		"--epochs=7",
		"--lr=0.005",
		"--hidden-dim=64",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.epochs != 7 {
		t.Errorf("epochs = %d, want 7", cfg.epochs)
	}
	if cfg.lr != 0.005 {
		t.Errorf("lr = %f, want 0.005", cfg.lr)
	}
	if cfg.hiddenDim != 64 {
		t.Errorf("hiddenDim = %d, want 64", cfg.hiddenDim)
	}
}

func TestEagleTrainCommand_ParseArgs_MissingRequired(t *testing.T) {
	cmd := NewEagleTrainCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{})
	if err == nil {
		t.Fatal("expected error for missing required flags")
	}
	if !strings.Contains(err.Error(), "required") {
		t.Errorf("error = %q, want message about required flags", err.Error())
	}
}

func TestEagleTrainCommand_ParseArgs_InvalidValues(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"bad epochs", []string{"--synthetic", "--epochs", "0"}, "--epochs must be >= 1"},
		{"bad epochs text", []string{"--synthetic", "--epochs", "abc"}, "--epochs must be >= 1"},
		{"bad lr zero", []string{"--synthetic", "--lr", "0"}, "--lr must be a positive number"},
		{"bad lr negative", []string{"--synthetic", "--lr", "-1"}, "--lr must be a positive number"},
		{"bad lr text", []string{"--synthetic", "--lr", "abc"}, "--lr must be a positive number"},
		{"bad max-samples", []string{"--synthetic", "--max-samples", "0"}, "--max-samples must be >= 1"},
		{"bad batch-size", []string{"--synthetic", "--batch-size", "-1"}, "--batch-size must be >= 1"},
		{"bad hidden-dim", []string{"--synthetic", "--hidden-dim", "0"}, "--hidden-dim must be >= 1"},
		{"unknown flag", []string{"--synthetic", "--foo"}, "unknown flag"},
		{"missing value", []string{"--synthetic", "--epochs"}, "requires a value"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewEagleTrainCommand(&bytes.Buffer{})
			_, err := cmd.parseArgs(tt.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want substring %q", err.Error(), tt.want)
			}
		})
	}
}

func TestEagleTrainCommand_SyntheticTrainingConverges(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewEagleTrainCommand(&buf)

	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "eagle_test.gguf")

	err := cmd.Run(context.Background(), []string{
		"--synthetic",
		"--hidden-dim", "8",
		"--max-samples", "20",
		"--epochs", "3",
		"--batch-size", "4",
		"--lr", "0.001",
		"--output", outputPath,
	})
	if err != nil {
		t.Fatalf("Run() error: %v", err)
	}

	output := buf.String()

	// Verify all 3 epochs were logged.
	if !strings.Contains(output, "epoch=1/3") {
		t.Error("missing epoch 1 log")
	}
	if !strings.Contains(output, "epoch=3/3") {
		t.Error("missing epoch 3 log")
	}

	// Verify GGUF was saved.
	if !strings.Contains(output, "EAGLE head saved to") {
		t.Error("missing GGUF save confirmation")
	}

	// Verify the output file exists.
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		t.Error("output GGUF file does not exist")
	}
}

func TestEagleTrainCommand_GGUFExportRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewEagleTrainCommand(&buf)

	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "eagle_roundtrip.gguf")

	err := cmd.Run(context.Background(), []string{
		"--synthetic",
		"--hidden-dim", "16",
		"--max-samples", "10",
		"--epochs", "1",
		"--batch-size", "5",
		"--lr", "0.001",
		"--output", outputPath,
	})
	if err != nil {
		t.Fatalf("Run() error: %v", err)
	}

	// Open and parse the GGUF file.
	f, err := os.Open(outputPath)
	if err != nil {
		t.Fatalf("open GGUF: %v", err)
	}
	defer f.Close()

	gf, err := modelgguf.Parse(f)
	if err != nil {
		t.Fatalf("parse GGUF: %v", err)
	}

	// Verify metadata.
	arch, ok := gf.Metadata["general.architecture"]
	if !ok {
		t.Fatal("missing general.architecture metadata")
	}
	if arch != "eagle" {
		t.Errorf("general.architecture = %v, want %q", arch, "eagle")
	}

	hiddenDim, ok := gf.Metadata["eagle.hidden_dim"]
	if !ok {
		t.Fatal("missing eagle.hidden_dim metadata")
	}
	if hiddenDim != uint32(16) {
		t.Errorf("eagle.hidden_dim = %v, want 16", hiddenDim)
	}

	// Verify tensor names.
	expectedTensors := map[string]bool{
		"eagle.norm.weight": false,
		"eagle.norm.bias":   false,
		"eagle.fc1.weight":  false,
		"eagle.fc2.weight":  false,
	}
	for _, ti := range gf.Tensors {
		if _, ok := expectedTensors[ti.Name]; ok {
			expectedTensors[ti.Name] = true
		}
	}
	for name, found := range expectedTensors {
		if !found {
			t.Errorf("missing tensor %q in GGUF", name)
		}
	}

	// Load tensors and verify we can reconstruct the EAGLE head via LoadEAGLEWeights.
	tensorMap, loadErr := modelgguf.LoadTensors(gf, f)
	if loadErr != nil {
		t.Fatalf("load tensors: %v", loadErr)
	}

	if !inference.HasEAGLEWeights(tensorMap) {
		t.Error("HasEAGLEWeights returned false for exported GGUF")
	}

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	head, loadHeadErr := inference.LoadEAGLEWeights(tensorMap, engine, ops)
	if loadHeadErr != nil {
		t.Fatalf("LoadEAGLEWeights: %v", loadHeadErr)
	}
	if head == nil {
		t.Fatal("LoadEAGLEWeights returned nil head")
	}
}

func TestEagleTrainCommand_HelpViaRegistry(t *testing.T) {
	var buf bytes.Buffer
	app := NewCLI()
	app.out = &buf
	app.RegisterCommand(NewEagleTrainCommand(&buf))

	err := app.Run(context.Background(), []string{"eagle-train", "--help"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "--model") {
		t.Error("--help output should contain --model")
	}
}

func TestGenerateSyntheticPairs(t *testing.T) {
	pairs, err := inference.GenerateSyntheticPairs(8, 5)
	if err != nil {
		t.Fatalf("GenerateSyntheticPairs error: %v", err)
	}
	if len(pairs) != 5 {
		t.Errorf("got %d pairs, want 5", len(pairs))
	}
	for i, p := range pairs {
		if got := p.Input.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] != 8 {
			t.Errorf("pair %d input shape = %v, want [1,1,8]", i, got)
		}
		if got := p.Target.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] != 8 {
			t.Errorf("pair %d target shape = %v, want [1,1,8]", i, got)
		}
	}
}

func TestGenerateSyntheticPairs_InvalidArgs(t *testing.T) {
	tests := []struct {
		name      string
		hiddenDim int
		count     int
	}{
		{"zero dim", 0, 5},
		{"negative dim", -1, 5},
		{"zero count", 8, 0},
		{"negative count", 8, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := inference.GenerateSyntheticPairs(tt.hiddenDim, tt.count)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestCollectPenultimateFeatures_NotImplemented(t *testing.T) {
	_, err := inference.CollectPenultimateFeatures("model.gguf", []int{1, 2, 3}, 10)
	if err == nil {
		t.Fatal("expected error for unimplemented function")
	}
	if !strings.Contains(err.Error(), "not yet implemented") {
		t.Errorf("error = %q, want 'not yet implemented'", err.Error())
	}
}
