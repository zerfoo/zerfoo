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
	"strconv"
	"strings"

	"github.com/zerfoo/zerfoo/inference/sentiment"
)

// SentimentCommand implements the "sentiment" CLI command for running
// sentiment classification from the command line.
type SentimentCommand struct {
	out io.Writer
}

// NewSentimentCommand creates a new SentimentCommand.
func NewSentimentCommand(out io.Writer) *SentimentCommand {
	if out == nil {
		out = os.Stdout
	}
	return &SentimentCommand{out: out}
}

// sentimentConfig holds parsed sentiment command flags.
type sentimentConfig struct {
	modelPath  string
	text       string
	filePath   string
	batchSize  int
	format     string
	device     string
	continuous bool
}

// Name implements Command.Name.
func (c *SentimentCommand) Name() string { return "sentiment" }

// Description implements Command.Description.
func (c *SentimentCommand) Description() string {
	return "Run sentiment classification on text"
}

// Run implements Command.Run.
func (c *SentimentCommand) Run(ctx context.Context, args []string) error {
	cfg, err := c.parseArgs(args)
	if err != nil {
		return err
	}

	// Build pipeline options.
	opts := []sentiment.Option{
		sentiment.WithBatchSize(cfg.batchSize),
		sentiment.WithDevice(cfg.device),
	}
	if cfg.continuous {
		opts = append(opts, sentiment.WithContinuous())
	}

	// Auto-detect vocab.txt alongside the model for WordPiece tokenization.
	modelDir := filepath.Dir(cfg.modelPath)
	vocabFile := filepath.Join(modelDir, "vocab.txt")
	if _, err := os.Stat(vocabFile); err == nil {
		opts = append(opts, sentiment.WithVocabFile(vocabFile))
	}

	pipe, err := sentiment.New(cfg.modelPath, opts...)
	if err != nil {
		return fmt.Errorf("create sentiment pipeline: %w", err)
	}
	defer pipe.Close() //nolint:errcheck

	// Collect texts to classify.
	var texts []string
	if cfg.text != "" {
		texts = []string{cfg.text}
	} else {
		texts, err = c.readTexts(cfg.filePath)
		if err != nil {
			return fmt.Errorf("read input texts: %w", err)
		}
	}

	if len(texts) == 0 {
		return fmt.Errorf("no texts to classify")
	}

	results, err := pipe.Classify(ctx, texts)
	if err != nil {
		return fmt.Errorf("classify: %w", err)
	}

	return c.writeResults(cfg.format, texts, results)
}

// Usage implements Command.Usage.
func (c *SentimentCommand) Usage() string {
	return `sentiment [OPTIONS]

Run sentiment classification on text.

OPTIONS:
  --model <path>       Path to GGUF sentiment model (required)
  --text <string>      Single text to classify
  --file <path>        Path to file with texts (one per line, or CSV/JSONL); use "-" for stdin
  --batch-size <int>   Batch size for processing (default: 64)
  --format <format>    Output format: text, json, csv (default: text)
  --device <device>    Compute device: cpu, cuda (default: cpu)
  --continuous         Continuous scoring mode (regression)`
}

// Examples implements Command.Examples.
func (c *SentimentCommand) Examples() []string {
	return []string{
		`sentiment --model finbert.gguf --text "Stock prices surged after earnings beat"`,
		`sentiment --model finbert.gguf --file headlines.csv --format json`,
		`sentiment --model finbert.gguf --file - --format csv`,
	}
}

func (c *SentimentCommand) parseArgs(args []string) (*sentimentConfig, error) {
	cfg := &sentimentConfig{
		batchSize: 64,
		format:    "text",
		device:    "cpu",
	}

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
		case "--model":
			v, err := nextVal("--model")
			if err != nil {
				return nil, err
			}
			cfg.modelPath = v
		case "--text":
			v, err := nextVal("--text")
			if err != nil {
				return nil, err
			}
			cfg.text = v
		case "--file":
			v, err := nextVal("--file")
			if err != nil {
				return nil, err
			}
			cfg.filePath = v
		case "--batch-size":
			v, err := nextVal("--batch-size")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("--batch-size must be >= 1")
			}
			cfg.batchSize = n
		case "--format":
			v, err := nextVal("--format")
			if err != nil {
				return nil, err
			}
			switch v {
			case "text", "json", "csv":
				cfg.format = v
			default:
				return nil, fmt.Errorf("--format must be text, json, or csv")
			}
		case "--device":
			v, err := nextVal("--device")
			if err != nil {
				return nil, err
			}
			cfg.device = v
		case "--continuous":
			cfg.continuous = true
		default:
			return nil, fmt.Errorf("unknown flag: %s", arg)
		}
	}

	if cfg.modelPath == "" {
		return nil, fmt.Errorf("--model is required")
	}
	if cfg.text == "" && cfg.filePath == "" {
		return nil, fmt.Errorf("either --text or --file is required")
	}
	if cfg.text != "" && cfg.filePath != "" {
		return nil, fmt.Errorf("--text and --file are mutually exclusive")
	}

	return cfg, nil
}

// readTexts reads lines from a file path. If path is "-", reads from stdin.
func (c *SentimentCommand) readTexts(path string) ([]string, error) {
	var r io.Reader
	if path == "-" {
		r = os.Stdin
	} else {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close() //nolint:errcheck
		r = f
	}

	var texts []string
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			texts = append(texts, line)
		}
	}
	return texts, scanner.Err()
}

// writeResults formats and writes classification results to stdout.
func (c *SentimentCommand) writeResults(format string, texts []string, results []sentiment.SentimentResult) error {
	switch format {
	case "json":
		return c.writeJSON(texts, results)
	case "csv":
		return c.writeCSV(texts, results)
	default:
		return c.writeText(texts, results)
	}
}

func (c *SentimentCommand) writeText(texts []string, results []sentiment.SentimentResult) error {
	for i, r := range results {
		text := texts[i]
		fmt.Fprintf(c.out, "%s (%.2f)  %q\n", r.Label, r.Score, text)
	}
	return nil
}

func (c *SentimentCommand) writeJSON(texts []string, results []sentiment.SentimentResult) error {
	type jsonRow struct {
		Text  string  `json:"text"`
		Label string  `json:"label"`
		Score float64 `json:"score"`
	}
	rows := make([]jsonRow, len(results))
	for i, r := range results {
		rows[i] = jsonRow{Text: texts[i], Label: r.Label, Score: r.Score}
	}
	enc := json.NewEncoder(c.out)
	enc.SetIndent("", "  ")
	return enc.Encode(rows)
}

func (c *SentimentCommand) writeCSV(texts []string, results []sentiment.SentimentResult) error {
	w := csv.NewWriter(c.out)
	defer w.Flush()
	if err := w.Write([]string{"text", "label", "score"}); err != nil {
		return err
	}
	for i, r := range results {
		if err := w.Write([]string{
			texts[i],
			r.Label,
			strconv.FormatFloat(r.Score, 'f', 4, 64),
		}); err != nil {
			return err
		}
	}
	return nil
}

// Static interface assertion.
var _ Command = (*SentimentCommand)(nil)
