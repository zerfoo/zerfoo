package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	tokenizer "github.com/zerfoo/ztoken"
)

func main() {
	text := flag.String("text", "", "Text to tokenize")
	vocabPath := flag.String("vocab", "", "Path to vocabulary file (one token per line)")
	flag.Parse()

	if err := run(*text, *vocabPath, os.Stdout); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(text, vocabPath string, w io.Writer) error {
	if text == "" {
		return fmt.Errorf("error: -text flag is required")
	}

	t := tokenizer.NewWhitespaceTokenizer()

	if vocabPath != "" {
		if err := loadVocab(t, vocabPath); err != nil {
			return fmt.Errorf("failed to load vocabulary: %w", err)
		}
	}

	tokenIDs, err := t.Encode(text)
	if err != nil {
		return fmt.Errorf("tokenization failed: %w", err)
	}

	_, _ = fmt.Fprintf(w, "Token IDs for '%s': %v\n", text, tokenIDs)
	return nil
}

func loadVocab(t *tokenizer.WhitespaceTokenizer, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close() //nolint:errcheck

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		token := strings.TrimSpace(scanner.Text())
		if token != "" {
			t.AddToken(token)
		}
	}
	return scanner.Err()
}
