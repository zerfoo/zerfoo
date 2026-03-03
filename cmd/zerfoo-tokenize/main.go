package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/pkg/tokenizer"
)

func main() {
	text := flag.String("text", "", "Text to tokenize")
	vocabPath := flag.String("vocab", "", "Path to vocabulary file (one token per line)")
	flag.Parse()

	if *text == "" {
		fmt.Println("Please provide text to tokenize using the -text flag.")
		os.Exit(1)
	}

	t := tokenizer.NewWhitespaceTokenizer()

	// Load vocabulary from file if provided
	if *vocabPath != "" {
		if err := loadVocab(t, *vocabPath); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to load vocabulary: %v\n", err)
			os.Exit(1)
		}
	}

	tokenIDs, err := t.Encode(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Tokenization failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Token IDs for '%s': %v\n", *text, tokenIDs)
}

func loadVocab(t *tokenizer.WhitespaceTokenizer, path string) error {
	file, err := os.Open(path) //nolint:gosec // user-provided path
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
