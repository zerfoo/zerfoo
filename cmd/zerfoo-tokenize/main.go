package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/pkg/tokenizer"
)

func main() {
	// Check if we should use the new CLI framework
	if len(os.Args) > 1 && (os.Args[1] == "--new-cli" || os.Getenv("ZERFOO_USE_NEW_CLI") == "true") {
		fmt.Println("NEW CLI: Use 'zerfoo tokenize' for the new framework-based tokenization.")
		fmt.Println("This legacy tool will be deprecated in future versions.")
		os.Exit(0)
	}
	
	// Legacy behavior for backward compatibility
	text := flag.String("text", "", "Text to tokenize")
	flag.Parse()

	if *text == "" {
		fmt.Println("Please provide text to tokenize using the -text flag.")
		os.Exit(1)
	}

	t := tokenizer.NewTokenizer()
	// This is a simple tokenizer, so we will add the words from the input to the vocab
	for _, word := range strings.Fields(*text) {
		t.AddToken(word)
	}

	tokenIDs := t.Encode(*text)

	fmt.Printf("Token IDs for '%s': %v\n", *text, tokenIDs)
}