// Command gguf-info prints tensor names, shapes, and GGML types from a GGUF file.
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/zerfoo/zerfoo/model/gguf"
)

// ggmlTypeName maps GGMLType constants to human-readable names.
var ggmlTypeName = map[gguf.GGMLType]string{
	gguf.GGMLTypeF32:     "F32",
	gguf.GGMLTypeF16:     "F16",
	gguf.GGMLTypeQ4_0:    "Q4_0",
	gguf.GGMLTypeQ4_1:    "Q4_1",
	gguf.GGMLTypeQ5_0:    "Q5_0",
	gguf.GGMLTypeQ5_1:    "Q5_1",
	gguf.GGMLTypeQ8_0:    "Q8_0",
	gguf.GGMLTypeQ8_1:    "Q8_1",
	gguf.GGMLTypeQ2_K:    "Q2_K",
	gguf.GGMLTypeQ3_K:    "Q3_K",
	gguf.GGMLTypeQ4_K:    "Q4_K",
	gguf.GGMLTypeQ5_K:    "Q5_K",
	gguf.GGMLTypeQ6_K:    "Q6_K",
	gguf.GGMLTypeQ8_K:    "Q8_K",
	gguf.GGMLTypeIQ2_XXS: "IQ2_XXS",
	gguf.GGMLTypeIQ3_S:   "IQ3_S",
	gguf.GGMLTypeIQ4_NL:  "IQ4_NL",
	gguf.GGMLTypeBF16:    "BF16",
	gguf.GGMLTypeTQ2_0:   "TQ2_0",
}

func typeName(t gguf.GGMLType) string {
	if name, ok := ggmlTypeName[t]; ok {
		return name
	}
	return fmt.Sprintf("type_%d", t)
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "usage: gguf-info <file.gguf>\n")
		os.Exit(1)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	gf, err := gguf.Parse(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse: %v\n", err)
		os.Exit(1)
	}

	// Print each tensor.
	counts := make(map[gguf.GGMLType]int)
	for _, ti := range gf.Tensors {
		// Convert GGML dimensions (innermost-first) to conventional shape (outermost-first).
		dims := make([]string, len(ti.Dimensions))
		for j, d := range ti.Dimensions {
			dims[len(ti.Dimensions)-1-j] = fmt.Sprintf("%d", d)
		}
		fmt.Printf("%-60s [%s]  %s\n", ti.Name, strings.Join(dims, ", "), typeName(ti.Type))
		counts[ti.Type]++
	}

	// Print type distribution summary.
	type entry struct {
		name  string
		typ   gguf.GGMLType
		count int
	}
	entries := make([]entry, 0, len(counts))
	for t, c := range counts {
		entries = append(entries, entry{typeName(t), t, c})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].count > entries[j].count
	})

	fmt.Printf("\n%d tensors total\n", len(gf.Tensors))
	for _, e := range entries {
		noun := "tensors"
		if e.count == 1 {
			noun = "tensor"
		}
		fmt.Printf("  %s: %d %s\n", e.name, e.count, noun)
	}
}
