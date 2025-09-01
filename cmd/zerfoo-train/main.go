package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Printf("DEPRECATED: zerfoo-train has been moved to audacity.\n")
	fmt.Printf("\nAs part of architectural refactoring, all Numerai-specific functionality\n")
	fmt.Printf("has been moved from the generic zerfoo framework to the audacity application.\n\n")
	fmt.Printf("Please use the training tools in the audacity project instead:\n")
	fmt.Printf("  cd ../audacity/\n")
	fmt.Printf("  go run cmd/numerai-train/main.go [args...]\n\n")
	fmt.Printf("This change ensures zerfoo remains a generic ML framework while\n")
	fmt.Printf("audacity provides domain-specific Numerai tournament functionality.\n\n")
	fmt.Printf("See docs/migration_plan.md for full migration details.\n")
	os.Exit(1)
}