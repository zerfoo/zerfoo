// Package main implements a linter that checks // Deprecated: doc comments
// for proper replacement guidance and version information.
//
// Usage:
//
//	go run ./cmd/deprecation-check/...
//	go run ./cmd/deprecation-check/... ./specific/package/...
package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Violation records a deprecation comment that is missing required information.
type Violation struct {
	File    string
	Line    int
	Name    string
	Comment string
	Missing []string // e.g. ["replacement guidance", "version"]
}

var versionRe = regexp.MustCompile(`v\d+\.\d+`)

func main() {
	roots := os.Args[1:]
	if len(roots) == 0 {
		roots = []string{"."}
	}

	var violations []Violation
	for _, root := range roots {
		v, err := checkDir(root)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(2)
		}
		violations = append(violations, v...)
	}

	for _, v := range violations {
		fmt.Printf("%s:%d: %s: Deprecated comment missing %s\n",
			v.File, v.Line, v.Name, strings.Join(v.Missing, " and "))
		fmt.Printf("  comment: %s\n", v.Comment)
		fmt.Println()
	}

	if len(violations) > 0 {
		fmt.Printf("Found %d deprecation comment(s) with missing information.\n", len(violations))
		os.Exit(1)
	}
}

func checkDir(root string) ([]Violation, error) {
	var violations []Violation

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			base := info.Name()
			if base == "vendor" || base == "testdata" || base == ".git" || base == ".claude" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}

		v, err := checkFile(path)
		if err != nil {
			return fmt.Errorf("parsing %s: %w", path, err)
		}
		violations = append(violations, v...)
		return nil
	})

	return violations, err
}

func checkFile(path string) ([]Violation, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	var violations []Violation

	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if v, ok := checkDoc(fset, path, d.Name.Name, d.Doc); ok {
				violations = append(violations, v)
			}
		case *ast.GenDecl:
			// Check the GenDecl-level doc (applies when there's a single spec).
			if d.Doc != nil && len(d.Specs) == 1 {
				if name := specName(d.Specs[0]); name != "" {
					if v, ok := checkDoc(fset, path, name, d.Doc); ok {
						violations = append(violations, v)
					}
				}
			}
			// Check individual spec docs.
			for _, spec := range d.Specs {
				name := specName(spec)
				if name == "" {
					continue
				}
				var doc *ast.CommentGroup
				switch s := spec.(type) {
				case *ast.TypeSpec:
					doc = s.Doc
				case *ast.ValueSpec:
					doc = s.Doc
				}
				if v, ok := checkDoc(fset, path, name, doc); ok {
					violations = append(violations, v)
				}
			}
		}
	}

	return violations, nil
}

func specName(spec ast.Spec) string {
	switch s := spec.(type) {
	case *ast.TypeSpec:
		return s.Name.Name
	case *ast.ValueSpec:
		if len(s.Names) > 0 {
			return s.Names[0].Name
		}
	}
	return ""
}

func checkDoc(fset *token.FileSet, path, name string, doc *ast.CommentGroup) (Violation, bool) {
	if doc == nil {
		return Violation{}, false
	}

	text := doc.Text()
	depIdx := strings.Index(text, "Deprecated:")
	if depIdx == -1 {
		return Violation{}, false
	}

	depText := text[depIdx:]

	var missing []string
	if !hasReplacementGuidance(depText) {
		missing = append(missing, "replacement guidance")
	}
	if !hasVersion(depText) {
		missing = append(missing, "version")
	}

	if len(missing) == 0 {
		return Violation{}, false
	}

	line := fset.Position(doc.Pos()).Line
	return Violation{
		File:    path,
		Line:    line,
		Name:    name,
		Comment: strings.TrimSpace(strings.SplitN(depText, "\n", 2)[0]),
		Missing: missing,
	}, true
}

// hasReplacementGuidance checks for common replacement patterns like
// "Use X instead", "See X", or "Replace with X".
func hasReplacementGuidance(text string) bool {
	lower := strings.ToLower(text)
	patterns := []string{
		"use ", "see ", "replace", "switch to", "migrate to",
		"prefer ", "instead", "superseded by", "removed",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func hasVersion(text string) bool {
	return versionRe.MatchString(text)
}
