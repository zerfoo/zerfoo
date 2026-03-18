package site

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewSite(t *testing.T) {
	site, err := NewSite("testdata/src", "testdata/out")
	if err != nil {
		t.Fatalf("NewSite: %v", err)
	}
	if site.SrcDir != "testdata/src" {
		t.Errorf("SrcDir = %q, want %q", site.SrcDir, "testdata/src")
	}
	if site.OutDir != "testdata/out" {
		t.Errorf("OutDir = %q, want %q", site.OutDir, "testdata/out")
	}
}

func TestBuild(t *testing.T) {
	srcDir := t.TempDir()
	outDir := t.TempDir()

	writeFile(t, filepath.Join(srcDir, "getting-started.md"), "# Getting Started\n\nWelcome to Zerfoo.\n")
	writeFile(t, filepath.Join(srcDir, "benchmarks.md"), "# Benchmarks\n\nPerformance results.\n")

	os.MkdirAll(filepath.Join(srcDir, "adr"), 0o755)
	writeFile(t, filepath.Join(srcDir, "adr", "001-model-format.md"), "# ADR 001\n\nUse GGUF.\n")

	site, err := NewSite(srcDir, outDir)
	if err != nil {
		t.Fatalf("NewSite: %v", err)
	}

	if err := site.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}

	if len(site.Pages) != 3 {
		t.Errorf("got %d pages, want 3", len(site.Pages))
	}

	for _, want := range []string{
		"getting-started.html",
		"benchmarks.html",
		"adr/001-model-format.html",
	} {
		path := filepath.Join(outDir, want)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("expected file %s to exist", want)
		}
	}

	data, err := os.ReadFile(filepath.Join(outDir, "getting-started.html"))
	if err != nil {
		t.Fatalf("reading output: %v", err)
	}
	html := string(data)
	if !strings.Contains(html, "Welcome to Zerfoo") {
		t.Error("output missing content 'Welcome to Zerfoo'")
	}
	if !strings.Contains(html, "<title>Getting Started — Zerfoo Docs</title>") {
		t.Error("output missing title tag")
	}
	if !strings.Contains(html, "style.css") {
		t.Error("output missing stylesheet link")
	}
	if !strings.Contains(html, "search.js") {
		t.Error("output missing search script")
	}

	idxData, err := os.ReadFile(filepath.Join(outDir, "search-index.json"))
	if err != nil {
		t.Fatalf("reading search index: %v", err)
	}
	var entries []SearchEntry
	if err := json.Unmarshal(idxData, &entries); err != nil {
		t.Fatalf("parsing search index: %v", err)
	}
	if len(entries) != 3 {
		t.Errorf("search index has %d entries, want 3", len(entries))
	}

	if _, err := os.Stat(filepath.Join(outDir, "static", "style.css")); os.IsNotExist(err) {
		t.Error("static/style.css not copied")
	}
	if _, err := os.Stat(filepath.Join(outDir, "static", "search.js")); os.IsNotExist(err) {
		t.Error("static/search.js not copied")
	}
}

func TestBuildWithCodeBlock(t *testing.T) {
	srcDir := t.TempDir()
	outDir := t.TempDir()

	writeFile(t, filepath.Join(srcDir, "example.md"), "# Example\n\n```go\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n```\n")

	site, err := NewSite(srcDir, outDir)
	if err != nil {
		t.Fatalf("NewSite: %v", err)
	}

	if err := site.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(outDir, "example.html"))
	if err != nil {
		t.Fatalf("reading output: %v", err)
	}

	html := string(data)
	if !strings.Contains(html, "chroma") && !strings.Contains(html, "style=") {
		t.Error("expected syntax-highlighted code with chroma classes or inline styles")
	}
}

func TestNavigation(t *testing.T) {
	srcDir := t.TempDir()
	outDir := t.TempDir()

	writeFile(t, filepath.Join(srcDir, "intro.md"), "# Intro\n")
	os.MkdirAll(filepath.Join(srcDir, "guides"), 0o755)
	writeFile(t, filepath.Join(srcDir, "guides", "setup.md"), "# Setup\n")
	writeFile(t, filepath.Join(srcDir, "guides", "deploy.md"), "# Deploy\n")

	site, err := NewSite(srcDir, outDir)
	if err != nil {
		t.Fatalf("NewSite: %v", err)
	}

	if err := site.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}

	if len(site.Nav) == 0 {
		t.Fatal("navigation is empty")
	}

	foundIntro := false
	foundGuides := false
	for _, item := range site.Nav {
		if item.Title == "Intro" {
			foundIntro = true
		}
		if item.Section == "guides" {
			foundGuides = true
			if len(item.Children) != 2 {
				t.Errorf("guides section has %d children, want 2", len(item.Children))
			}
		}
	}
	if !foundIntro {
		t.Error("navigation missing 'Intro' root item")
	}
	if !foundGuides {
		t.Error("navigation missing 'guides' section")
	}

	data, err := os.ReadFile(filepath.Join(outDir, "intro.html"))
	if err != nil {
		t.Fatalf("reading output: %v", err)
	}
	html := string(data)
	if !strings.Contains(html, "/guides/setup.html") {
		t.Error("navigation missing link to guides/setup")
	}
}

func TestBuildEmptyDir(t *testing.T) {
	srcDir := t.TempDir()
	outDir := t.TempDir()

	site, err := NewSite(srcDir, outDir)
	if err != nil {
		t.Fatalf("NewSite: %v", err)
	}

	if err := site.Build(); err != nil {
		t.Fatalf("Build on empty dir: %v", err)
	}

	if len(site.Pages) != 0 {
		t.Errorf("got %d pages, want 0", len(site.Pages))
	}
}

func TestTitleFromPath(t *testing.T) {
	tests := []struct {
		path string
		want string
	}{
		{"getting-started.md", "Getting Started"},
		{"benchmarks.md", "Benchmarks"},
		{"adr/001-model-format.md", "001 Model Format"},
		{"gpu-setup.md", "Gpu Setup"},
		{"QUALITY.md", "QUALITY"},
	}

	for _, tt := range tests {
		got := titleFromPath(tt.path)
		if got != tt.want {
			t.Errorf("titleFromPath(%q) = %q, want %q", tt.path, got, tt.want)
		}
	}
}

func TestStripHTML(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"<p>hello</p>", " hello "},
		{"<a href='x'>link</a>", " link "},
		{"no tags", "no tags"},
		{"<h1>Title</h1><p>Body</p>", " Title  Body "},
	}

	for _, tt := range tests {
		got := stripHTML(tt.input)
		if got != tt.want {
			t.Errorf("stripHTML(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
}
