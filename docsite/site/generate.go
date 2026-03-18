// Package site is a static site generator for Zerfoo documentation.
// It converts markdown files from docs/ into a navigable HTML site.
package site

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yuin/goldmark"
	highlighting "github.com/yuin/goldmark-highlighting/v2"
)

//go:embed template/*.html static/*
var embedded embed.FS

// Page represents a single documentation page.
type Page struct {
	Title    string
	Path     string // URL path (e.g., "/getting-started")
	Content  template.HTML
	Section  string // top-level directory or "" for root
	SrcPath  string // relative source path
	Children []*Page
}

// NavItem represents an entry in the navigation sidebar.
type NavItem struct {
	Title    string
	Path     string
	Active   bool
	Section  string
	Children []NavItem
}

// SearchEntry is indexed for client-side search.
type SearchEntry struct {
	Title string `json:"title"`
	Path  string `json:"path"`
	Body  string `json:"body"`
}

// Site holds the complete generated site state.
type Site struct {
	Pages     []*Page
	Nav       []NavItem
	SrcDir    string
	OutDir    string
	md        goldmark.Markdown
	tmpl      *template.Template
	searchIdx []SearchEntry
}

// NewSite creates a site generator that reads from srcDir and writes to outDir.
func NewSite(srcDir, outDir string) (*Site, error) {
	md := goldmark.New(
		goldmark.WithExtensions(
			highlighting.NewHighlighting(
				highlighting.WithStyle("monokai"),
				highlighting.WithFormatOptions(),
			),
		),
	)

	funcMap := template.FuncMap{
		"eq": func(a, b string) bool { return a == b },
	}

	tmpl, err := template.New("").Funcs(funcMap).ParseFS(embedded, "template/*.html")
	if err != nil {
		return nil, fmt.Errorf("parsing templates: %w", err)
	}

	return &Site{
		SrcDir: srcDir,
		OutDir: outDir,
		md:     md,
		tmpl:   tmpl,
	}, nil
}

// Build generates the complete static site.
func (s *Site) Build() error {
	if err := s.collectPages(); err != nil {
		return fmt.Errorf("collecting pages: %w", err)
	}

	s.buildNav()

	if err := os.MkdirAll(s.OutDir, 0o755); err != nil {
		return err
	}

	for _, p := range s.Pages {
		if err := s.renderPage(p); err != nil {
			return fmt.Errorf("rendering %s: %w", p.SrcPath, err)
		}
	}

	if err := s.writeSearchIndex(); err != nil {
		return fmt.Errorf("writing search index: %w", err)
	}

	if err := s.copyStatic(); err != nil {
		return fmt.Errorf("copying static files: %w", err)
	}

	return nil
}

func (s *Site) collectPages() error {
	return filepath.WalkDir(s.SrcDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if filepath.Ext(path) != ".md" {
			return nil
		}

		rel, err := filepath.Rel(s.SrcDir, path)
		if err != nil {
			return err
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		var buf bytes.Buffer
		if err := s.md.Convert(data, &buf); err != nil {
			return fmt.Errorf("converting %s: %w", rel, err)
		}

		urlPath := "/" + strings.TrimSuffix(rel, ".md")
		if strings.HasSuffix(urlPath, "/index") {
			urlPath = strings.TrimSuffix(urlPath, "/index")
		}
		if urlPath == "" {
			urlPath = "/"
		}

		section := ""
		if parts := strings.SplitN(rel, string(filepath.Separator), 2); len(parts) > 1 {
			section = parts[0]
		}

		page := &Page{
			Title:   titleFromPath(rel),
			Path:    urlPath,
			Content: template.HTML(buf.String()),
			Section: section,
			SrcPath: rel,
		}

		s.Pages = append(s.Pages, page)
		s.searchIdx = append(s.searchIdx, SearchEntry{
			Title: page.Title,
			Path:  page.Path + ".html",
			Body:  stripHTML(buf.String()),
		})

		return nil
	})
}

func (s *Site) buildNav() {
	sections := map[string][]NavItem{}
	var rootItems []NavItem

	sort.Slice(s.Pages, func(i, j int) bool {
		return s.Pages[i].Path < s.Pages[j].Path
	})

	for _, p := range s.Pages {
		item := NavItem{
			Title:   p.Title,
			Path:    p.Path + ".html",
			Section: p.Section,
		}
		if p.Section == "" {
			rootItems = append(rootItems, item)
		} else {
			sections[p.Section] = append(sections[p.Section], item)
		}
	}

	s.Nav = rootItems

	sectionNames := make([]string, 0, len(sections))
	for name := range sections {
		sectionNames = append(sectionNames, name)
	}
	sort.Strings(sectionNames)

	for _, name := range sectionNames {
		s.Nav = append(s.Nav, NavItem{
			Title:    titleCase(name),
			Path:     "#",
			Section:  name,
			Children: sections[name],
		})
	}
}

func (s *Site) renderPage(p *Page) error {
	nav := make([]NavItem, len(s.Nav))
	for i, item := range s.Nav {
		nav[i] = item
		nav[i].Active = item.Path == p.Path+".html"
		children := make([]NavItem, len(item.Children))
		for j, child := range item.Children {
			children[j] = child
			children[j].Active = child.Path == p.Path+".html"
		}
		nav[i].Children = children
	}

	data := struct {
		Page *Page
		Nav  []NavItem
	}{
		Page: p,
		Nav:  nav,
	}

	outPath := filepath.Join(s.OutDir, p.Path+".html")
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}

	var buf bytes.Buffer
	if err := s.tmpl.ExecuteTemplate(&buf, "page.html", data); err != nil {
		return err
	}

	return os.WriteFile(outPath, buf.Bytes(), 0o644)
}

func (s *Site) writeSearchIndex() error {
	data, err := json.Marshal(s.searchIdx)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(s.OutDir, "search-index.json"), data, 0o644)
}

func (s *Site) copyStatic() error {
	staticDir := filepath.Join(s.OutDir, "static")
	if err := os.MkdirAll(staticDir, 0o755); err != nil {
		return err
	}

	entries, err := fs.ReadDir(embedded, "static")
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		data, err := fs.ReadFile(embedded, "static/"+entry.Name())
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(staticDir, entry.Name()), data, 0o644); err != nil {
			return err
		}
	}

	return nil
}

func titleFromPath(path string) string {
	base := strings.TrimSuffix(filepath.Base(path), ".md")
	return titleCase(base)
}

func titleCase(s string) string {
	s = strings.ReplaceAll(s, "-", " ")
	s = strings.ReplaceAll(s, "_", " ")
	if len(s) == 0 {
		return s
	}
	words := strings.Fields(s)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, " ")
}

func stripHTML(s string) string {
	var buf strings.Builder
	inTag := false
	for _, r := range s {
		switch {
		case r == '<':
			inTag = true
		case r == '>':
			inTag = false
			buf.WriteRune(' ')
		case !inTag:
			buf.WriteRune(r)
		}
	}
	return buf.String()
}
