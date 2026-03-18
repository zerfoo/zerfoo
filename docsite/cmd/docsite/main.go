// Command docsite generates a static documentation site from markdown files
// and optionally serves it locally for preview.
//
// Usage:
//
//	go run ./cmd/docsite -src ../docs -out _site          # generate site
//	go run ./cmd/docsite -src ../docs -out _site -serve   # generate and serve at :8080
//
//go:generate go run . -src ../../../docs -out ../../_site
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/docsite/site"
)

func main() {
	src := flag.String("src", "../docs", "source directory containing markdown files")
	out := flag.String("out", "_site", "output directory for generated HTML")
	serve := flag.Bool("serve", false, "serve the site after generation")
	addr := flag.String("addr", ":8080", "address to serve on (with -serve)")
	flag.Parse()

	absSrc, err := filepath.Abs(*src)
	if err != nil {
		log.Fatalf("resolving source path: %v", err)
	}

	absOut, err := filepath.Abs(*out)
	if err != nil {
		log.Fatalf("resolving output path: %v", err)
	}

	s, err := site.NewSite(absSrc, absOut)
	if err != nil {
		log.Fatalf("initializing site: %v", err)
	}

	if err := s.Build(); err != nil {
		log.Fatalf("building site: %v", err)
	}

	pageCount := len(s.Pages)
	fmt.Fprintf(os.Stderr, "docsite: generated %d pages in %s\n", pageCount, absOut)

	if *serve {
		fmt.Fprintf(os.Stderr, "docsite: serving at http://localhost%s\n", *addr)
		handler := http.FileServer(http.Dir(absOut))
		if err := http.ListenAndServe(*addr, handler); err != nil {
			log.Fatalf("server: %v", err)
		}
	}
}
