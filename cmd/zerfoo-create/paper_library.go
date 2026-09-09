package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"sort"
	"strings"
)

// searchPaperLibrary reads the corpus as data without running its Python code
// or loading its vector extension. Directory contents cannot assert support.
func searchPaperLibrary(ctx context.Context, path, query string) (cards []evidenceCard, retErr error) {
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, err
	}
	defer func() { retErr = errors.Join(retErr, root.Close()) }()
	dir, err := root.Open("papers")
	if err != nil {
		return nil, fmt.Errorf("paper-library papers directory: %w", err)
	}
	entries, readErr := dir.ReadDir(10002)
	if errors.Is(readErr, io.EOF) {
		readErr = nil
	}
	if err := errors.Join(readErr, dir.Close()); err != nil {
		return nil, err
	}
	if len(entries) > 10001 {
		return nil, fmt.Errorf("paper-library exceeds 10000 records")
	}
	type hit struct {
		card  evidenceCard
		score int
	}
	hits := []hit{}
	terms := strings.Fields(strings.ToLower(query))
	total := 0
	seen := map[string]bool{}
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if entry.Name() == "index.json" || strings.HasPrefix(entry.Name(), ".") || strings.HasPrefix(entry.Name(), "_") || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		if !entry.Type().IsRegular() {
			return nil, fmt.Errorf("paper-library record must be a regular file: %s", entry.Name())
		}
		f, err := root.Open("papers/" + entry.Name())
		if err != nil {
			return nil, err
		}
		raw, readErr := io.ReadAll(io.LimitReader(f, (1<<20)+1))
		if err := errors.Join(readErr, f.Close()); err != nil {
			return nil, err
		}
		total += len(raw)
		if len(raw) > 1<<20 || total > 32<<20 {
			return nil, fmt.Errorf("paper-library exceeds record or corpus byte limit")
		}
		// This external schema may gain fields; decode only the documented metadata.
		var paper struct {
			ID       string   `json:"id"`
			Title    string   `json:"title"`
			Abstract string   `json:"abstract"`
			URL      string   `json:"arxiv_url"`
			Tags     []string `json:"tags"`
		}
		if err := json.Unmarshal(raw, &paper); err != nil {
			return nil, fmt.Errorf("paper %s: %w", entry.Name(), err)
		}
		u, err := url.Parse(paper.URL)
		if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || paper.ID == "" || paper.Title == "" || paper.Abstract == "" || seen[paper.ID] {
			return nil, fmt.Errorf("invalid or duplicate paper %q", paper.ID)
		}
		seen[paper.ID] = true
		title := strings.ToLower(paper.Title)
		body := strings.ToLower(paper.Abstract + " " + strings.Join(paper.Tags, " "))
		score := 0
		matched := true
		for _, term := range terms {
			if strings.Contains(title, term) {
				score += 3
			} else if strings.Contains(body, term) {
				score++
			} else {
				matched = false
				break
			}
		}
		if !matched {
			continue
		}
		hash := sha256.Sum256(raw)
		hits = append(hits, hit{card: evidenceCard{
			ID: "arxiv:" + paper.ID, Title: paper.Title, URL: paper.URL,
			Version: "sha256:" + hex.EncodeToString(hash[:]), Section: "abstract",
			Summary: paper.Abstract, Components: []string{}, Eligible: false,
			SupportGap: "Paper metadata has no reviewed Zerfoo component mapping; not eligible as executable plan evidence.",
		}, score: score})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].card.ID < hits[j].card.ID
	})
	if len(hits) > 20 {
		hits = hits[:20]
	}
	result := make([]evidenceCard, 0, len(hits))
	for _, h := range hits {
		result = append(result, h.card)
	}
	return result, nil
}
