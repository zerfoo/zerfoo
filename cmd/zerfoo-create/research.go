package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"strings"
)

// Evidence is data, never executable instructions. Only supported components
// are eligible; references are retained without claiming paper reproduction.
type evidenceCard struct {
	ID         string   `json:"id"`
	Title      string   `json:"title"`
	URL        string   `json:"url"`
	Version    string   `json:"version"`
	Section    string   `json:"section"`
	Summary    string   `json:"summary"`
	Components []string `json:"components"`
}

func (s *service) search(query string) ([]evidenceCard, error) {
	result := []evidenceCard{}
	if s.library == "" {
		return result, nil
	}
	f, err := os.Open(s.library)
	if err != nil {
		return nil, err
	}
	raw, readErr := io.ReadAll(io.LimitReader(f, (8<<20)+1))
	if err := errors.Join(readErr, f.Close()); err != nil {
		return nil, err
	}
	if len(raw) > 8<<20 {
		return nil, fmt.Errorf("research catalog exceeds 8 MiB")
	}
	var catalog struct {
		Version int            `json:"version"`
		Cards   []evidenceCard `json:"cards"`
	}
	if err := strictJSON(json.RawMessage(raw), &catalog); err != nil {
		return nil, err
	}
	if catalog.Version != 1 {
		return nil, fmt.Errorf("unsupported evidence schema")
	}
	supported := map[string]bool{"linear": true, "relu": true, "softmax": true, "cross_entropy": true, "adamw": true}
	seen := map[string]bool{}
	for _, card := range catalog.Cards {
		u, err := url.Parse(card.URL)
		if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || card.ID == "" || seen[card.ID] || card.Version == "" || card.Section == "" || card.Title == "" {
			return nil, fmt.Errorf("invalid evidence card %q", card.ID)
		}
		seen[card.ID] = true
		eligible := len(card.Components) > 0
		for _, component := range card.Components {
			if !supported[component] {
				eligible = false
			}
		}
		if eligible && strings.Contains(strings.ToLower(card.Title+" "+card.Summary), strings.ToLower(query)) {
			result = append(result, card)
		}
	}
	return result, nil
}
