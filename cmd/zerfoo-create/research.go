package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/model"
)

// Evidence is data, never executable instructions. Only supported components
// are eligible; references are retained without claiming paper reproduction.
type evidenceCard struct {
	Eligible   bool     `json:"eligible"`
	SupportGap string   `json:"support_gap,omitempty"`
	ID         string   `json:"id"`
	Title      string   `json:"title"`
	URL        string   `json:"url"`
	Version    string   `json:"version"`
	Section    string   `json:"section"`
	Summary    string   `json:"summary"`
	Components []string `json:"components"`
}

func (s *service) search(ctx context.Context, query string) ([]evidenceCard, error) {
	result := []evidenceCard{}
	if s.library == "" {
		return result, nil
	}
	info, err := os.Stat(s.library)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		return searchPaperLibrary(ctx, s.library, query)
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
	supported := map[string]bool{}
	for _, descriptor := range model.ListComponents() {
		_, rule, ok := model.Component(descriptor.Kind, descriptor.ID)
		if ok && (rule != nil || descriptor.Kind == "loss" || descriptor.Kind == "optimizer") {
			supported[strings.ToLower(descriptor.ID)] = true
		}
	}
	seen := map[string]bool{}
	for _, card := range catalog.Cards {
		u, err := url.Parse(card.URL)
		if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || card.ID == "" || seen[card.ID] || card.Version == "" || card.Section == "" || card.Title == "" {
			return nil, fmt.Errorf("invalid evidence card %q", card.ID)
		}
		seen[card.ID] = true
		eligible := len(card.Components) > 0
		for _, component := range card.Components {
			if !supported[strings.ToLower(component)] {
				eligible = false
			}
		}
		card.Eligible = eligible
		if eligible && strings.Contains(strings.ToLower(card.Title+" "+card.Summary), strings.ToLower(query)) {
			result = append(result, card)
		}
	}
	return result, nil
}
