package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPaperLibraryCandidatesAndProvenance(t *testing.T) {
	s, p := setupPlan(t)
	s.library = t.TempDir()
	dir := filepath.Join(s.library, "papers")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".fetch_state.json"), []byte(`not a paper`), 0600); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "1234.56789.json")
	raw := []byte(`{"id":"1234.56789","title":"Residual classifiers","abstract":"Shared parameters for tabular data","arxiv_url":"https://arxiv.org/abs/1234.56789","tags":["classification"],"authors":["Fixture"],"components":["Dense"]}`)
	if err := os.WriteFile(file, raw, 0600); err != nil {
		t.Fatal(err)
	}
	cards := invoke(t, s, "research_search", map[string]any{"query": "residual tabular"}).([]evidenceCard)
	if len(cards) != 1 || cards[0].Eligible || len(cards[0].Components) != 0 || cards[0].SupportGap == "" || !strings.HasPrefix(cards[0].Version, "sha256:") {
		t.Fatalf("invalid candidate: %+v", cards)
	}
	args, err := json.Marshal(map[string]any{"project": p.Project, "dataset": p.Dataset, "rationale": "candidate is not verified support", "evidence": []string{cards[0].ID}, "epochs": 1, "batch_size": 15, "learning_rate": .01})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.call(context.Background(), "plan_create", args); err == nil || !strings.Contains(err.Error(), "unqualified evidence") {
		t.Fatalf("candidate accepted as plan evidence: %v", err)
	}
	if err := os.WriteFile(file, append(raw, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	changed, err := s.search(context.Background(), "residual")
	if err != nil {
		t.Fatal(err)
	}
	if changed[0].Version == cards[0].Version {
		t.Fatal("source change did not change identity")
	}
	if err := os.Symlink(file, filepath.Join(dir, "alias.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.search(context.Background(), ""); err == nil {
		t.Fatal("symlink record accepted")
	}
}
