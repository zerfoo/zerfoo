package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDistillationBindingAndEligibility(t *testing.T) {
	s := &service{distillations: t.TempDir()}
	if err := os.Mkdir(filepath.Join(s.distillations, "notes"), 0700); err != nil {
		t.Fatal(err)
	}
	note := map[string]any{"version": 1, "paper_id": "2410.15735", "record_sha256": "recordhash", "source_scope": "full_paper", "eligible": false, "review_status": "source_anchors_checked_semantic_review_required", "architecture": map[string]any{"blocks": []string{"fixture"}}}
	write := func() {
		t.Helper()
		raw, err := json.Marshal(note)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(s.distillations, "notes", "2410.15735.json"), raw, 0600); err != nil {
			t.Fatal(err)
		}
	}
	write()
	cards := []evidenceCard{{ID: "arxiv:2410.15735", Version: "sha256:recordhash"}}
	if err := s.attachDistillations(context.Background(), cards); err != nil {
		t.Fatal(err)
	}
	if len(cards[0].Distillation) == 0 || cards[0].Eligible {
		t.Fatal("guidance lost or falsely qualified")
	}
	note["eligible"] = true
	write()
	if err := s.attachDistillations(context.Background(), cards); err == nil {
		t.Fatal("external eligibility accepted")
	}
	note["eligible"] = false
	note["record_sha256"] = "stale"
	write()
	if err := s.attachDistillations(context.Background(), cards); err == nil {
		t.Fatal("stale source accepted")
	}
}
