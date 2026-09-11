package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

var paperIDPattern = regexp.MustCompile(`^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$`)

// Full-paper notes enrich retrieval, but cannot grant executable eligibility.
func (s *service) attachDistillations(ctx context.Context, cards []evidenceCard) (retErr error) {
	if s.distillations == "" {
		return nil
	}
	root, err := os.OpenRoot(s.distillations)
	if err != nil {
		return err
	}
	defer func() { retErr = errors.Join(retErr, root.Close()) }()
	for i := range cards {
		if err := ctx.Err(); err != nil {
			return err
		}
		card := &cards[i]
		id := strings.TrimPrefix(card.ID, "arxiv:")
		if !paperIDPattern.MatchString(id) {
			return fmt.Errorf("invalid distillation paper ID")
		}
		f, err := root.Open("notes/" + id + ".json")
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return err
		}
		raw, readErr := io.ReadAll(io.LimitReader(f, (1<<20)+1))
		if err := errors.Join(readErr, f.Close()); err != nil {
			return err
		}
		if len(raw) > 1<<20 {
			return fmt.Errorf("distillation exceeds 1 MiB")
		}
		var note struct {
			Version      int    `json:"version"`
			PaperID      string `json:"paper_id"`
			RecordSHA256 string `json:"record_sha256"`
			SourceScope  string `json:"source_scope"`
			ReviewStatus string `json:"review_status"`
			Eligible     bool   `json:"eligible"`
		}
		if err := json.Unmarshal(raw, &note); err != nil {
			return err
		}
		if note.Version != 1 || note.PaperID != id || "sha256:"+note.RecordSHA256 != card.Version || note.SourceScope != "full_paper" || note.Eligible || note.ReviewStatus != "source_anchors_checked_semantic_review_required" {
			return fmt.Errorf("distillation identity, source or review-state mismatch for %s", id)
		}
		card.Distillation = raw
		card.Eligible = false
		card.SupportGap = "Full-paper guidance available; source anchors checked, semantic and execution review still required."
	}
	return nil
}
