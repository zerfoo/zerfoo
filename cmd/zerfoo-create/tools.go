package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/model/dsl"
	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	bolt "go.etcd.io/bbolt"
)

func strictJSON(raw []byte, value any) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(value); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("expected one JSON object")
	}
	return nil
}
func (s *service) call(ctx context.Context, name string, raw json.RawMessage) (any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch name {
	case "capabilities":
		return map[string]any{"version": 2,
				"task":       "numeric_classification",
				"components": model.ListComponents(), "definition_schema": dsl.DefinitionSchema(),
				"device":               "cpu",
				"max_hidden_layers":    4,
				"max_hidden_width":     1024,
				"max_epochs":           200,
				"max_duration_seconds": 120,
				"planner":              "calling coding agent",
				"research_configured":  s.library != "",
				"deployment":           "model_predict provides local raw-feature prediction; HTTP serving is not included"},
			nil
	case "project_create":
		var a struct {
			Objective string `json:"objective"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		if strings.TrimSpace(a.Objective) == "" {
			return nil, fmt.Errorf("objective required")
		}
		id, err := newID()
		if err != nil {
			return nil, err
		}
		p := project{id, a.Objective}
		return p, s.save("project/"+id, p)
	case "project_get":
		var a struct {
			ID string `json:"id"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		var p project
		err := s.load("project/"+a.ID, &p)
		return p, err
	case "dataset_inspect":
		var a struct {
			Project string `json:"project"`
			Path    string `json:"path"`
			Target  string `json:"target"`
			Group   string `json:"group,omitempty"`
			Time    string `json:"time,omitempty"`
			Split   string `json:"split,omitempty"`
			Seed    uint64 `json:"seed"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		var p project
		if err := s.load("project/"+a.Project, &p); err != nil {
			return nil, err
		}
		root, err := os.OpenRoot(s.data)
		if err != nil {
			return nil, err
		}
		file, err := root.Open(a.Path)
		if err != nil {
			return nil, errors.Join(err, root.Close())
		}
		data, readErr := io.ReadAll(io.LimitReader(file, (64<<20)+1))
		if err := errors.Join(readErr, file.Close(), root.Close()); err != nil {
			return nil, err
		}
		if len(data) > 64<<20 {
			return nil, fmt.Errorf("CSV exceeds 64 MiB")
		}
		if a.Split == "" {
			a.Split = "stratified"
		}
		dataset, err := tabular.InspectCSV(ctx, bytes.NewReader(data), tabular.DatasetOptions{Target: a.Target, Group: a.Group, Time: a.Time, Split: a.Split, Seed: a.Seed})
		if err != nil {
			return nil, err
		}
		id, err := newID()
		if err != nil {
			return nil, err
		}
		state, err := os.OpenRoot(s.state)
		if err != nil {
			return nil, err
		}
		if err := errors.Join(state.WriteFile(id+".csv", data, 0600), state.Close()); err != nil {
			return nil, err
		}
		record := datasetRecord{id, a.Project, dataset.Manifest()}
		return record, s.save("dataset/"+id, record)
	case "research_search":
		var a struct {
			Query string `json:"query"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		return s.search(ctx, a.Query)
	case "plan_create":
		var a struct {
			Project    string          `json:"project"`
			Dataset    string          `json:"dataset"`
			Rationale  string          `json:"rationale"`
			Evidence   []string        `json:"evidence"`
			Definition *dsl.Definition `json:"definition,omitempty"`
			Hidden     []int           `json:"hidden_dims"`
			Epochs     int             `json:"epochs"`
			Batch      int             `json:"batch_size"`
			Rate       float64         `json:"learning_rate"`
			Seed       uint64          `json:"seed"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		var d datasetRecord
		if err := s.load("dataset/"+a.Dataset, &d); err != nil {
			return nil, err
		}
		if d.Project != a.Project {
			return nil, fmt.Errorf("dataset belongs to a different project")
		}
		if a.Rationale == "" || a.Epochs < 1 || a.Epochs > 200 || a.Batch < 1 || a.Batch > 256 || a.Rate <= 0 || a.Rate > 1 || len(a.Hidden) > 4 {
			return nil, fmt.Errorf("invalid rationale or training limits; inspect capabilities")
		}
		for _, width := range a.Hidden {
			if width < 1 || width > 1024 {
				return nil, fmt.Errorf("hidden width must be in [1,1024]")
			}
		}
		cards, err := s.search(ctx, "")
		if err != nil && len(a.Evidence) > 0 {
			return nil, err
		}
		known := map[string]bool{}
		for _, card := range cards {
			known[card.ID] = card.Eligible
		}
		for _, id := range a.Evidence {
			if !known[id] {
				return nil, fmt.Errorf("unknown or unqualified evidence %q", id)
			}
		}
		id, err := newID()
		if err != nil {
			return nil, err
		}
		p := plan{Version: 2, ID: id,
			Project:   a.Project,
			Dataset:   a.Dataset,
			Rationale: a.Rationale,
			Evidence:  a.Evidence,
			Config: tabular.ClassifierConfig{InputDim: len(d.Manifest.Options.Features),
				ClassCount: len(d.Manifest.Labels),
				Labels:     d.Manifest.Labels,
				HiddenDims: a.Hidden,
				Seed:       a.Seed},
			Options: tabular.FitOptions{Epochs: a.Epochs,
				BatchSize:    a.Batch,
				LearningRate: a.Rate,
				WeightDecay:  0.0001,
				Seed:         a.Seed,
				MaxDuration:  120 * time.Second}}

		if a.Definition != nil && len(a.Hidden) > 0 {
			return nil, fmt.Errorf("provide definition or legacy hidden_dims, not both")
		}
		p.Config.Definition = a.Definition
		definition, err := tabular.ClassifierDefinition(p.Config)
		if err != nil {
			return nil, err
		}
		p.Config.Definition = &definition
		p.Config.HiddenDims = nil
		checked, err := dsl.ValidateExecution(definition, "training", "cpu", "float32", "eager")
		if err != nil {
			return nil, err
		}
		p.DefinitionSHA256 = checked.ID
		for _, card := range cards {
			for _, id := range a.Evidence {
				if card.ID == id {
					p.EvidenceCards = append(p.EvidenceCards, card)
				}
			}
		}
		if _, err := tabular.NewClassifier(p.Config, compute.NewCPUEngine(numeric.Float32Ops{})); err != nil {
			return nil, err
		}
		return p, s.save("plan/"+id, p)
	case "plan_get":
		var a struct {
			ID string `json:"id"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		var p plan
		err := s.load("plan/"+a.ID, &p)
		return p, err
	case "run_start":
		var a struct {
			Plan string `json:"plan"`
			Key  string `json:"idempotency_key"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		if a.Key == "" || len(a.Key) > 256 {
			return nil, fmt.Errorf("idempotency_key required, max 256 bytes")
		}
		var p plan
		if err := s.load("plan/"+a.Plan, &p); err != nil {
			return nil, err
		}
		id, err := newID()
		if err != nil {
			return nil, err
		}
		r := run{ID: id, Plan: p.ID, Status: "queued", Created: time.Now().UTC(), Updated: time.Now().UTC()}
		created := false
		err = s.transaction(func(b *bolt.Bucket) error {
			key := "request/" + a.Key
			if prior := b.Get([]byte(key)); prior != nil {
				if err := get(b, "run/"+string(prior), &r); err != nil {
					return err
				}
				if r.Plan != p.ID {
					return fmt.Errorf("idempotency key used for another plan")
				}
				return nil
			}

			if err := b.ForEach(func(k, v []byte) error {
				if !strings.HasPrefix(string(k), "run/") {
					return nil
				}
				var active run
				if err := json.Unmarshal(v, &active); err != nil {
					return err
				}
				if active.Status == "queued" || active.Status == "running" {
					return fmt.Errorf("run %s is active; cancel or recover it before starting another", active.ID)
				}
				return nil
			}); err != nil {
				return err
			}
			if err := put(b, "run/"+id, r); err != nil {
				return err
			}
			if err := b.Put([]byte(key), []byte(id)); err != nil {
				return err
			}
			created = true
			return nil
		})
		if err != nil {
			return nil, err
		}
		if created {
			if s.launch == nil {
				err = fmt.Errorf("worker launcher unavailable")
			} else {
				err = s.launch(id)
			}
			if err != nil {
				r.Status = "failed"
				r.Error = err.Error()
				return r, errors.Join(err, s.save("run/"+id, r))
			}
		}
		return r, nil

	case "run_recover":
		var a struct {
			ID string `json:"id"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		lease, err := bolt.Open(filepath.Join(s.state, "worker.db"), 0600, &bolt.Options{Timeout: time.Second})
		if err != nil {
			return nil, fmt.Errorf("worker is still active: %w", err)
		}
		var r run
		err = s.transaction(func(b *bolt.Bucket) error {
			if err := get(b, "run/"+a.ID, &r); err != nil {
				return err
			}
			if r.Status == "queued" || r.Status == "running" {
				r.Status = "interrupted"
				r.Error = "worker absent; restart requires a new idempotency key"
				r.Updated = time.Now().UTC()
				return put(b, "run/"+r.ID, r)
			}
			return nil
		})
		return r, errors.Join(err, lease.Close())
	case "run_status", "run_cancel":
		var a struct {
			ID string `json:"id"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		var r run
		err := s.transaction(func(b *bolt.Bucket) error {
			if err := get(b, "run/"+a.ID, &r); err != nil {
				return err
			}
			if name == "run_cancel" && (r.Status == "queued" || r.Status == "running") {
				r.Cancel = true
				r.Updated = time.Now().UTC()
				return put(b, "run/"+r.ID, r)
			}
			return nil
		})
		return r, err
	case "model_predict":
		var a struct {
			Run  string      `json:"run"`
			Rows [][]float64 `json:"rows"`
		}
		if err := decode(raw, &a); err != nil {
			return nil, err
		}
		if len(a.Rows) == 0 || len(a.Rows) > 256 {
			return nil, fmt.Errorf("rows must contain 1–256 records")
		}
		var r run
		if err := s.load("run/"+a.Run, &r); err != nil {
			return nil, err
		}
		if r.Status != "succeeded" {
			return nil, fmt.Errorf("run has no completed model")
		}
		bundle, err := tabular.LoadClassifierBundle(ctx, r.Artifact, r.Hash, compute.NewCPUEngine(numeric.Float32Ops{}))
		if err != nil {
			return nil, err
		}
		out := make([]tabular.Prediction, len(a.Rows))
		for i, row := range a.Rows {
			out[i], err = bundle.PredictRaw(ctx, row)
			if err != nil {
				return nil, err
			}
		}
		return map[string]any{"artifact_sha256": r.Hash, "features": bundle.Manifest.Features, "predictions": out}, nil
	default:
		return nil, fmt.Errorf("unknown tool %q", name)
	}
}
