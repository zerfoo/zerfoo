package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func invoke(t *testing.T, s *service, name string, args any) any {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	result, err := s.call(context.Background(), name, raw)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return result
}
func setupPlan(t *testing.T) (*service, plan) {
	t.Helper()
	s, err := newService(t.TempDir(), "../../tabular/testdata/model_creation", "")
	if err != nil {
		t.Fatal(err)
	}
	p := invoke(t, s, "project_create", map[string]any{"objective": "Classify iris species"}).(project)
	d := invoke(t, s, "dataset_inspect", map[string]any{"project": p.ID, "path": "iris.csv", "target": "species", "seed": 42}).(datasetRecord)
	result := invoke(t, s, "plan_create", map[string]any{"project": p.ID, "dataset": d.ID, "rationale": "Small ReLU network supported by the capability registry", "hidden_dims": []int{16}, "epochs": 20, "batch_size": 15, "learning_rate": 0.01, "seed": 42}).(plan)
	return s, result
}
func TestCreationLifecycle(t *testing.T) {
	s, p := setupPlan(t)
	launches := 0
	s.launch = func(string) error { launches++; return nil }
	args := map[string]any{"plan": p.ID, "idempotency_key": "first"}
	r := invoke(t, s, "run_start", args).(run)
	same := invoke(t, s, "run_start", args).(run)
	if r.ID != same.ID || launches != 1 {
		t.Fatal("retry duplicated job")
	}
	if err := s.work(context.Background(), r.ID); err != nil {
		t.Fatal(err)
	}
	reopened, err := newService(s.state, s.data, "")
	if err != nil {
		t.Fatal(err)
	}
	done := invoke(t, reopened, "run_status", map[string]any{"id": r.ID}).(run)
	if done.Status != "succeeded" || done.Progress.Steps == 0 || done.Validation == nil || done.Validation.Accuracy < 0.8 || done.Hash == "" {
		t.Fatalf("invalid trained result: %+v", done)
	}
	result := invoke(t, reopened, "model_predict", map[string]any{"run": r.ID, "rows": [][]float64{{5.1, 3.5, 1.4, 0.2}}}).(map[string]any)
	if result["artifact_sha256"] != done.Hash {
		t.Fatal("prediction used different model")
	}
	if err := s.work(context.Background(), r.ID); err == nil {
		t.Fatal("duplicate worker reran terminal job")
	}
}
func TestCreationRejectsInvalidDesignAndPath(t *testing.T) {
	s, p := setupPlan(t)
	cases := []struct {
		name, tool string
		args       any
	}{
		{"path_escape", "dataset_inspect", map[string]any{"project": p.Project, "path": "../secret.csv", "target": "species", "seed": 42}},
		{"unsupported_operator", "plan_create", map[string]any{"attention": true}},
		{"unknown_paper", "plan_create", map[string]any{"project": p.Project, "dataset": p.Dataset, "rationale": "claim", "evidence": []string{"nonexistent"}, "epochs": 1, "batch_size": 1, "learning_rate": 0.01}},
		{"excessive_width", "plan_create", map[string]any{"project": p.Project, "dataset": p.Dataset, "rationale": "claim", "hidden_dims": []int{4096}, "epochs": 1, "batch_size": 1, "learning_rate": 0.01}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			raw, err := json.Marshal(test.args)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := s.call(context.Background(), test.tool, raw); err == nil {
				t.Fatal("invalid request accepted")
			}
		})
	}
}
func TestCreationCancelAndRecovery(t *testing.T) {
	s, p := setupPlan(t)
	s.launch = func(string) error { return nil }
	r := invoke(t, s, "run_start", map[string]any{"plan": p.ID, "idempotency_key": "cancel"}).(run)
	invoke(t, s, "run_cancel", map[string]any{"id": r.ID})
	if err := s.work(context.Background(), r.ID); err == nil {
		t.Fatal("canceled run trained")
	}
	canceled := invoke(t, s, "run_status", map[string]any{"id": r.ID}).(run)
	if canceled.Status != "canceled" {
		t.Fatalf("got %s", canceled.Status)
	}
	r = invoke(t, s, "run_start", map[string]any{"plan": p.ID, "idempotency_key": "crash"}).(run)
	recovered := invoke(t, s, "run_recover", map[string]any{"id": r.ID}).(run)
	if recovered.Status != "interrupted" {
		t.Fatal("dead worker reported success")
	}
}
func TestResearchEligibility(t *testing.T) {
	s, err := newService(t.TempDir(), ".", filepath.Join(t.TempDir(), "cards.json"))
	if err != nil {
		t.Fatal(err)
	}
	catalog := `{"version":1,"cards":[{"id":"supported","title":"Linear classifier","url":"https://example.org/paper","version":"1","section":"2","summary":"Evidence only","components":["linear"]},{"id":"unsupported","title":"Attention","url":"https://example.org/other","version":"1","section":"3","summary":"Not executable","components":["attention"]}]}`
	if err := os.WriteFile(s.library, []byte(catalog), 0600); err != nil {
		t.Fatal(err)
	}
	cards, err := s.search("")
	if err != nil {
		t.Fatal(err)
	}
	if len(cards) != 1 || cards[0].ID != "supported" {
		t.Fatal("unsupported research presented as executable")
	}
}
func TestMCPProtocol(t *testing.T) {
	s, err := newService(t.TempDir(), ".", "")
	if err != nil {
		t.Fatal(err)
	}
	input := strings.Join([]string{`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"capabilities","arguments":{}}}`},
		"\n")
	var out bytes.Buffer
	if err := serveMCP(context.Background(), s, strings.NewReader(input), &out); err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSpace(out.Bytes()), []byte("\n"))
	if len(lines) != 3 {
		t.Fatalf("notifications got responses: %s", out.String())
	}
	for _, line := range lines {
		var response map[string]any
		if err := json.Unmarshal(line, &response); err != nil {
			t.Fatal(err)
		}
		if response["result"] == nil {
			t.Fatalf("bad response: %s", line)
		}
	}
	if !strings.Contains(out.String(), "run_start") || !strings.Contains(out.String(), "numeric_classification") {
		t.Fatal("missing tool or actual capability result")
	}
}

func TestCreationIdempotencyConflictAndActiveLimit(t *testing.T) {
	s, p := setupPlan(t)
	s.launch = func(string) error { return nil }
	invoke(t, s, "run_start", map[string]any{"plan": p.ID, "idempotency_key": "same"})
	raw, err := json.Marshal(map[string]any{"plan": p.ID, "idempotency_key": "another"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.call(context.Background(), "run_start", raw); err == nil {
		t.Fatal("concurrent compute accepted")
	}
	other := p
	other.ID = "other-plan"
	if err := s.save("plan/"+other.ID, other); err != nil {
		t.Fatal(err)
	}
	raw, err = json.Marshal(map[string]any{"plan": other.ID, "idempotency_key": "same"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.call(context.Background(), "run_start", raw); err == nil {
		t.Fatal("conflicting idempotency accepted")
	}
}
