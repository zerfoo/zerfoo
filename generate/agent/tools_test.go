package agent

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestToolRegistry(t *testing.T) {
	r := NewToolRegistry()

	err := r.Register(ToolDef{
		Name:        "echo",
		Description: "Echoes the input",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"msg":{"type":"string"}}}`),
	}, func(args json.RawMessage) (string, error) {
		var p struct{ Msg string `json:"msg"` }
		if err := json.Unmarshal(args, &p); err != nil {
			return "", err
		}
		return p.Msg, nil
	})
	if err != nil {
		t.Fatalf("register echo: %v", err)
	}

	err = r.Register(ToolDef{
		Name:        "add",
		Description: "Adds two numbers",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}`),
	}, func(args json.RawMessage) (string, error) {
		var p struct{ A, B float64 }
		if err := json.Unmarshal(args, &p); err != nil {
			return "", err
		}
		return fmt.Sprintf("%g", p.A+p.B), nil
	})
	if err != nil {
		t.Fatalf("register add: %v", err)
	}

	// Call echo.
	res := r.Call(ToolCall{ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"msg":"hello"}`)})
	if res.IsError {
		t.Fatalf("echo call failed: %s", res.Output)
	}
	if res.CallID != "c1" {
		t.Errorf("CallID = %q, want %q", res.CallID, "c1")
	}
	if res.Output != "hello" {
		t.Errorf("Output = %q, want %q", res.Output, "hello")
	}

	// Call add.
	res = r.Call(ToolCall{ID: "c2", Name: "add", Arguments: json.RawMessage(`{"A":3,"B":4}`)})
	if res.IsError {
		t.Fatalf("add call failed: %s", res.Output)
	}
	if res.Output != "7" {
		t.Errorf("Output = %q, want %q", res.Output, "7")
	}
}

func TestToolCallError(t *testing.T) {
	r := NewToolRegistry()
	_ = r.Register(ToolDef{Name: "fail", Description: "Always fails"}, func(args json.RawMessage) (string, error) {
		return "", fmt.Errorf("something went wrong")
	})

	res := r.Call(ToolCall{ID: "c1", Name: "fail"})
	if !res.IsError {
		t.Fatal("expected IsError=true")
	}
	if res.Output != "something went wrong" {
		t.Errorf("Output = %q, want %q", res.Output, "something went wrong")
	}
}

func TestToolNotFound(t *testing.T) {
	r := NewToolRegistry()

	res := r.Call(ToolCall{ID: "c1", Name: "nonexistent"})
	if !res.IsError {
		t.Fatal("expected IsError=true")
	}
	if got := res.Output; got == "" {
		t.Fatal("expected non-empty output")
	}
	// Must contain "not found".
	if !contains(res.Output, "not found") {
		t.Errorf("Output = %q, want it to contain %q", res.Output, "not found")
	}
}

func TestToolPanic(t *testing.T) {
	r := NewToolRegistry()
	_ = r.Register(ToolDef{Name: "boom", Description: "Panics"}, func(args json.RawMessage) (string, error) {
		panic("kaboom")
	})

	res := r.Call(ToolCall{ID: "c1", Name: "boom"})
	if !res.IsError {
		t.Fatal("expected IsError=true")
	}
	if !contains(res.Output, "panic") {
		t.Errorf("Output = %q, want it to contain %q", res.Output, "panic")
	}
}

func TestToolList(t *testing.T) {
	r := NewToolRegistry()
	noop := func(json.RawMessage) (string, error) { return "", nil }

	_ = r.Register(ToolDef{Name: "zeta"}, noop)
	_ = r.Register(ToolDef{Name: "alpha"}, noop)
	_ = r.Register(ToolDef{Name: "mid"}, noop)

	defs := r.List()
	if len(defs) != 3 {
		t.Fatalf("len = %d, want 3", len(defs))
	}
	want := []string{"alpha", "mid", "zeta"}
	for i, d := range defs {
		if d.Name != want[i] {
			t.Errorf("List()[%d].Name = %q, want %q", i, d.Name, want[i])
		}
	}
}

func TestRegisterDuplicate(t *testing.T) {
	r := NewToolRegistry()
	noop := func(json.RawMessage) (string, error) { return "", nil }
	_ = r.Register(ToolDef{Name: "dup"}, noop)
	err := r.Register(ToolDef{Name: "dup"}, noop)
	if err == nil {
		t.Fatal("expected error for duplicate registration")
	}
}

func TestRegisterEmptyName(t *testing.T) {
	r := NewToolRegistry()
	err := r.Register(ToolDef{Name: ""}, func(json.RawMessage) (string, error) { return "", nil })
	if err == nil {
		t.Fatal("expected error for empty name")
	}
}

func TestGet(t *testing.T) {
	r := NewToolRegistry()
	_ = r.Register(ToolDef{Name: "lookup", Description: "test"}, func(json.RawMessage) (string, error) { return "", nil })

	def, ok := r.Get("lookup")
	if !ok {
		t.Fatal("expected ok=true")
	}
	if def.Description != "test" {
		t.Errorf("Description = %q, want %q", def.Description, "test")
	}

	_, ok = r.Get("missing")
	if ok {
		t.Fatal("expected ok=false for missing tool")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && searchSubstring(s, sub)
}

func searchSubstring(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
