package agent

import (
	"encoding/json"
	"fmt"
	"sort"
	"sync"
)

// ToolDef describes a tool that can be invoked by an agent.
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolCall represents a request from a model to invoke a tool.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolResult is the outcome of executing a ToolCall.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Output  string `json:"output"`
	IsError bool   `json:"is_error,omitempty"`
}

type toolEntry struct {
	def ToolDef
	fn  func(args json.RawMessage) (string, error)
}

// ToolRegistry maps tool names to their definitions and handler functions.
type ToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]toolEntry
}

// NewToolRegistry creates an empty ToolRegistry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{tools: make(map[string]toolEntry)}
}

// Register adds a tool to the registry. It returns an error if the name is
// empty or already registered.
func (r *ToolRegistry) Register(def ToolDef, fn func(args json.RawMessage) (string, error)) error {
	if def.Name == "" {
		return fmt.Errorf("tool name must not be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[def.Name]; exists {
		return fmt.Errorf("tool %q already registered", def.Name)
	}
	r.tools[def.Name] = toolEntry{def: def, fn: fn}
	return nil
}

// Call invokes the handler for the named tool. If the tool is not found, or if
// the handler returns an error or panics, the result has IsError set to true.
func (r *ToolRegistry) Call(call ToolCall) (result ToolResult) {
	result.CallID = call.ID

	r.mu.RLock()
	entry, ok := r.tools[call.Name]
	r.mu.RUnlock()

	if !ok {
		result.IsError = true
		result.Output = fmt.Sprintf("tool %q not found", call.Name)
		return result
	}

	defer func() {
		if rec := recover(); rec != nil {
			result.IsError = true
			result.Output = fmt.Sprintf("panic in tool %q: %v", call.Name, rec)
		}
	}()

	output, err := entry.fn(call.Arguments)
	if err != nil {
		result.IsError = true
		result.Output = err.Error()
		return result
	}
	result.Output = output
	return result
}

// List returns all registered tool definitions sorted by name.
func (r *ToolRegistry) List() []ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()
	defs := make([]ToolDef, 0, len(r.tools))
	for _, e := range r.tools {
		defs = append(defs, e.def)
	}
	sort.Slice(defs, func(i, j int) bool { return defs[i].Name < defs[j].Name })
	return defs
}

// Get looks up a tool definition by name.
func (r *ToolRegistry) Get(name string) (ToolDef, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entry, ok := r.tools[name]
	if !ok {
		return ToolDef{}, false
	}
	return entry.def, true
}
