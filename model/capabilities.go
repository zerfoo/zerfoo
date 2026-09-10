package model

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
)

// AttributeSpec is a machine-readable scalar configuration constraint.
type AttributeSpec struct {
	Type     string   `json:"type"`
	Required bool     `json:"required,omitempty"`
	Minimum  *float64 `json:"minimum,omitempty"`
	Maximum  *float64 `json:"maximum,omitempty"`
}

// ExecutionSupport scopes evidence to an operation and execution mode. A
// registered builder without evidence is discoverable, not qualified.
type ExecutionSupport struct {
	Operation string   `json:"operation"`
	Device    string   `json:"device"`
	Precision string   `json:"precision"`
	Mode      string   `json:"mode"`
	Status    string   `json:"status"`
	Evidence  []string `json:"evidence,omitempty"`
}

// ComponentDescriptor is shared by graph construction and application adapters.
// Parameter slots refer to explicitly declared, potentially shared tensors.
type ComponentDescriptor struct {
	ID         string                   `json:"id"`
	Version    int                      `json:"version"`
	Kind       string                   `json:"kind"`
	Inputs     []string                 `json:"inputs"`
	Outputs    []string                 `json:"outputs"`
	Parameters []string                 `json:"parameters"`
	Attributes map[string]AttributeSpec `json:"attributes"`
	Support    []ExecutionSupport       `json:"support"`
}

// ShapeRule validates all ports, parameter slots and attributes before allocation.
type ShapeRule func(inputs map[string][]int, parameters map[string][]int, attributes map[string]any) ([]int, error)
type componentEntry struct {
	descriptor ComponentDescriptor
	shape      ShapeRule
}

var components = struct {
	sync.RWMutex
	entries map[string]componentEntry
}{entries: map[string]componentEntry{}}

// RegisterComponent annotates an existing builder. Its descriptor is copied.
func RegisterComponent(descriptor ComponentDescriptor, shape ShapeRule) error {
	if descriptor.ID == "" || descriptor.Version != 1 || (descriptor.Kind != "operator" && descriptor.Kind != "architecture" && descriptor.Kind != "loss" && descriptor.Kind != "optimizer") {
		return fmt.Errorf("model: invalid component identity")
	}
	raw, err := json.Marshal(descriptor)
	if err != nil {
		return err
	}
	var copy ComponentDescriptor
	if err := json.Unmarshal(raw, &copy); err != nil {
		return err
	}
	components.Lock()
	defer components.Unlock()
	components.entries[descriptor.Kind+"/"+descriptor.ID] = componentEntry{copy, shape}
	return nil
}

// Component returns owned metadata and its allocation-free validator.
func Component(kind, id string) (ComponentDescriptor, ShapeRule, bool) {
	components.RLock()
	entry, ok := components.entries[kind+"/"+id]
	components.RUnlock()
	if !ok {
		return ComponentDescriptor{}, nil, false
	}
	raw, err := json.Marshal(entry.descriptor)
	if err != nil {
		return ComponentDescriptor{}, nil, false
	}
	var copy ComponentDescriptor
	if json.Unmarshal(raw, &copy) != nil {
		return ComponentDescriptor{}, nil, false
	}
	return copy, entry.shape, true
}

// ListComponents includes unannotated layer builders as registered/unverified.
func ListComponents() []ComponentDescriptor {
	result := []ComponentDescriptor{}
	components.RLock()
	keys := make([]string, 0, len(components.entries))
	for key := range components.entries {
		keys = append(keys, key)
	}
	components.RUnlock()
	sort.Strings(keys)
	seen := map[string]bool{}
	for _, key := range keys {
		components.RLock()
		entry := components.entries[key]
		components.RUnlock()
		d, _, ok := Component(entry.descriptor.Kind, entry.descriptor.ID)
		if ok {
			result = append(result, d)
			seen[key] = true
		}
	}
	for _, name := range ListLayerBuilders() {
		if !seen["operator/"+name] {
			result = append(result, ComponentDescriptor{ID: name, Kind: "operator", Version: 1, Support: []ExecutionSupport{{Operation: "construction", Status: "unverified"}}})
		}
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Kind != result[j].Kind {
			return result[i].Kind < result[j].Kind
		}
		return result[i].ID < result[j].ID
	})
	return result
}

// ObserveArchitecture records discovery without asserting executable training.
func ObserveArchitecture(name string) {
	d := ComponentDescriptor{ID: name, Version: 1, Kind: "architecture", Support: []ExecutionSupport{{Operation: "inference", Status: "unverified"}}}
	components.Lock()
	defer components.Unlock()
	components.entries["architecture/"+name] = componentEntry{descriptor: d}
}

// ValidateAttributes enforces declared scalar schemas and normalizes JSON
// integer values to the Go ints expected by existing layer builders.
func ValidateAttributes(descriptor ComponentDescriptor, attributes map[string]any) (map[string]any, error) {
	result := map[string]any{}
	for name, spec := range descriptor.Attributes {
		if spec.Required {
			if _, ok := attributes[name]; !ok {
				return nil, fmt.Errorf("missing attribute %s", name)
			}
		}
	}
	for name, value := range attributes {
		spec, ok := descriptor.Attributes[name]
		if !ok {
			return nil, fmt.Errorf("unknown attribute %s", name)
		}
		switch spec.Type {
		case "integer", "number":
			var number float64
			switch v := value.(type) {
			case float64:
				number = v
			case int:
				number = float64(v)
			case int64:
				number = float64(v)
			default:
				return nil, fmt.Errorf("attribute %s must be numeric", name)
			}
			if math.IsNaN(number) || math.IsInf(number, 0) || (spec.Minimum != nil && number < *spec.Minimum) || (spec.Maximum != nil && number > *spec.Maximum) {
				return nil, fmt.Errorf("attribute %s is outside its bounds", name)
			}
			if spec.Type == "integer" {
				if math.Trunc(number) != number || number >= float64(int(^uint(0)>>1)) || number < float64(-int(^uint(0)>>1)-1) {
					return nil, fmt.Errorf("attribute %s must be a representable integer", name)
				}
				result[name] = int(number)
			} else {
				result[name] = number
			}
		case "string":
			v, ok := value.(string)
			if !ok {
				return nil, fmt.Errorf("attribute %s must be a string", name)
			}
			result[name] = v
		case "boolean":
			v, ok := value.(bool)
			if !ok {
				return nil, fmt.Errorf("attribute %s must be boolean", name)
			}
			result[name] = v
		default:
			return nil, fmt.Errorf("unsupported attribute schema %s", spec.Type)
		}
	}
	return result, nil
}
