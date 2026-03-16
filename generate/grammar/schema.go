// Package grammar converts a subset of JSON Schema into a context-free grammar
// state machine that can constrain token-by-token generation to produce only
// valid JSON conforming to the schema.
package grammar

// JSONSchema represents the subset of JSON Schema supported by the converter.
// Unsupported features ($ref, oneOf, anyOf, allOf, pattern, additionalProperties)
// cause Convert to return an error.
type JSONSchema struct {
	Type       string                 // "object", "array", "string", "number", "integer", "boolean", "null"
	Properties map[string]*JSONSchema // for "object"
	Required   []string              // for "object"
	Items      *JSONSchema           // for "array"
	Enum       []any                 // enum values
	Const      any                   // const value
	MinLength  int                   // for "string"
	MaxLength  int                   // for "string", 0 = unlimited

	// Unsupported fields - presence triggers an error from Convert.
	Ref                  string `json:"$ref,omitempty"`
	OneOf                []any  `json:"oneOf,omitempty"`
	AnyOf                []any  `json:"anyOf,omitempty"`
	AllOf                []any  `json:"allOf,omitempty"`
	Pattern              string `json:"pattern,omitempty"`
	AdditionalProperties *bool  `json:"additionalProperties,omitempty"`
}
