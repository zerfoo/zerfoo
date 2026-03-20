package grammar

import (
	"testing"
)

// feedString advances the grammar through every byte in s.
// Returns the resulting grammar and true, or nil and false if any byte is rejected.
func feedString(g *Grammar, s string) (*Grammar, bool) {
	for i := 0; i < len(s); i++ {
		next, ok := g.Advance(s[i])
		if !ok {
			return nil, false
		}
		g = next
	}
	return g, true
}

func TestNullSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "null"})
	if err != nil {
		t.Fatal(err)
	}
	final, ok := feedString(g, "null")
	if !ok {
		t.Fatal("expected null to be accepted")
	}
	if !final.IsComplete() {
		t.Fatal("expected IsComplete after null")
	}
	// Reject invalid.
	if _, ok := feedString(g, "nul"); ok {
		// "nul" should feed ok but not be complete.
		if g2, _ := feedString(g, "nul"); g2 != nil && g2.IsComplete() {
			t.Fatal("nul should not be complete")
		}
	}
	if _, ok := feedString(g, "NULL"); ok {
		t.Fatal("NULL should be rejected")
	}
}

func TestBooleanSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "boolean"})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input    string
		accepted bool
		complete bool
	}{
		{"true", true, true},
		{"false", true, true},
		{"tru", true, false},
		{"fals", true, false},
		{"True", false, false},
		{"1", false, false},
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if ok != tc.accepted {
				t.Fatalf("feedString(%q): got accepted=%v, want %v", tc.input, ok, tc.accepted)
			}
			if ok && final.IsComplete() != tc.complete {
				t.Fatalf("IsComplete(%q): got %v, want %v", tc.input, final.IsComplete(), tc.complete)
			}
		})
	}
}

func TestIntegerSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "integer"})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input    string
		accepted bool
		complete bool
	}{
		{"0", true, true},
		{"1", true, true},
		{"42", true, true},
		{"-1", true, true},
		{"-0", true, true},
		{"123", true, true},
		// Leading zero followed by digit should be rejected.
		{"00", false, false},
		{"01", false, false},
		{"-", true, false},
		{"abc", false, false},
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if ok != tc.accepted {
				t.Fatalf("feedString(%q): got accepted=%v, want %v", tc.input, ok, tc.accepted)
			}
			if ok && final.IsComplete() != tc.complete {
				t.Fatalf("IsComplete(%q): got %v, want %v", tc.input, final.IsComplete(), tc.complete)
			}
		})
	}
}

func TestNumberSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "number"})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input    string
		accepted bool
		complete bool
	}{
		{"0", true, true},
		{"42", true, true},
		{"-1", true, true},
		{"3.14", true, true},
		{"0.5", true, true},
		{"-0.5", true, true},
		{"1e10", true, true},
		{"1E10", true, true},
		{"1e+10", true, true},
		{"1e-10", true, true},
		{"1.5e2", true, true},
		{"-", true, false},
		{"1.", true, false},
		{"1e", true, false},
		{"1e+", true, false},
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if ok != tc.accepted {
				t.Fatalf("feedString(%q): got accepted=%v, want %v", tc.input, ok, tc.accepted)
			}
			if ok && final.IsComplete() != tc.complete {
				t.Fatalf("IsComplete(%q): got %v, want %v", tc.input, final.IsComplete(), tc.complete)
			}
		})
	}
}

func TestStringSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "string"})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input    string
		accepted bool
		complete bool
	}{
		{`""`, true, true},
		{`"hello"`, true, true},
		{`"he`, true, false},
		{`"a\\b"`, true, true},  // escaped backslash
		{`"a\"b"`, true, true},  // escaped quote
		{`"a\nb"`, true, true},  // escaped newline
		{`hello`, false, false}, // no opening quote
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if ok != tc.accepted {
				t.Fatalf("feedString(%q): got accepted=%v, want %v", tc.input, ok, tc.accepted)
			}
			if ok && final.IsComplete() != tc.complete {
				t.Fatalf("IsComplete(%q): got %v, want %v", tc.input, final.IsComplete(), tc.complete)
			}
		})
	}
}

func TestStringMinMaxLength(t *testing.T) {
	// MinLength=2, MaxLength=4
	g, err := Convert(&JSONSchema{Type: "string", MinLength: 2, MaxLength: 4})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input    string
		accepted bool
		complete bool
	}{
		{`"a"`, false, false},    // too short — closing quote rejected
		{`"ab"`, true, true},     // exactly 2
		{`"abcd"`, true, true},   // exactly 4
		{`"abcde"`, false, false}, // 5th char rejected
		{`"abc"`, true, true},    // 3, within range
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if ok != tc.accepted {
				t.Fatalf("feedString(%q): got accepted=%v, want %v", tc.input, ok, tc.accepted)
			}
			if ok && final.IsComplete() != tc.complete {
				t.Fatalf("IsComplete(%q): got %v, want %v", tc.input, final.IsComplete(), tc.complete)
			}
		})
	}
}

func TestConstSchema(t *testing.T) {
	// String const.
	g, err := Convert(&JSONSchema{Const: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	final, ok := feedString(g, `"hello"`)
	if !ok || !final.IsComplete() {
		t.Fatal("expected const string to match")
	}
	// "world" should be rejected — only "hello" is valid for this const.
	if _, ok := feedString(g, `"world"`); ok {
		t.Fatal("expected const string to reject non-matching value")
	}

	// Integer const.
	g2, err := Convert(&JSONSchema{Const: 42})
	if err != nil {
		t.Fatal(err)
	}
	final2, ok := feedString(g2, "42")
	if !ok || !final2.IsComplete() {
		t.Fatal("expected const int to match")
	}

	// Boolean const.
	g3, err := Convert(&JSONSchema{Const: true})
	if err != nil {
		t.Fatal(err)
	}
	final3, ok := feedString(g3, "true")
	if !ok || !final3.IsComplete() {
		t.Fatal("expected const bool to match")
	}

	// Null const.
	g4, err := Convert(&JSONSchema{Const: nil})
	if err != nil {
		t.Fatal(err)
	}
	final4, ok := feedString(g4, "null")
	if !ok || !final4.IsComplete() {
		t.Fatal("expected const null to match")
	}
}

func TestEnumSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Enum: []any{"red", "green", "blue"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input    string
		accepted bool
		complete bool
	}{
		{`"red"`, true, true},
		{`"green"`, true, true},
		{`"blue"`, true, true},
		{`"yellow"`, false, false}, // diverges at 'y'
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if ok != tc.accepted {
				t.Fatalf("feedString(%q): got accepted=%v, want %v", tc.input, ok, tc.accepted)
			}
			if ok && final.IsComplete() != tc.complete {
				t.Fatalf("IsComplete(%q): got %v, want %v", tc.input, final.IsComplete(), tc.complete)
			}
		})
	}
}

func TestEnumMixedTypes(t *testing.T) {
	g, err := Convert(&JSONSchema{Enum: []any{"hello", 42, true, nil}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input    string
		complete bool
	}{
		{`"hello"`, true},
		{`42`, true},
		{`true`, true},
		{`null`, true},
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if !ok {
				t.Fatalf("feedString(%q): rejected", tc.input)
			}
			if final.IsComplete() != tc.complete {
				t.Fatalf("IsComplete(%q): got %v, want %v", tc.input, final.IsComplete(), tc.complete)
			}
		})
	}
}

func TestEnumSpecialChars(t *testing.T) {
	g, err := Convert(&JSONSchema{Enum: []any{"a\"b", "c\\d"}})
	if err != nil {
		t.Fatal(err)
	}
	// "a\"b" → JSON literal is "a\"b" with escaping.
	final, ok := feedString(g, `"a\"b"`)
	if !ok || !final.IsComplete() {
		t.Fatal("expected enum with special chars to match")
	}
	final2, ok := feedString(g, `"c\\d"`)
	if !ok || !final2.IsComplete() {
		t.Fatal("expected enum with backslash to match")
	}
}

func TestSimpleObject(t *testing.T) {
	g, err := Convert(&JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"name": {Type: "string"},
			"age":  {Type: "integer"},
		},
		Required: []string{"name"},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Valid: {"name":"Alice","age":30}
	input := `{"name":"Alice","age":30}`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input)
	}

	// Valid: only required field.
	input2 := `{"name":"Bob"}`
	final2, ok := feedString(g, input2)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input2)
	}
	if !final2.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input2)
	}

	// Invalid: missing required field.
	input3 := `{"age":30}`
	final3, ok := feedString(g, input3)
	if ok && final3.IsComplete() {
		t.Fatalf("expected missing required field to prevent completion")
	}
}

func TestEmptyObject(t *testing.T) {
	g, err := Convert(&JSONSchema{
		Type:       "object",
		Properties: map[string]*JSONSchema{},
	})
	if err != nil {
		t.Fatal(err)
	}
	final, ok := feedString(g, `{}`)
	if !ok || !final.IsComplete() {
		t.Fatal("expected empty object to be accepted")
	}
}

func TestSimpleArray(t *testing.T) {
	g, err := Convert(&JSONSchema{
		Type:  "array",
		Items: &JSONSchema{Type: "integer"},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input    string
		accepted bool
		complete bool
	}{
		{`[]`, true, true},
		{`[1]`, true, true},
		{`[1,2,3]`, true, true},
		{`[42]`, true, true},
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if ok != tc.accepted {
				t.Fatalf("feedString(%q): got accepted=%v, want %v", tc.input, ok, tc.accepted)
			}
			if ok && final.IsComplete() != tc.complete {
				t.Fatalf("IsComplete(%q): got %v, want %v", tc.input, final.IsComplete(), tc.complete)
			}
		})
	}
}

func TestNestedObjectInArray(t *testing.T) {
	schema := &JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"items": {
				Type: "array",
				Items: &JSONSchema{
					Type: "object",
					Properties: map[string]*JSONSchema{
						"id":   {Type: "integer"},
						"name": {Type: "string"},
					},
					Required: []string{"id"},
				},
			},
		},
		Required: []string{"items"},
	}
	g, err := Convert(schema)
	if err != nil {
		t.Fatal(err)
	}
	input := `{"items":[{"id":1,"name":"a"},{"id":2}]}`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input)
	}
}

func TestDeeplyNested(t *testing.T) {
	// 5 levels deep: object > array > object > array > integer
	schema := &JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"a": {
				Type: "array",
				Items: &JSONSchema{
					Type: "object",
					Properties: map[string]*JSONSchema{
						"b": {
							Type:  "array",
							Items: &JSONSchema{Type: "integer"},
						},
					},
				},
			},
		},
	}
	g, err := Convert(schema)
	if err != nil {
		t.Fatal(err)
	}
	input := `{"a":[{"b":[1,2]}]}`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input)
	}
}

func TestEmptySchema(t *testing.T) {
	// Empty schema accepts any valid JSON.
	g, err := Convert(&JSONSchema{})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		input string
	}{
		{`null`},
		{`true`},
		{`false`},
		{`42`},
		{`"hello"`},
		{`[]`},
		{`{}`},
	} {
		t.Run(tc.input, func(t *testing.T) {
			final, ok := feedString(g, tc.input)
			if !ok {
				t.Fatalf("feedString(%q): rejected", tc.input)
			}
			if !final.IsComplete() {
				t.Fatalf("IsComplete(%q): got false, want true", tc.input)
			}
		})
	}
}

func TestUnsupportedFeatures(t *testing.T) {
	tests := []struct {
		name   string
		schema *JSONSchema
		errMsg string
	}{
		{"oneOf", &JSONSchema{OneOf: []any{1}}, "unsupported JSON Schema feature: oneOf"},
		{"anyOf", &JSONSchema{AnyOf: []any{1}}, "unsupported JSON Schema feature: anyOf"},
		{"allOf", &JSONSchema{AllOf: []any{1}}, "unsupported JSON Schema feature: allOf"},
		{"pattern", &JSONSchema{Pattern: "^a"}, "unsupported JSON Schema feature: pattern"},
		{"additionalProperties", &JSONSchema{AdditionalProperties: boolPtr(false)}, "unsupported JSON Schema feature: additionalProperties"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Convert(tc.schema)
			if err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
			if err.Error() != tc.errMsg {
				t.Fatalf("got error %q, want %q", err.Error(), tc.errMsg)
			}
		})
	}
}

func TestUnsupportedNested(t *testing.T) {
	// Unresolvable $ref nested in a property (no definitions provided).
	schema := &JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"x": {Ref: "#/$defs/X"},
		},
	}
	_, err := Convert(schema)
	if err == nil {
		t.Fatal("expected error for nested $ref without definition")
	}
}

func TestValidBytesNotEmpty(t *testing.T) {
	// For a fresh grammar, ValidBytes should not be empty.
	schemas := []*JSONSchema{
		{Type: "null"},
		{Type: "boolean"},
		{Type: "integer"},
		{Type: "number"},
		{Type: "string"},
		{Type: "object", Properties: map[string]*JSONSchema{"a": {Type: "string"}}},
		{Type: "array", Items: &JSONSchema{Type: "integer"}},
		{},
	}
	for _, s := range schemas {
		g, err := Convert(s)
		if err != nil {
			t.Fatal(err)
		}
		vb := g.ValidBytes()
		if len(vb) == 0 {
			t.Fatalf("ValidBytes() empty for type=%q", s.Type)
		}
	}
}

func TestValidBytesEmptyAtCompletion(t *testing.T) {
	// After consuming "null", ValidBytes should be empty and IsComplete true.
	g, err := Convert(&JSONSchema{Type: "null"})
	if err != nil {
		t.Fatal(err)
	}
	final, ok := feedString(g, "null")
	if !ok {
		t.Fatal("rejected")
	}
	if !final.IsComplete() {
		t.Fatal("not complete")
	}
	if len(final.ValidBytes()) != 0 {
		t.Fatalf("expected no valid bytes at completion, got %v", final.ValidBytes())
	}
}

func TestInvalidByteRejected(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "integer"})
	if err != nil {
		t.Fatal(err)
	}
	// 'a' is not a valid first byte of an integer.
	if _, ok := g.Advance('a'); ok {
		t.Fatal("expected 'a' to be rejected for integer")
	}
}

func TestObjectRequiredFieldEnforced(t *testing.T) {
	g, err := Convert(&JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"x": {Type: "integer"},
			"y": {Type: "integer"},
		},
		Required: []string{"x", "y"},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Only x provided — should not complete (missing y).
	input := `{"x":1}`
	final, ok := feedString(g, input)
	if ok && final != nil && final.IsComplete() {
		t.Fatal("should not be complete with missing required field y")
	}
}

func TestArrayOfStrings(t *testing.T) {
	g, err := Convert(&JSONSchema{
		Type:  "array",
		Items: &JSONSchema{Type: "string"},
	})
	if err != nil {
		t.Fatal(err)
	}
	input := `["a","b","c"]`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false", input)
	}
}

func TestRefResolution_Simple(t *testing.T) {
	// A schema with one $ref pointing to a definition.
	schema := &JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"addr": {Ref: "#/definitions/Address"},
		},
		Required: []string{"addr"},
		Definitions: map[string]*JSONSchema{
			"Address": {
				Type: "object",
				Properties: map[string]*JSONSchema{
					"city": {Type: "string"},
				},
				Required: []string{"city"},
			},
		},
	}
	g, err := Convert(schema)
	if err != nil {
		t.Fatal(err)
	}

	input := `{"addr":{"city":"NYC"}}`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input)
	}
}

func TestRefResolution_Defs(t *testing.T) {
	// Same as Simple but using $defs instead of definitions.
	schema := &JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"color": {Ref: "#/$defs/Color"},
		},
		Required: []string{"color"},
		Defs: map[string]*JSONSchema{
			"Color": {Type: "string"},
		},
	}
	g, err := Convert(schema)
	if err != nil {
		t.Fatal(err)
	}

	input := `{"color":"red"}`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input)
	}
}

func TestRefResolution_Nested(t *testing.T) {
	// $ref chain: A refs B, B refs C.
	schema := &JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"val": {Ref: "#/definitions/A"},
		},
		Required: []string{"val"},
		Definitions: map[string]*JSONSchema{
			"A": {Ref: "#/definitions/B"},
			"B": {Ref: "#/definitions/C"},
			"C": {Type: "integer"},
		},
	}
	g, err := Convert(schema)
	if err != nil {
		t.Fatal(err)
	}

	input := `{"val":42}`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input)
	}
}

func TestRefResolution_Circular(t *testing.T) {
	// Circular $ref: A refs B, B refs A. Should not infinite loop.
	// At max depth, the reference becomes an empty schema (any JSON).
	schema := &JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"x": {Ref: "#/definitions/A"},
		},
		Required: []string{"x"},
		Definitions: map[string]*JSONSchema{
			"A": {
				Type: "object",
				Properties: map[string]*JSONSchema{
					"child": {Ref: "#/definitions/B"},
				},
			},
			"B": {
				Type: "object",
				Properties: map[string]*JSONSchema{
					"child": {Ref: "#/definitions/A"},
				},
			},
		},
	}
	g, err := Convert(schema)
	if err != nil {
		t.Fatal(err)
	}

	// At some depth the circular ref becomes any-JSON (empty schema).
	// A simple object should be accepted.
	input := `{"x":{}}`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input)
	}
}

func TestRefResolution_NotFound(t *testing.T) {
	// $ref to non-existent definition returns error.
	tests := []struct {
		name   string
		schema *JSONSchema
	}{
		{
			name: "missing definition",
			schema: &JSONSchema{
				Type: "object",
				Properties: map[string]*JSONSchema{
					"x": {Ref: "#/definitions/DoesNotExist"},
				},
			},
		},
		{
			name: "unsupported ref format",
			schema: &JSONSchema{
				Type: "object",
				Properties: map[string]*JSONSchema{
					"x": {Ref: "http://example.com/schema.json"},
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Convert(tc.schema)
			if err == nil {
				t.Fatal("expected error for unresolvable $ref")
			}
		})
	}
}

func TestRefResolution_InArray(t *testing.T) {
	// $ref used in array items.
	schema := &JSONSchema{
		Type: "array",
		Items: &JSONSchema{Ref: "#/definitions/Item"},
		Definitions: map[string]*JSONSchema{
			"Item": {Type: "integer"},
		},
	}
	g, err := Convert(schema)
	if err != nil {
		t.Fatal(err)
	}

	input := `[1,2,3]`
	final, ok := feedString(g, input)
	if !ok {
		t.Fatalf("feedString(%q): rejected", input)
	}
	if !final.IsComplete() {
		t.Fatalf("IsComplete(%q): got false, want true", input)
	}
}

func boolPtr(b bool) *bool {
	return &b
}
