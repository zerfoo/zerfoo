package grammar_test

import (
	"fmt"

	"github.com/zerfoo/zerfoo/generate/grammar"
)

func ExampleConvert() {
	schema := &grammar.JSONSchema{
		Type: "object",
		Properties: map[string]*grammar.JSONSchema{
			"name": {Type: "string"},
			"age":  {Type: "number"},
		},
		Required: []string{"name", "age"},
	}

	g, err := grammar.Convert(schema)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Advance through a valid JSON string character by character.
	input := `{"name":"Alice","age":30}`
	current := g
	ok := true
	for _, b := range []byte(input) {
		current, ok = current.Advance(b)
		if !ok {
			fmt.Printf("rejected at byte %q\n", string(b))
			return
		}
	}
	fmt.Println(current.IsComplete())
	// Output: true
}
