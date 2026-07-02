package grammar

import (
	"fmt"
	"sort"
	"strconv"
)

// maxRefDepth is the maximum recursion depth for $ref resolution. When this
// depth is exceeded (e.g. circular references), the reference is replaced with
// an empty schema that accepts any valid JSON value.
const maxRefDepth = 10

// Convert transforms a JSONSchema into a Grammar state machine. It returns an
// error if the schema uses unsupported features or contains unresolvable $refs.
func Convert(schema *JSONSchema) (*Grammar, error) {
	// Collect definitions from the root schema.
	defs := mergeDefinitions(schema.Definitions, schema.Defs)

	// Resolve all $ref pointers before validation.
	resolved, err := resolveRefs(schema, defs, 0)
	if err != nil {
		return nil, err
	}

	if err := rejectUnsupported(resolved); err != nil {
		return nil, err
	}
	n, err := buildNode(resolved)
	if err != nil {
		return nil, err
	}
	return &Grammar{node: n}, nil
}

// mergeDefinitions merges "definitions" and "$defs" maps, with $defs taking
// precedence on conflicts.
func mergeDefinitions(definitions, defs map[string]*JSONSchema) map[string]*JSONSchema {
	if len(definitions) == 0 && len(defs) == 0 {
		return nil
	}
	merged := make(map[string]*JSONSchema, len(definitions)+len(defs))
	for k, v := range definitions {
		merged[k] = v
	}
	for k, v := range defs {
		merged[k] = v
	}
	return merged
}

// resolveRefs recursively inlines $ref pointers. When depth exceeds
// maxRefDepth the reference is replaced with an empty schema (any JSON value).
func resolveRefs(s *JSONSchema, defs map[string]*JSONSchema, depth int) (*JSONSchema, error) {
	if s == nil {
		return nil, nil
	}

	// Handle $ref.
	if s.Ref != "" {
		if depth >= maxRefDepth {
			// Circular reference protection — emit any-type schema.
			return &JSONSchema{}, nil
		}
		name, err := parseLocalRef(s.Ref)
		if err != nil {
			return nil, err
		}
		target, ok := defs[name]
		if !ok {
			return nil, fmt.Errorf("unresolved $ref: %q (definition %q not found)", s.Ref, name)
		}
		// Resolve the target recursively (it may contain further $refs).
		return resolveRefs(target, defs, depth+1)
	}

	// Deep-copy only the fields that may contain nested schemas.
	out := *s // shallow copy
	out.Definitions = nil
	out.Defs = nil

	if len(s.Properties) > 0 {
		out.Properties = make(map[string]*JSONSchema, len(s.Properties))
		for k, v := range s.Properties {
			resolved, err := resolveRefs(v, defs, depth)
			if err != nil {
				return nil, err
			}
			out.Properties[k] = resolved
		}
	}
	if s.Items != nil {
		resolved, err := resolveRefs(s.Items, defs, depth)
		if err != nil {
			return nil, err
		}
		out.Items = resolved
	}
	return &out, nil
}

// parseLocalRef extracts the definition name from a local $ref pointer.
// Supported formats:
//
//	"#/definitions/Foo" → "Foo"
//	"#/$defs/Foo"       → "Foo"
func parseLocalRef(ref string) (string, error) {
	for _, prefix := range []string{"#/definitions/", "#/$defs/"} {
		if len(ref) > len(prefix) && ref[:len(prefix)] == prefix {
			return ref[len(prefix):], nil
		}
	}
	return "", fmt.Errorf("unsupported $ref format: %q (only local #/definitions/ and #/$defs/ are supported)", ref)
}

func rejectUnsupported(s *JSONSchema) error {
	if s == nil {
		return nil
	}
	if s.Ref != "" {
		// This should not happen — refs are resolved before rejectUnsupported
		// is called. If we get here, it means resolveRefs missed something.
		return fmt.Errorf("unresolved $ref: %q", s.Ref)
	}
	if len(s.OneOf) > 0 {
		return fmt.Errorf("unsupported JSON Schema feature: oneOf")
	}
	if len(s.AnyOf) > 0 {
		return fmt.Errorf("unsupported JSON Schema feature: anyOf")
	}
	if len(s.AllOf) > 0 {
		return fmt.Errorf("unsupported JSON Schema feature: allOf")
	}
	if s.Pattern != "" {
		return fmt.Errorf("unsupported JSON Schema feature: pattern")
	}
	if s.AdditionalProperties != nil {
		return fmt.Errorf("unsupported JSON Schema feature: additionalProperties")
	}
	// Recurse into nested schemas.
	for _, p := range s.Properties {
		if err := rejectUnsupported(p); err != nil {
			return err
		}
	}
	if s.Items != nil {
		if err := rejectUnsupported(s.Items); err != nil {
			return err
		}
	}
	return nil
}

func buildNode(s *JSONSchema) (node, error) {
	// Handle enum/const first — they override type.
	if s.Const != nil {
		return buildConstNode(s.Const)
	}
	if len(s.Enum) > 0 {
		return buildEnumNode(s.Enum)
	}

	switch s.Type {
	case "object":
		return buildObjectNode(s)
	case "array":
		return buildArrayNode(s)
	case "string":
		return buildStringNode(s)
	case "number":
		return &numberNode{state: numStart}, nil
	case "integer":
		return &integerNode{state: intStart}, nil
	case "boolean":
		return buildLiteralAlternation([]string{"true", "false"})
	case "null":
		return &literalNode{remaining: "null"}, nil
	case "":
		// Empty schema — accept any valid JSON value.
		return &anyJSONNode{state: anyStart}, nil
	default:
		return nil, fmt.Errorf("unsupported JSON Schema type: %q", s.Type)
	}
}

// ---------------------------------------------------------------------------
// literalNode — matches a fixed byte sequence (e.g. "null", "true", "false")
// ---------------------------------------------------------------------------

type literalNode struct {
	remaining string // bytes still to consume
}

func (n *literalNode) advance(b byte) (node, bool) {
	if n.remaining == "" || b != n.remaining[0] {
		return nil, false
	}
	return &literalNode{remaining: n.remaining[1:]}, true
}

func (n *literalNode) validBytes() []byte {
	if n.remaining == "" {
		return nil
	}
	return []byte{n.remaining[0]}
}

func (n *literalNode) isComplete() bool {
	return n.remaining == ""
}

// ---------------------------------------------------------------------------
// alternationNode — tries multiple sub-nodes; valid bytes are the union.
// ---------------------------------------------------------------------------

type alternationNode struct {
	options []node
}

func buildLiteralAlternation(literals []string) (node, error) {
	opts := make([]node, len(literals))
	for i, l := range literals {
		opts[i] = &literalNode{remaining: l}
	}
	return &alternationNode{options: opts}, nil
}

func (n *alternationNode) advance(b byte) (node, bool) {
	var next []node
	for _, opt := range n.options {
		if nn, ok := opt.advance(b); ok {
			next = append(next, nn)
		}
	}
	if len(next) == 0 {
		return nil, false
	}
	if len(next) == 1 {
		return next[0], true
	}
	return &alternationNode{options: next}, true
}

func (n *alternationNode) validBytes() []byte {
	seen := make(map[byte]bool)
	for _, opt := range n.options {
		for _, b := range opt.validBytes() {
			seen[b] = true
		}
	}
	out := make([]byte, 0, len(seen))
	for b := range seen {
		out = append(out, b)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func (n *alternationNode) isComplete() bool {
	for _, opt := range n.options {
		if opt.isComplete() {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// constNode / enumNode — exact JSON literal(s)
// ---------------------------------------------------------------------------

func buildConstNode(v any) (node, error) {
	s, err := jsonLiteral(v)
	if err != nil {
		return nil, err
	}
	return &literalNode{remaining: s}, nil
}

func buildEnumNode(values []any) (node, error) {
	opts := make([]node, len(values))
	for i, v := range values {
		s, err := jsonLiteral(v)
		if err != nil {
			return nil, err
		}
		opts[i] = &literalNode{remaining: s}
	}
	if len(opts) == 1 {
		return opts[0], nil
	}
	return &alternationNode{options: opts}, nil
}

func jsonLiteral(v any) (string, error) {
	switch val := v.(type) {
	case string:
		return strconv.Quote(val), nil
	case float64:
		if val == float64(int64(val)) {
			return strconv.FormatInt(int64(val), 10), nil
		}
		return strconv.FormatFloat(val, 'f', -1, 64), nil
	case int:
		return strconv.Itoa(val), nil
	case int64:
		return strconv.FormatInt(val, 10), nil
	case bool:
		if val {
			return "true", nil
		}
		return "false", nil
	case nil:
		return "null", nil
	default:
		return "", fmt.Errorf("unsupported enum/const value type %T", v)
	}
}

// ---------------------------------------------------------------------------
// stringNode — matches a JSON string with optional min/maxLength constraints.
// Tracks character count (not byte count) for length validation.
// ---------------------------------------------------------------------------

type stringPhase int

const (
	strWantOpen stringPhase = iota
	strBody
	strEscape
	strDone
)

type stringNode struct {
	phase     stringPhase
	charCount int // number of characters consumed so far (not bytes)
	minLen    int
	maxLen    int // 0 = unlimited
}

func buildStringNode(s *JSONSchema) (node, error) {
	return &stringNode{
		phase:  strWantOpen,
		minLen: s.MinLength,
		maxLen: s.MaxLength,
	}, nil
}

func (n *stringNode) advance(b byte) (node, bool) {
	switch n.phase {
	case strWantOpen:
		if b == '"' {
			return &stringNode{phase: strBody, minLen: n.minLen, maxLen: n.maxLen}, true
		}
		return nil, false

	case strBody:
		if b == '\\' {
			return &stringNode{phase: strEscape, charCount: n.charCount, minLen: n.minLen, maxLen: n.maxLen}, true
		}
		if b == '"' {
			if n.charCount < n.minLen {
				return nil, false
			}
			return &stringNode{phase: strDone, charCount: n.charCount, minLen: n.minLen, maxLen: n.maxLen}, true
		}
		newCount := n.charCount + 1
		if n.maxLen > 0 && newCount > n.maxLen {
			return nil, false
		}
		return &stringNode{phase: strBody, charCount: newCount, minLen: n.minLen, maxLen: n.maxLen}, true

	case strEscape:
		// Accept any single escape character after backslash.
		newCount := n.charCount + 1
		if n.maxLen > 0 && newCount > n.maxLen {
			return nil, false
		}
		return &stringNode{phase: strBody, charCount: newCount, minLen: n.minLen, maxLen: n.maxLen}, true

	case strDone:
		return nil, false
	}
	return nil, false
}

func (n *stringNode) validBytes() []byte {
	switch n.phase {
	case strWantOpen:
		return []byte{'"'}
	case strBody:
		var out []byte
		// Close quote if minLen satisfied.
		if n.charCount >= n.minLen {
			out = append(out, '"')
		}
		// Any printable ASCII except control chars, if maxLen not exceeded.
		if n.maxLen == 0 || n.charCount < n.maxLen {
			out = append(out, '\\')
			for b := byte(0x20); b < 0x7f; b++ {
				if b != '"' && b != '\\' {
					out = append(out, b)
				}
			}
		}
		return out
	case strEscape:
		// Simplified: accept common JSON escape chars.
		return []byte{'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
	case strDone:
		return nil
	}
	return nil
}

func (n *stringNode) isComplete() bool {
	return n.phase == strDone
}

// ---------------------------------------------------------------------------
// numberNode — matches a JSON number: [-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?
// ---------------------------------------------------------------------------

type numState int

const (
	numStart numState = iota
	numAfterMinus
	numLeadZero
	numIntDigits
	numDot
	numFracDigits
	numExpMark
	numExpSign
	numExpDigits
	numDone // virtual — numbers are complete when no more digits come
)

type numberNode struct {
	state numState
}

func (n *numberNode) advance(b byte) (node, bool) {
	switch n.state {
	case numStart:
		if b == '-' {
			return &numberNode{numAfterMinus}, true
		}
		if b == '0' {
			return &numberNode{numLeadZero}, true
		}
		if b >= '1' && b <= '9' {
			return &numberNode{numIntDigits}, true
		}
	case numAfterMinus:
		if b == '0' {
			return &numberNode{numLeadZero}, true
		}
		if b >= '1' && b <= '9' {
			return &numberNode{numIntDigits}, true
		}
	case numLeadZero:
		if b == '.' {
			return &numberNode{numDot}, true
		}
		if b == 'e' || b == 'E' {
			return &numberNode{numExpMark}, true
		}
	case numIntDigits:
		if b >= '0' && b <= '9' {
			return &numberNode{numIntDigits}, true
		}
		if b == '.' {
			return &numberNode{numDot}, true
		}
		if b == 'e' || b == 'E' {
			return &numberNode{numExpMark}, true
		}
	case numDot:
		if b >= '0' && b <= '9' {
			return &numberNode{numFracDigits}, true
		}
	case numFracDigits:
		if b >= '0' && b <= '9' {
			return &numberNode{numFracDigits}, true
		}
		if b == 'e' || b == 'E' {
			return &numberNode{numExpMark}, true
		}
	case numExpMark:
		if b == '+' || b == '-' {
			return &numberNode{numExpSign}, true
		}
		if b >= '0' && b <= '9' {
			return &numberNode{numExpDigits}, true
		}
	case numExpSign:
		if b >= '0' && b <= '9' {
			return &numberNode{numExpDigits}, true
		}
	case numExpDigits:
		if b >= '0' && b <= '9' {
			return &numberNode{numExpDigits}, true
		}
	}
	return nil, false
}

func (n *numberNode) validBytes() []byte {
	switch n.state {
	case numStart:
		return append([]byte{'-', '0'}, digitRange('1', '9')...)
	case numAfterMinus:
		return append([]byte{'0'}, digitRange('1', '9')...)
	case numLeadZero:
		return []byte{'.', 'e', 'E'}
	case numIntDigits:
		return append([]byte{'.', 'e', 'E'}, digitRange('0', '9')...)
	case numDot:
		return digitRange('0', '9')
	case numFracDigits:
		return append([]byte{'e', 'E'}, digitRange('0', '9')...)
	case numExpMark:
		return append([]byte{'+', '-'}, digitRange('0', '9')...)
	case numExpSign:
		return digitRange('0', '9')
	case numExpDigits:
		return digitRange('0', '9')
	}
	return nil
}

func (n *numberNode) isComplete() bool {
	switch n.state {
	case numLeadZero, numIntDigits, numFracDigits, numExpDigits:
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// integerNode — matches a JSON integer: [-]?[0-9]+
// ---------------------------------------------------------------------------

type intState int

const (
	intStart intState = iota
	intAfterMinus
	intLeadZero
	intDigits
)

type integerNode struct {
	state intState
}

func (n *integerNode) advance(b byte) (node, bool) {
	switch n.state {
	case intStart:
		if b == '-' {
			return &integerNode{intAfterMinus}, true
		}
		if b == '0' {
			return &integerNode{intLeadZero}, true
		}
		if b >= '1' && b <= '9' {
			return &integerNode{intDigits}, true
		}
	case intAfterMinus:
		if b == '0' {
			return &integerNode{intLeadZero}, true
		}
		if b >= '1' && b <= '9' {
			return &integerNode{intDigits}, true
		}
	case intLeadZero:
		// No more digits after leading zero.
	case intDigits:
		if b >= '0' && b <= '9' {
			return &integerNode{intDigits}, true
		}
	}
	return nil, false
}

func (n *integerNode) validBytes() []byte {
	switch n.state {
	case intStart:
		return append([]byte{'-', '0'}, digitRange('1', '9')...)
	case intAfterMinus:
		return append([]byte{'0'}, digitRange('1', '9')...)
	case intLeadZero:
		return nil
	case intDigits:
		return digitRange('0', '9')
	}
	return nil
}

func (n *integerNode) isComplete() bool {
	return n.state == intLeadZero || n.state == intDigits
}

func digitRange(lo, hi byte) []byte {
	out := make([]byte, 0, hi-lo+1)
	for b := lo; b <= hi; b++ {
		out = append(out, b)
	}
	return out
}

// ---------------------------------------------------------------------------
// objectNode — matches a JSON object with known properties.
// ---------------------------------------------------------------------------

type objPhase int

const (
	objWantOpen    objPhase = iota // expect {
	objWantKeyOrClose             // expect " (key) or }
	objInKey                      // consuming a key literal
	objWantColon                  // expect :
	objInValue                    // consuming the value
	objWantCommaOrClose           // expect , or }
	objDone
)

type objectNode struct {
	phase    objPhase
	keys     []string        // sorted property names
	schemas  map[string]*JSONSchema
	required map[string]bool
	seen     map[string]bool // properties already emitted
	curKey   string          // the key currently being parsed
	keyNode  node            // sub-node for consuming the key literal
	valNode  node            // sub-node for consuming the value
}

func buildObjectNode(s *JSONSchema) (node, error) {
	keys := make([]string, 0, len(s.Properties))
	for k := range s.Properties {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	req := make(map[string]bool, len(s.Required))
	for _, r := range s.Required {
		req[r] = true
	}

	return &objectNode{
		phase:    objWantOpen,
		keys:     keys,
		schemas:  s.Properties,
		required: req,
		seen:     make(map[string]bool),
	}, nil
}

func (n *objectNode) clone() *objectNode {
	seen := make(map[string]bool, len(n.seen))
	for k, v := range n.seen {
		seen[k] = v
	}
	return &objectNode{
		phase:    n.phase,
		keys:     n.keys,
		schemas:  n.schemas,
		required: n.required,
		seen:     seen,
		curKey:   n.curKey,
		keyNode:  n.keyNode,
		valNode:  n.valNode,
	}
}

func (n *objectNode) advance(b byte) (node, bool) {
	switch n.phase {
	case objWantOpen:
		if b == '{' {
			c := n.clone()
			c.phase = objWantKeyOrClose
			return c, true
		}

	case objWantKeyOrClose:
		if b == '}' {
			// Closing — check all required keys are present.
			for r := range n.required {
				if !n.seen[r] {
					return nil, false
				}
			}
			c := n.clone()
			c.phase = objDone
			return c, true
		}
		// Must be start of a key — try each unseen key.
		return n.tryStartKey(b)

	case objInKey:
		next, ok := n.keyNode.advance(b)
		if !ok {
			return nil, false
		}
		c := n.clone()
		c.keyNode = next
		if next.isComplete() {
			c.phase = objWantColon
		}
		return c, true

	case objWantColon:
		if b == ':' {
			valSchema := n.schemas[n.curKey]
			valN, err := buildNode(valSchema)
			if err != nil {
				return nil, false
			}
			c := n.clone()
			c.phase = objInValue
			c.valNode = valN
			return c, true
		}

	case objInValue:
		next, ok := n.valNode.advance(b)
		if !ok {
			return nil, false
		}
		c := n.clone()
		c.valNode = next
		if next.isComplete() {
			c.phase = objWantCommaOrClose
			c.seen[n.curKey] = true
		}
		return c, true

	case objWantCommaOrClose:
		// The value node may still accept more bytes (e.g. more digits in a number).
		// Try continuing the value first.
		if n.valNode != nil {
			if next, ok := n.valNode.advance(b); ok {
				c := n.clone()
				c.valNode = next
				if !next.isComplete() {
					// Value is no longer complete; we must keep consuming it.
					c.phase = objInValue
				}
				return c, true
			}
		}
		if b == ',' {
			// Only accept comma if there are unseen properties to emit.
			hasUnseen := false
			for _, k := range n.keys {
				if !n.seen[k] {
					hasUnseen = true
					break
				}
			}
			if !hasUnseen {
				return nil, false
			}
			c := n.clone()
			c.phase = objWantKeyOrClose
			return c, true
		}
		if b == '}' {
			for r := range n.required {
				if !n.seen[r] {
					return nil, false
				}
			}
			c := n.clone()
			c.phase = objDone
			return c, true
		}
	}
	return nil, false
}

func (n *objectNode) tryStartKey(b byte) (node, bool) {
	// Each key is a JSON string literal — build alternation of unseen keys.
	var opts []node
	for _, k := range n.keys {
		if n.seen[k] {
			continue
		}
		lit := &literalNode{remaining: strconv.Quote(k)}
		if nn, ok := lit.advance(b); ok {
			opts = append(opts, &objectKeyWrap{
				parent:  n,
				key:     k,
				keyNode: nn,
			})
		}
	}
	if len(opts) == 0 {
		return nil, false
	}
	if len(opts) == 1 {
		return opts[0], true
	}
	return &alternationNode{options: opts}, true
}

// objectKeyWrap wraps key parsing so that when the key completes, we transition
// back into the objectNode.
type objectKeyWrap struct {
	parent  *objectNode
	key     string
	keyNode node
}

func (w *objectKeyWrap) advance(b byte) (node, bool) {
	next, ok := w.keyNode.advance(b)
	if !ok {
		return nil, false
	}
	if next.isComplete() {
		c := w.parent.clone()
		c.phase = objWantColon
		c.curKey = w.key
		c.keyNode = next
		return c, true
	}
	return &objectKeyWrap{parent: w.parent, key: w.key, keyNode: next}, true
}

func (w *objectKeyWrap) validBytes() []byte {
	return w.keyNode.validBytes()
}

func (w *objectKeyWrap) isComplete() bool {
	return false // key wrap is never a final state
}

func (n *objectNode) validBytes() []byte {
	switch n.phase {
	case objWantOpen:
		return []byte{'{'}
	case objWantKeyOrClose:
		var out []byte
		// Can close if all required are seen.
		allReqSeen := true
		for r := range n.required {
			if !n.seen[r] {
				allReqSeen = false
				break
			}
		}
		if allReqSeen {
			out = append(out, '}')
		}
		// Start of key — must be '"'.
		hasUnseen := false
		for _, k := range n.keys {
			if !n.seen[k] {
				hasUnseen = true
				break
			}
		}
		if hasUnseen {
			out = append(out, '"')
		}
		return out
	case objInKey:
		return n.keyNode.validBytes()
	case objWantColon:
		return []byte{':'}
	case objInValue:
		return n.valNode.validBytes()
	case objWantCommaOrClose:
		var out []byte
		// Value may still accept more bytes.
		if n.valNode != nil {
			out = append(out, n.valNode.validBytes()...)
		}
		allReqSeen := true
		for r := range n.required {
			if !n.seen[r] {
				allReqSeen = false
				break
			}
		}
		hasUnseen := false
		for _, k := range n.keys {
			if !n.seen[k] {
				hasUnseen = true
				break
			}
		}
		if hasUnseen {
			out = append(out, ',')
		}
		if allReqSeen {
			out = append(out, '}')
		}
		return out
	case objDone:
		return nil
	}
	return nil
}

func (n *objectNode) isComplete() bool {
	return n.phase == objDone
}

// ---------------------------------------------------------------------------
// arrayNode — matches a JSON array with items constrained by a sub-schema.
// ---------------------------------------------------------------------------

type arrPhase int

const (
	arrWantOpen         arrPhase = iota // expect [
	arrWantItemOrClose                  // expect value or ]
	arrInItem                           // consuming an item
	arrWantCommaOrClose                 // expect , or ]
	arrDone
)

type arrayNode struct {
	phase      arrPhase
	itemSchema *JSONSchema // nil = any JSON value
	itemNode   node
}

func buildArrayNode(s *JSONSchema) (node, error) {
	return &arrayNode{
		phase:      arrWantOpen,
		itemSchema: s.Items,
	}, nil
}

func (n *arrayNode) advance(b byte) (node, bool) {
	switch n.phase {
	case arrWantOpen:
		if b == '[' {
			return &arrayNode{phase: arrWantItemOrClose, itemSchema: n.itemSchema}, true
		}

	case arrWantItemOrClose:
		if b == ']' {
			return &arrayNode{phase: arrDone, itemSchema: n.itemSchema}, true
		}
		return n.tryStartItem(b)

	case arrInItem:
		next, ok := n.itemNode.advance(b)
		if !ok {
			return nil, false
		}
		a := &arrayNode{phase: arrInItem, itemSchema: n.itemSchema, itemNode: next}
		if next.isComplete() {
			a.phase = arrWantCommaOrClose
		}
		return a, true

	case arrWantCommaOrClose:
		// Value may still accept more bytes.
		if n.itemNode != nil {
			if next, ok := n.itemNode.advance(b); ok {
				a := &arrayNode{phase: arrWantCommaOrClose, itemSchema: n.itemSchema, itemNode: next}
				if !next.isComplete() {
					a.phase = arrInItem
				}
				return a, true
			}
		}
		if b == ',' {
			return &arrayNode{phase: arrWantItemOrClose, itemSchema: n.itemSchema}, true
		}
		if b == ']' {
			return &arrayNode{phase: arrDone, itemSchema: n.itemSchema}, true
		}
	}
	return nil, false
}

func (n *arrayNode) tryStartItem(b byte) (node, bool) {
	schema := n.itemSchema
	if schema == nil {
		schema = &JSONSchema{} // any JSON
	}
	itemN, err := buildNode(schema)
	if err != nil {
		return nil, false
	}
	next, ok := itemN.advance(b)
	if !ok {
		return nil, false
	}
	a := &arrayNode{phase: arrInItem, itemSchema: n.itemSchema, itemNode: next}
	if next.isComplete() {
		a.phase = arrWantCommaOrClose
	}
	return a, true
}

func (n *arrayNode) validBytes() []byte {
	switch n.phase {
	case arrWantOpen:
		return []byte{'['}
	case arrWantItemOrClose:
		out := []byte{']'}
		// Figure out what bytes can start an item.
		schema := n.itemSchema
		if schema == nil {
			schema = &JSONSchema{}
		}
		itemN, err := buildNode(schema)
		if err == nil {
			out = append(out, itemN.validBytes()...)
		}
		return out
	case arrInItem:
		return n.itemNode.validBytes()
	case arrWantCommaOrClose:
		out := []byte{',', ']'}
		if n.itemNode != nil {
			out = append(out, n.itemNode.validBytes()...)
		}
		return out
	case arrDone:
		return nil
	}
	return nil
}

func (n *arrayNode) isComplete() bool {
	return n.phase == arrDone
}

// ---------------------------------------------------------------------------
// anyJSONNode — accepts any valid JSON value (for empty schemas).
// This is a simplified version that delegates to sub-nodes.
// ---------------------------------------------------------------------------

type anyState int

const (
	anyStart anyState = iota
	anyInSub
)

type anyJSONNode struct {
	state anyState
	sub   node
}

func (n *anyJSONNode) advance(b byte) (node, bool) {
	if n.state == anyInSub {
		return n.sub.advance(b)
	}
	// Try each possible JSON value start.
	candidates := []node{
		&literalNode{remaining: "null"},
		&literalNode{remaining: "true"},
		&literalNode{remaining: "false"},
		&numberNode{state: numStart},
		&stringNode{phase: strWantOpen},
	}
	// Also try object and array.
	objN, _ := buildObjectNode(&JSONSchema{Properties: map[string]*JSONSchema{}})
	arrN, _ := buildArrayNode(&JSONSchema{})
	candidates = append(candidates, objN, arrN)

	var matched []node
	for _, c := range candidates {
		if next, ok := c.advance(b); ok {
			matched = append(matched, next)
		}
	}
	if len(matched) == 0 {
		return nil, false
	}
	if len(matched) == 1 {
		return matched[0], true
	}
	return &alternationNode{options: matched}, true
}

func (n *anyJSONNode) validBytes() []byte {
	if n.state == anyInSub {
		return n.sub.validBytes()
	}
	// Any JSON value can start with: { [ " - 0-9 t f n
	out := []byte{'{', '[', '"', '-', 't', 'f', 'n'}
	out = append(out, digitRange('0', '9')...)
	return out
}

func (n *anyJSONNode) isComplete() bool {
	if n.state == anyInSub {
		return n.sub.isComplete()
	}
	return false
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

