package dsl

// DefinitionSchema describes the serialized architecture shape. The shared
// component descriptors provide port/config constraints; Validate enforces them.
func DefinitionSchema() map[string]any {
	str := map[string]any{"type": "string"}
	array := func(items any) map[string]any { return map[string]any{"type": "array", "items": items} }
	object := func(properties map[string]any, required ...string) map[string]any {
		return map[string]any{"type": "object", "properties": properties, "required": required, "additionalProperties": false}
	}
	ref := object(map[string]any{"node": str, "port": str}, "node", "port")
	tensor := map[string]any{"name": str, "dtype": map[string]any{"const": "float32"}, "shape": array(map[string]any{"type": "integer"})}
	parameter := map[string]any{"name": str, "dtype": map[string]any{"const": "float32"}, "shape": array(map[string]any{"type": "integer"}), "initializer": map[string]any{"enum": []string{"zeros", "he_normal"}}}
	node := object(map[string]any{"name": str,
		"operator": str,
		"version":  map[string]any{"const": 1},
		"inputs": map[string]any{"type": "object",
			"additionalProperties": ref},
		"parameters": map[string]any{"type": "object",
			"additionalProperties": str},
		"attributes": map[string]any{"type": "object"}},
		"name",
		"operator",
		"version",
		"inputs")
	return object(map[string]any{"version": map[string]any{"const": 1},
		"name": str,
		"inputs": array(object(tensor,
			"name",
			"dtype",
			"shape")),
		"parameters": array(object(parameter,
			"name",
			"dtype",
			"shape",
			"initializer")),
		"nodes": array(node),
		"outputs": array(object(map[string]any{"name": str,
			"source": ref},
			"name",
			"source"))},
		"version",
		"name",
		"inputs",
		"parameters",
		"nodes",
		"outputs")
}
