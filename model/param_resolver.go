package model

import "regexp"

// ParamResolver maps architecture-specific parameter names to canonical names
// used by Zerfoo layers during model building. Canonical names follow the
// Llama/Gemma convention (the most common HuggingFace naming pattern).
type ParamResolver interface {
	// Resolve returns the canonical name for a model-specific parameter name.
	// Returns the input unchanged if no mapping applies.
	Resolve(name string) string
}

// NewParamResolver returns a resolver for the given architecture type.
// Architecture types match the model_type field from HuggingFace config.json.
func NewParamResolver(arch string) ParamResolver {
	switch arch {
	case "phi", "phi3":
		return newPhiResolver()
	default:
		// Llama, Gemma, Gemma2, Gemma3, Mistral, Qwen2, DeepSeek V3, and
		// unknown architectures all use canonical naming (no renaming needed).
		return &identityResolver{}
	}
}

// ResolveAll takes a resolver and a map keyed by model-specific names, and
// returns a new map containing both the original names and any canonical
// aliases produced by the resolver. This allows parameter lookups by either
// the original or canonical name.
func ResolveAll[V any](r ParamResolver, params map[string]V) map[string]V {
	result := make(map[string]V, len(params))
	for name, val := range params {
		result[name] = val
		canonical := r.Resolve(name)
		if canonical != name {
			result[canonical] = val
		}
	}
	return result
}

// identityResolver returns names unchanged. Used for architectures whose
// parameter names already follow the canonical convention.
type identityResolver struct{}

func (r *identityResolver) Resolve(name string) string { return name }

// regexResolver applies an ordered list of regex-based renaming rules.
// The first matching rule wins.
type regexResolver struct {
	rules []resolveRule
}

type resolveRule struct {
	pattern     *regexp.Regexp
	replacement string
}

func (r *regexResolver) Resolve(name string) string {
	for _, rule := range r.rules {
		if rule.pattern.MatchString(name) {
			return rule.pattern.ReplaceAllString(name, rule.replacement)
		}
	}
	return name
}

// newPhiResolver creates a resolver for Phi-family models.
// Phi uses "dense_proj" where other architectures use "o_proj" for the
// attention output projection.
func newPhiResolver() *regexResolver {
	return &regexResolver{
		rules: []resolveRule{
			{
				pattern:     regexp.MustCompile(`^(model\.layers\.\d+\.self_attn\.)dense(_proj\..+)$`),
				replacement: "${1}o${2}",
			},
		},
	}
}
