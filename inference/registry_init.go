package inference

// This file registers all built-in architectures with the architecture registry.
// New architectures should add a RegisterArchitecture call here.
func init() {
	RegisterArchitecture("llama", buildLlamaGraph)
	RegisterArchitecture("gemma", buildGemmaGraph)
	RegisterArchitecture("gemma3", buildGemmaGraph)
	RegisterArchitecture("qwen2", buildQwenGraph)
	RegisterArchitecture("mistral", buildMistralGraph)
	RegisterArchitecture("phi", buildPhiGraph)
	RegisterArchitecture("phi3", buildPhiGraph)
	RegisterArchitecture("deepseek_v3", buildDeepSeekGraph)
	RegisterArchitecture("deepseek2", buildDeepSeekGraph)
	RegisterArchitecture("mamba", buildMambaGraph)
	RegisterArchitecture("mamba3", buildMamba3Graph)
	RegisterArchitecture("jamba", buildJambaGraph)
	RegisterArchitecture("whisper", buildWhisperGraph)
}
