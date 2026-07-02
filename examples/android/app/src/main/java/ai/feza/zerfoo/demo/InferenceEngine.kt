package ai.feza.zerfoo.demo

import mobile.Mobile
import mobile.Engine
import mobile.GenerateConfig

/**
 * Wraps the gomobile [Engine] with lifecycle management.
 * Call [load] before [generate] or [tokenize], and [close] when done.
 */
class InferenceEngine {
    private var engine: Engine? = null

    val isLoaded: Boolean get() = engine != null

    /**
     * Load a GGUF model from [path]. Throws on failure.
     */
    fun load(path: String) {
        close()
        engine = Mobile.newEngine(path)
    }

    /**
     * Generate text from [prompt] with up to [maxTokens] tokens.
     */
    fun generate(prompt: String, maxTokens: Int = 256): String {
        val eng = engine ?: throw IllegalStateException("Model not loaded")
        return eng.generate(prompt, maxTokens.toLong())
    }

    /**
     * Generate text with explicit sampling configuration.
     */
    fun generateWithConfig(
        prompt: String,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        topK: Int = 40,
        maxTokens: Int = 256,
    ): String {
        val eng = engine ?: throw IllegalStateException("Model not loaded")
        val config = GenerateConfig().apply {
            this.temperature = temperature
            this.topP = topP
            this.topK = topK.toLong()
            this.maxTokens = maxTokens.toLong()
        }
        return eng.generateWithConfig(prompt, config)
    }

    /**
     * Tokenize [text] and return token IDs as a JSON array string.
     */
    fun tokenize(text: String): String {
        val eng = engine ?: throw IllegalStateException("Model not loaded")
        return eng.tokenize(text)
    }

    /**
     * Release all resources. Safe to call multiple times.
     */
    fun close() {
        engine?.close()
        engine = null
    }
}
