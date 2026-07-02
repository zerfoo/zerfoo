package ai.feza.zerfoo.demo

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private val engine = InferenceEngine()

    private lateinit var modelPathInput: TextInputEditText
    private lateinit var promptInput: TextInputEditText
    private lateinit var loadButton: MaterialButton
    private lateinit var generateButton: MaterialButton
    private lateinit var statusText: TextView
    private lateinit var outputText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelPathInput = findViewById(R.id.modelPathInput)
        promptInput = findViewById(R.id.promptInput)
        loadButton = findViewById(R.id.loadButton)
        generateButton = findViewById(R.id.generateButton)
        statusText = findViewById(R.id.statusText)
        outputText = findViewById(R.id.outputText)

        loadButton.setOnClickListener { onLoadModel() }
        generateButton.setOnClickListener { onGenerate() }
    }

    private fun onLoadModel() {
        val path = modelPathInput.text?.toString().orEmpty().trim()
        if (path.isEmpty()) {
            statusText.text = "Please enter a model path."
            return
        }

        setLoading(true)
        statusText.text = "Loading model..."
        outputText.text = ""

        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    engine.load(path)
                }
                statusText.text = "Model loaded."
                generateButton.isEnabled = true
            } catch (e: Exception) {
                statusText.text = "Load failed: ${e.message}"
                generateButton.isEnabled = false
            } finally {
                setLoading(false)
            }
        }
    }

    private fun onGenerate() {
        val prompt = promptInput.text?.toString().orEmpty().trim()
        if (prompt.isEmpty()) {
            statusText.text = "Please enter a prompt."
            return
        }

        setLoading(true)
        statusText.text = "Generating..."

        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    engine.generate(prompt, maxTokens = 256)
                }
                outputText.text = result
                statusText.text = "Done."
            } catch (e: Exception) {
                statusText.text = "Error: ${e.message}"
            } finally {
                setLoading(false)
            }
        }
    }

    private fun setLoading(loading: Boolean) {
        loadButton.isEnabled = !loading
        generateButton.isEnabled = !loading && engine.isLoaded
        promptInput.isEnabled = !loading
        modelPathInput.isEnabled = !loading
    }

    override fun onDestroy() {
        engine.close()
        super.onDestroy()
    }
}
