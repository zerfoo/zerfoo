import Foundation
import Mobile

/// ViewModel that wraps the gomobile Engine for on-device inference.
@MainActor
class InferenceViewModel: ObservableObject {
    @Published var prompt: String = "Once upon a time"
    @Published var maxTokens: Int = 64
    @Published var output: String = ""
    @Published var isLoading: Bool = false
    @Published var isModelLoaded: Bool = false
    @Published var errorMessage: String?

    private var engine: MobileEngine?

    /// Searches the app's Documents directory for the first .gguf file and loads it.
    func loadModel() {
        isLoading = true
        errorMessage = nil

        Task.detached { [weak self] in
            guard let self else { return }

            let documentsURL = FileManager.default.urls(
                for: .documentDirectory, in: .userDomainMask
            ).first!

            // Find first .gguf file in Documents.
            let contents = (try? FileManager.default.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: nil
            )) ?? []

            guard let modelURL = contents.first(where: { $0.pathExtension == "gguf" }) else {
                await MainActor.run {
                    self.errorMessage = "No .gguf model found in Documents directory."
                    self.isLoading = false
                }
                return
            }

            var error: NSError?
            let eng = MobileNewEngine(modelURL.path, &error)

            await MainActor.run {
                if let error {
                    self.errorMessage = "Failed to load model: \(error.localizedDescription)"
                } else {
                    self.engine = eng
                    self.isModelLoaded = true
                    self.output = "Model loaded: \(modelURL.lastPathComponent)"
                }
                self.isLoading = false
            }
        }
    }

    /// Runs text generation with the current prompt and maxTokens.
    func generate() {
        guard let engine else { return }
        isLoading = true
        errorMessage = nil

        let currentPrompt = prompt
        let currentMaxTokens = maxTokens

        Task.detached { [weak self] in
            guard let self else { return }

            var error: NSError?
            let result = engine.generate(currentPrompt, maxTokens: currentMaxTokens, error: &error)

            await MainActor.run {
                if let error {
                    self.errorMessage = "Generation failed: \(error.localizedDescription)"
                } else {
                    self.output = result
                }
                self.isLoading = false
            }
        }
    }

    /// Tokenizes the current prompt and displays the token IDs.
    func tokenize() {
        guard let engine else { return }
        isLoading = true
        errorMessage = nil

        let currentPrompt = prompt

        Task.detached { [weak self] in
            guard let self else { return }

            var error: NSError?
            let result = engine.tokenize(currentPrompt, error: &error)

            await MainActor.run {
                if let error {
                    self.errorMessage = "Tokenization failed: \(error.localizedDescription)"
                } else {
                    self.output = "Token IDs: \(result)"
                }
                self.isLoading = false
            }
        }
    }

    deinit {
        try? engine?.close()
    }
}
