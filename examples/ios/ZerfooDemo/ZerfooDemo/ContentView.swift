import SwiftUI
import Mobile

struct ContentView: View {
    @StateObject private var viewModel = InferenceViewModel()

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                Text("Zerfoo On-Device Inference")
                    .font(.headline)

                TextField("Enter prompt...", text: $viewModel.prompt)
                    .textFieldStyle(.roundedBorder)
                    .padding(.horizontal)

                HStack {
                    Text("Max Tokens: \(viewModel.maxTokens)")
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.maxTokens) },
                            set: { viewModel.maxTokens = Int($0) }
                        ),
                        in: 1...512,
                        step: 1
                    )
                }
                .padding(.horizontal)

                HStack(spacing: 12) {
                    Button("Generate") {
                        viewModel.generate()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isLoading || !viewModel.isModelLoaded)

                    Button("Tokenize") {
                        viewModel.tokenize()
                    }
                    .buttonStyle(.bordered)
                    .disabled(viewModel.isLoading || !viewModel.isModelLoaded)
                }

                if viewModel.isLoading {
                    ProgressView("Running inference...")
                }

                if let error = viewModel.errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                        .padding(.horizontal)
                }

                ScrollView {
                    Text(viewModel.output)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .background(Color(.systemGroupedBackground))
                .cornerRadius(8)
                .padding(.horizontal)

                Spacer()

                if !viewModel.isModelLoaded {
                    Button("Load Model") {
                        viewModel.loadModel()
                    }
                    .buttonStyle(.borderedProminent)

                    Text("Place a GGUF model in the app's Documents directory.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
            }
            .padding(.vertical)
            .navigationTitle("Zerfoo Demo")
        }
    }
}
