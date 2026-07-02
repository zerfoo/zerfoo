# Zerfoo iOS Demo

A minimal iOS app demonstrating on-device ML inference using Zerfoo's gomobile bindings.

## Prerequisites

- macOS with Xcode 15+ installed
- Go 1.25+
- [gomobile](https://pkg.go.dev/golang.org/x/mobile/cmd/gomobile) installed and initialized:

```bash
go install golang.org/x/mobile/cmd/gomobile@latest
gomobile init
```

## Building the Mobile Framework

Generate the `Mobile.xcframework` from the Go `mobile/` package:

```bash
cd $(git rev-parse --show-toplevel)
gomobile bind -target=ios -o examples/ios/Mobile.xcframework ./mobile/
```

This produces `examples/ios/Mobile.xcframework` containing the compiled Go
library for `arm64` (device) and `amd64`/`arm64` (simulator).

## Opening the Project

1. Open `ZerfooDemo/` in Xcode (File > Open > select the `ZerfooDemo` folder).
2. Xcode will create a project from the Swift package structure, or you can create
   a new Xcode project and add the existing Swift files.
3. Add `Mobile.xcframework` to the project:
   - Select the project in the navigator.
   - Go to the target's **General** tab.
   - Under **Frameworks, Libraries, and Embedded Content**, click **+**.
   - Choose **Add Other... > Add Files...** and select `Mobile.xcframework`.
   - Set embed mode to **Embed & Sign**.

## Running the App

1. Connect an iOS device or select a simulator.
2. Copy a GGUF model file (e.g., a small Gemma or Phi model) to the app's
   Documents directory. On simulator, you can use:
   ```bash
   xcrun simctl get_app_container booted com.zerfoo.demo data
   # Copy model into the returned Documents/ path
   ```
3. Build and run (`Cmd+R`).
4. Tap **Load Model** to load the GGUF file from Documents.
5. Enter a prompt and tap **Generate** or **Tokenize**.

## Architecture

```
ZerfooDemo/
  ZerfooDemoApp.swift       # App entry point
  ContentView.swift         # Main UI with prompt input and output display
  InferenceViewModel.swift  # Wraps Mobile.Engine for async inference
```

The app uses SwiftUI and calls the Go `mobile.Engine` API via the generated
Objective-C bridge in `Mobile.xcframework`:

- `MobileNewEngine(path)` — loads a GGUF model
- `engine.generate(prompt, maxTokens:)` — runs text generation
- `engine.tokenize(text)` — returns token IDs as a JSON array string
- `engine.close()` — releases resources

## Model Recommendations

For on-device inference, use small quantized models:

| Model | Size | Notes |
|-------|------|-------|
| Gemma 3 1B Q4_K_M | ~700 MB | Best quality/size ratio |
| Phi 3.5 Mini Q4_0 | ~2 GB | Good for code tasks |
| Qwen 2.5 0.5B Q8_0 | ~500 MB | Smallest option |

## Troubleshooting

- **"No .gguf model found"**: Ensure a `.gguf` file is in the app's Documents directory.
- **Build errors with Mobile.xcframework**: Re-run `gomobile bind` and verify the
  framework is properly linked in Xcode.
- **Slow generation**: Use smaller quantized models (Q4_K_M or Q4_0). On-device
  inference is CPU-bound on iOS.
