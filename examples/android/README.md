# Android Demo

A minimal Android app that demonstrates on-device ML inference using Zerfoo's gomobile bindings.

## Prerequisites

- Go 1.25+
- Android SDK (API 24+)
- Android NDK (r25+)
- [gomobile](https://pkg.go.dev/golang.org/x/mobile/cmd/gomobile)
- A GGUF model file small enough for on-device use (e.g., Gemma 3 1B Q4)

## Building the Go AAR

First, install gomobile and initialize it:

```bash
go install golang.org/x/mobile/cmd/gomobile@latest
go install golang.org/x/mobile/cmd/gobind@latest
gomobile init
```

Build the AAR from the mobile package:

```bash
cd $(go env GOPATH)/src/github.com/zerfoo/zerfoo
gomobile bind -target=android -androidapi=24 -o examples/android/app/libs/mobile.aar ./mobile/
```

This produces `mobile.aar` containing the Go runtime and the `mobile` package bindings for `arm64-v8a`, `armeabi-v7a`, and `x86_64`.

## Building the Android App

Open the `examples/android/` directory in Android Studio, or build from the command line:

```bash
cd examples/android
./gradlew assembleDebug
```

The APK will be at `app/build/outputs/apk/debug/app-debug.apk`.

## Installing and Running

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

Before launching, push a GGUF model to the device:

```bash
adb push path/to/model.gguf /sdcard/Download/model.gguf
```

Then open the "Zerfoo Demo" app. Enter the model path (`/sdcard/Download/model.gguf`) and a prompt, then tap **Generate**.

## Architecture

```
app/src/main/java/ai/feza/zerfoo/demo/
  MainActivity.kt    -- UI: model path input, prompt input, generate button, output display
  InferenceEngine.kt -- Wrapper around mobile.Engine with lifecycle management
```

The app uses the gomobile-generated `mobile.Engine` class:

1. `mobile.Mobile.newEngine(path)` loads a GGUF model
2. `engine.generate(prompt, maxTokens)` runs inference
3. `engine.tokenize(text)` returns token IDs as a JSON string
4. `engine.close()` releases resources

## Project Structure

```
examples/android/
  build.gradle.kts          -- Root Gradle build
  settings.gradle.kts       -- Project settings
  gradle.properties         -- Gradle/Android properties
  app/
    build.gradle.kts         -- App module build config
    src/main/
      AndroidManifest.xml
      java/ai/feza/zerfoo/demo/
        MainActivity.kt
        InferenceEngine.kt
      res/
        layout/activity_main.xml
        values/strings.xml
```
