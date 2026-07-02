# Voxtral Transcribe 2 Support

## Context

Voxtral is Mistral AI's speech-to-text model family. Unlike Whisper (encoder-only
in Zerfoo), Voxtral uses an encoder-projector-decoder architecture:

```
Audio -> Mel Spectrogram (128 bins) -> Whisper-large-v3 Encoder -> MLP Adapter (4x downsample) -> Mistral Decoder -> Text
```

### Models

| Model | Params | GGUF | License | Status |
|-------|--------|------|---------|--------|
| Voxtral Mini 3B | 4.7B | ggml-org/Voxtral-Mini-3B-2507-GGUF | Apache 2.0 | Target |
| Voxtral Small 24B | 24.3B | bartowski GGUF | Apache 2.0 | Future |
| Voxtral Realtime 4B | 4B | Not available | Apache 2.0 | Deferred (llama.cpp WIP) |
| Voxtral Mini Transcribe V2 | Unknown | N/A | Proprietary | N/A |

### Architecture vs Whisper

| Feature | Whisper (current) | Voxtral Mini 3B |
|---------|-------------------|-----------------|
| Pattern | Encoder-only | Encoder + Projector + Decoder |
| Encoder | 6 heads, 80 mels | 20 heads, 128 mels, 32 layers |
| Decoder | None | Full Mistral/Llama autoregressive |
| Output | Audio embeddings | Text tokens |
| GGUF | Single file | Two files (main + mmproj) |

### GGUF Structure
- Main GGUF: `general.architecture = "llama"` (text decoder -- already supported)
- mmproj GGUF: `projector_type = "voxtral"`, Whisper-style encoder with `a.` prefix,
  adapter MLP with `mm.a.mlp.` prefix, 4x frame stacking

---

## Checkable Work Breakdown

### E1: Audio Encoder Upgrade

- [x] T1.1 Generalize WhisperEncoder for configurable mel bins and intermediate size  Owner: TBD  Est: 1h  verifies: [infrastructure]
  File: layers/audio/whisper_encoder.go
  Add IntermediateSize field (default 4*HiddenDim). Support 128 mels.
  Acceptance: Existing Whisper tests pass. Config with 128 mels/5120 intermediate works.

- [x] T1.2 Add attention bias support to WhisperEncoder  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: layers/audio/whisper_encoder.go
  Voxtral encoder has Q/K/V biases. Add optional bias loading.
  Acceptance: Encoder loads with and without bias tensors.

### E2: mmproj GGUF Loading

- [x] T2.1 Add audio config fields to ModelConfig  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: model/gguf/arch.go
  Add: AudioHiddenSize, AudioNumLayers, AudioNumHeads, AudioNumMels,
  AudioIntermediateSize, AudioProjectorType, AudioProjectorStackFactor.

- [x] T2.2 Add Voxtral mmproj tensor name mapping  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: model/gguf/arch.go
  Map a.conv1d.*, a.blk.*.ln1/ln2/attn_*/ffn_*, a.post_ln.*, mm.a.mlp.*.

- [x] T2.3 Implement two-file model loading for Voxtral  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  Deps: T2.1, T2.2
  Follow LLaVA pattern (arch_llava.go) for loading main + mmproj GGUFs.
  Acceptance: Both files loaded, tensors merged under expected names.

### E3: Graph Builder

- [x] T3.1 Implement buildVoxtralGraph  Owner: TBD  Est: 4h  verifies: [UC-001]
  Deps: T1.1, T1.2, T2.3
  File: inference/arch_voxtral.go (new)
  Combine: WhisperEncoder -> frame stacking (4x) -> MLP adapter -> Llama decoder.
  Follow arch_llava.go pattern for multimodal merging.
  Acceptance: Graph builds from synthetic weights. Forward produces logits.

- [x] T3.2 Register "voxtral" architecture  Owner: TBD  Est: 0.25h  verifies: [UC-001]
  Deps: T3.1
  File: inference/registry_init.go

- [x] T3.3 Add parseVoxtralConfig  Owner: TBD  Est: 0.5h  verifies: [UC-001]
  File: inference/arch_config.go

### E4: Audio Preprocessing

- [x] T4.1 Implement mel spectrogram extraction (128 bins)  Owner: TBD  Est: 2h  verifies: [UC-001]
  File: layers/audio/mel.go (new or extend existing)
  128 mel bins, 400 FFT, 160 hop, 16kHz. Whisper-compatible.

- [x] T4.2 Implement 30-second chunking with silence padding  Owner: TBD  Est: 1h  verifies: [UC-001]
  Deps: T4.1

### E5: Integration

- [x] T5.1 Unit tests for Voxtral builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  Deps: T3.2
  Synthetic weight tests, forward pass, registration.

- [x] T5.2 Add /v1/audio/transcriptions API endpoint  Owner: TBD  Est: 2h  verifies: [UC-001]
  Deps: T3.2, T4.1
  File: serve/audio.go (new). OpenAI-compatible.

- [x] T5.3 Add `zerfoo transcribe` CLI command  Owner: TBD  Est: 1h  verifies: [UC-001]
  Deps: T5.2

- [x] T5.4 Run go vet and linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T3.2

- [x] T5.5 Update README  Owner: TBD  Est: 0.5h  delivers: [README update]

---

## Waves

#### Wave 1: Foundation (4 agents)
- [x] T1.1 Generalize WhisperEncoder
- [x] T1.2 Add attention bias support
- [x] T2.1 Audio config fields
- [x] T2.2 Voxtral tensor name mapping

#### Wave 2: Loading + Preprocessing (3 agents)
- [x] T2.3 Two-file model loading  Deps: T2.1, T2.2
- [x] T4.1 Mel spectrogram  (independent)
- [x] T4.2 Audio chunking  Deps: T4.1

#### Wave 3: Builder (2 agents)
- [x] T3.1 buildVoxtralGraph  Deps: T1.1, T1.2, T2.3
- [x] T3.2 Register architecture  Deps: T3.1
- [x] T3.3 parseVoxtralConfig  (independent)

#### Wave 4: Integration (5 agents)
- [x] T5.1 Unit tests  Deps: T3.2
- [x] T5.2 Audio API endpoint  Deps: T3.2, T4.1
- [x] T5.3 CLI transcribe  Deps: T5.2
- [x] T5.4 go vet  Deps: T3.2
- [x] T5.5 README  Deps: T3.2

---

## Progress Log

### 2026-03-28: Plan created
- Voxtral Transcribe 2 support plan: 15 tasks, 4 waves
- Encoder-projector-decoder architecture (Whisper-large-v3 + Mistral decoder)
- Two-file GGUF loading (main + mmproj), following LLaVA pattern
- Voxtral Realtime deferred (llama.cpp support not finalized)
