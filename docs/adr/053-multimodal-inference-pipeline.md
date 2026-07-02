# ADR 053: Multi-Modal Inference Pipeline

## Status
Accepted

## Date
2026-03-17

## Context
Wolf's trading decision loop will benefit from multi-modal inputs: (1) vision input
from earnings report charts, SEC filing tables, and financial news screenshots, and
(2) audio input from Federal Reserve press conference transcripts and earnings calls.
Vision-language models (VLMs) are production-proven in 2025 (Gemma 3 multimodal,
LLaVA, InternVL, Qwen-VL). Audio models (Whisper-family) are production-proven.

Zerfoo already imports SigLIP vision encoder weights for Gemma 3 multimodal in the
GGUF loader (ADR-019 BF16+SigLIP). The infrastructure exists; it needs to be
generalized.

## Decision
Generalize multi-modal inference into a composable pipeline in inference/multimodal/:

Vision Encoder (inference/multimodal/vision_encoder.go):
- Interface: VisionEncoder[T].Encode(image []byte, format ImageFormat) (*Tensor[T], error)
- SigLIP implementation: loads from GGUF vision tower weights
- CLIP implementation: alternative for non-Gemma models (LLaVA, InternVL)
- Output: vision embedding tensor of shape [num_patches, hidden_dim]
- Image preprocessing: resize to 224x224 (SigLIP) or model-specified resolution,
  normalize to [-1, 1], convert to patch embeddings

Connector (inference/multimodal/connector.go):
- Projection-based (default): linear projection from vision_hidden_dim to text_hidden_dim
- Query-based: Q-Former with fixed number of learnable query tokens (BLIP-2 style)
- Connector weights loaded from GGUF: "mm.projector.weight", "mm.projector.bias"

Text+Vision Merge (inference/multimodal/merge.go):
- Inserts vision embeddings at <image> token positions in the text embedding sequence
- Supports multiple images per request (multiimage GGUF metadata)
- Position IDs for vision tokens are assigned contiguously before text tokens

Audio Pipeline (inference/multimodal/audio.go):
- Mel-spectrogram extraction: 80-mel filterbank, 25ms window, 10ms hop
- Whisper-style encoder: convolutional frontend + Transformer encoder
- Output: audio embedding tensor aligned with text token positions
- Used for Fed call audio; files streamed in via serve API multipart/form-data

GGUF Extension for Multi-Modal:
- New metadata keys: "vision.encoder.type", "vision.hidden_size", "audio.encoder.type"
- Wolf-specific use case: process PDF pages as images, extract tables/charts via VLM

## Consequences
Positive:
- Generalizes existing Gemma 3 multimodal support to all VLM architectures
- Audio support enables Wolf to process Fed call recordings automatically
- Composable design allows mixing vision+audio in a single inference session

Negative:
- Image preprocessing is CPU-bound; parallel preprocessing workers needed for
  high-throughput batch image inference
- Multi-modal GGUF schemas are not yet standardized (as of 2026); schema churn risk
- Vision encoder adds 200-500MB to model size; memory budget planning required
- Audio pipeline requires external audio decoding library (mp3/opus); CGo risk;
  prefer pure Go implementations (e.g., opus decoder in pure Go)
