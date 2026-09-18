# Golang Weekly Newsletter Pitch

## Subject Line

Zerfoo: Pure Go ML inference -- 235 tok/s, zero CGo by default

## Pitch

Zerfoo is an ML inference framework written entirely in Go. It runs
transformer models (Llama, Gemma, Mistral, Qwen, Phi, DeepSeek) as a
library call -- zero CGo in the default build, no Python, no sidecar
processes. GPU acceleration
is loaded dynamically via purego/dlopen so `go build` just works. On
Gemma 3 1B Q4_K_M it decodes at 235 tok/s, 1.25x Ollama on the same
hardware, at parity by 3B. https://github.com/zerfoo/zerfoo
