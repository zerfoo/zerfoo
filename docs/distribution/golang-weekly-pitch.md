# Golang Weekly Newsletter Pitch

## Subject Line

Zerfoo: Pure Go ML inference -- 241 tok/s, zero CGo

## Pitch

Zerfoo is an ML inference framework written entirely in Go. It runs
transformer models (Llama, Gemma, Mistral, Qwen, Phi, DeepSeek) as a
library call -- no CGo, no Python, no sidecar processes. GPU acceleration
is loaded dynamically via purego/dlopen so `go build` just works. On
Gemma 3 1B Q4_K_M it decodes at 241 tok/s, 28% faster than Ollama
on the same hardware. https://github.com/zerfoo/zerfoo
