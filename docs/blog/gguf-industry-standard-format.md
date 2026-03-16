# GGUF: Why We Standardized on the Industry Format

*How we went from a custom protobuf format to GGUF -- and deleted 7,500 lines of code in the process.*

## Introduction

Model formats are infrastructure. They determine how fast models load, how much memory they consume, what tools can read them, and how painful it is to ship a model from training to production. Getting the format right matters more than most framework decisions because it affects every user, every model, and every deployment.

Zerfoo uses GGUF as its sole model format. Not one of several supported formats -- the only one. This is a deliberate choice and, honestly, a correction of an earlier mistake.

## What GGUF Is

GGUF (GPT-Generated Unified Format) was created by the llama.cpp project as a successor to the earlier GGML format. It is a binary file format designed for one purpose: storing quantized neural network weights for fast inference.

The format is straightforward:

1. **Magic number and version** -- 4 bytes of magic (`GGUF`), a version number, and counts of metadata entries and tensors.
2. **Metadata key-value pairs** -- Architecture name, tensor count, embedding dimension, attention head count, vocabulary, tokenizer config, and any other model metadata. Keys are strings, values are typed (uint32, float32, string, arrays, etc.).
3. **Tensor info** -- For each tensor: name, dimensions, data type (F32, F16, Q4_0, Q4_K_M, Q8_0, etc.), and an offset into the data section.
4. **Tensor data** -- Raw tensor bytes, aligned to a configurable boundary (typically 32 bytes). This is the bulk of the file.

The alignment matters. Because tensor data is aligned and the format has no encoding overhead (no protobuf varint encoding, no JSON escaping, no compression), the entire data section can be memory-mapped directly. `mmap` the file, read the header to find tensor offsets, and you have zero-copy access to every weight in the model. No deserialization, no temporary buffers, no 2x memory overhead.

## Why Not ONNX, SafeTensors, or PyTorch Pickle

**ONNX** stores computation graphs, not just weights. An ONNX file contains every operation in the model as decomposed primitives -- a single RMSNorm becomes Pow, ReduceMean, Add, Sqrt, Div, Mul. This is useful for portability across runtimes, but it means every inference framework has to either execute the decomposed graph (slow) or reverse-engineer fused operations from the decomposed pattern (fragile). For Zerfoo, the decomposed ONNX graph produced 4-16 tok/s. The architecture-specific GGUF path produces 232+ tok/s. The computation graph belongs in the framework, not the file format.

**SafeTensors** is a good format. It is simple, memory-mappable, and safe (no arbitrary code execution). But it stores unquantized weights only. It has no built-in support for the quantization types that make small-model inference practical (Q4_0, Q4_K_M, Q8_0). And its ecosystem is smaller -- while HuggingFace supports SafeTensors natively, GGUF has become the de facto standard for quantized inference models.

**PyTorch pickle** (`model.bin`, `.pt` files) uses Python's pickle protocol, which can execute arbitrary code during deserialization. This is a security concern for any model loaded from an untrusted source. Beyond security, pickle files are Python-specific, require PyTorch to load, and do not support mmap.

## Why GGUF Won

GGUF won because the llama.cpp project won. llama.cpp became the dominant open-source inference engine, and its model format became the standard. The practical consequences:

**Every model has a GGUF version.** Search HuggingFace for any popular open-weights model (Llama, Gemma, Mistral, Qwen, Phi) and you will find GGUF files in multiple quantization levels. Community members quantize and upload GGUF variants within hours of a model release. This is the kind of ecosystem momentum that no technical argument can overcome.

**Quantization is a first-class concept.** GGUF supports over 20 quantization types, from simple Q4_0 (4-bit uniform) to sophisticated Q4_K_M (4-bit with per-block scaling and mixed precision). The quantization type is stored per-tensor in the file header, so a single file can mix precision levels across layers. This is how the K-quants work: attention layers get higher precision than feed-forward layers.

**Mmap makes loading fast.** On the DGX Spark with NVMe storage, loading a 1B parameter model from GGUF takes milliseconds because there is no deserialization -- just mmap and read the header. The weights stay on disk until they are accessed, and the OS page cache handles the rest.

**Interoperability comes for free.** A GGUF file that works with Zerfoo also works with llama.cpp, Ollama, LM Studio, GPT4All, and every other tool that reads GGUF. Users do not have to convert models to use Zerfoo, and they do not have to convert away if they stop using it.

**The format is simple.** The GGUF specification is a single document. A minimal parser that reads metadata and tensor info is a few hundred lines of code in any language. Zerfoo's GGUF parser in `model/gguf.go` handles the full spec including all quantization types.

## Our Journey: The ZMF Mistake

We did not start with GGUF. Zerfoo originally used a custom format called ZMF (Zerfoo Model Format), stored in a separate `zmf` repository. ZMF used Protocol Buffers to serialize model weights and metadata.

This was a mistake, and we should be honest about why.

**Protobuf is wrong for tensor storage.** Protocol Buffers encode bytes fields with length-prefixed varint encoding. This means every tensor is copied during deserialization -- you cannot mmap a protobuf file and get zero-copy access to tensor data. For a 1B parameter model in float32, that is 4 GB of data that must be fully deserialized into memory before inference can begin. Loading was roughly 50x slower than GGUF's mmap path.

**Two inference paths doubled the maintenance burden.** With ZMF, Zerfoo had two ways to run inference: the GGUF path (architecture-specific builders with fused operations) and the ZMF/ONNX path (generic graph execution with decomposed operations). The GGUF path was fast. The ZMF path was 15-60x slower and produced lower quality output because decomposed operations lost numerical precision at the boundaries.

**The graph fusion pass never worked reliably.** To close the performance gap, we built a fusion pass (`graph/fusion.go`) that tried to recognize decomposed patterns and replace them with fused operations. For example, detect the six-operation RMSNorm pattern (Pow, ReduceMean, Add, Sqrt, Div, Mul) and replace it with a single FusedRMSNorm. This pass was blocked by a runtime slot resolution bug (PR #70) that was never fixed. The bug could not be fixed without a fundamental redesign of the slot allocation system, which was not worth the effort when the GGUF path already worked.

**Nobody used ZMF.** Every user who wanted fast inference used GGUF files. The ZMF path existed as a technical curiosity that demonstrated ONNX compatibility but was never fast enough for production use.

ADR 037 formalized the decision: GGUF is the sole model format. We deleted approximately 7,500 lines of code from zerfoo (the ZMF loader, exporter, tensor encoder/decoder, graph builder, fusion pass, and associated tests), closed PR #70 without fixing it, and archived the zmf repository.

The `zonnx` tool was pivoted from ONNX-to-ZMF conversion to ONNX-to-GGUF conversion, making it the only pure-Go ONNX-to-GGUF converter in the ecosystem.

## Practical Benefits

For users, the GGUF-only strategy means:

**Download and run.** Find a GGUF model on HuggingFace. Download it. Point Zerfoo at it. There is no conversion step, no format negotiation, no "which format should I use?" decision.

```bash
# Download a model
wget https://huggingface.co/.../gemma-3-1b-q4_k_m.gguf

# Run inference
zerfoo run -model gemma-3-1b-q4_k_m.gguf -prompt "Hello, world"
```

**Same files work everywhere.** The GGUF file you use with Zerfoo is the same file you can use with Ollama, llama.cpp, or LM Studio. If you evaluate Zerfoo and decide it is not for you, your model files still work.

**Quantization choices are pre-made.** GGUF files on HuggingFace come in multiple quantization levels (Q4_0, Q4_K_M, Q8_0, F16). Pick the one that fits your memory budget and throughput requirements. No need to run a quantization pipeline yourself.

**ONNX models are supported via conversion.** If you have an ONNX model, `zonnx` converts it to GGUF:

```bash
zonnx convert model.onnx -o model.gguf
```

The conversion happens once, offline. The resulting GGUF file loads at full speed.

**Future: training checkpoints in GGUF.** When Zerfoo implements training checkpoint saving, it will write GGUF files. Model weights go in as tensors; training state (epoch, learning rate, optimizer name) goes in as metadata key-value pairs; optimizer state (Adam first and second moments) goes in as additional tensors with a naming convention. One format for inference and training.

## Conclusion

Standards win. Not because they are technically perfect -- GGUF has quirks, the spec has evolved through several breaking versions, and some metadata conventions are underspecified. Standards win because they eliminate the conversion tax, the compatibility matrix, and the "which format?" decision.

We tried building a custom format. It was slower, harder to maintain, and nobody used it. GGUF is faster, simpler, and already has every model we care about. The ecosystem chose GGUF. We chose to follow.
