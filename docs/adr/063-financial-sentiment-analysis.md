# ADR 063: Financial Sentiment Analysis Architecture

## Status

Accepted

## Date

2026-03-20

## Context

The internal consumer needs high-performance financial sentiment analysis to extract
directional signals from news headlines, earnings calls, SEC filings, and social
media. The current approach uses grammar-constrained LLM decoding (examples/classification)
which works but is fundamentally bottlenecked by autoregressive generation speed
(~245 tok/s) and wastes compute generating tokens when a single classification score
is needed.

Financial sentiment analysis in production requires:
- Sub-millisecond per-headline latency for real-time signal extraction
- Batch processing of thousands of headlines per second for historical backtesting
- Three-class (positive/neutral/negative) and continuous sentiment scoring
- Domain-specific understanding of financial language (e.g., "beat expectations" = positive)
- Fine-tunability on proprietary labeled data

The industry standard is encoder-only transformer models (BERT/RoBERTa/DeBERTa) with
a sequence classification head. FinBERT (ProsusAI/finbert, based on BERT-base) achieves
~87% accuracy on Financial PhraseBank and processes 10,000+ sentences/second on GPU.

Zerfoo currently has no encoder-only architecture support. All 20+ validated architectures
are decoder-only (causal attention) or encoder-decoder (Whisper). Adding bidirectional
self-attention and a classification head requires new inference and layer code but can
reuse existing Engine[T], attention, normalization, embedding, and training infrastructure.

## Decision

1. **Add encoder-only transformer support** via a new architecture builder
   (inference/arch_bert.go) that constructs bidirectional self-attention blocks
   (no causal mask) with BERT/RoBERTa/DeBERTa compatibility.

2. **Add a sequence classification head** (layers/core/seq_classification.go) that
   performs CLS-token pooling (or mean pooling) followed by a linear projection to
   num_classes. Supports both softmax classification and sigmoid regression.

3. **Add WordPiece tokenizer** to ztoken for BERT-family model compatibility.
   BERT uses WordPiece (not BPE); ztoken currently rejects WordPiece models.

4. **Use GGUF as the model format** for encoder models. Convert FinBERT and other
   BERT-family models from HuggingFace safetensors to GGUF via zonnx. This maintains
   the single-format constraint (ADR-037).

5. **Support fine-tuning** via existing Trainer[T] + LoRA infrastructure. Users load
   a pre-trained encoder GGUF, attach a classification head, and fine-tune on
   domain-specific labeled data using cross-entropy loss.

6. **Add a dedicated sentiment API endpoint** (/v1/sentiment) to the OpenAI-compatible
   server for batch sentiment scoring, separate from the chat/completion endpoints.

7. **Target performance**: 10,000+ sentences/second on DGX Spark (BERT-base, batch=64,
   CUDA, FP16). This is ~40x faster than LLM-based classification.

## Consequences

**Positive:**
- Unlocks encoder-only model family for Zerfoo (BERT, RoBERTa, DeBERTa, DistilBERT,
  ALBERT, XLNet, ELECTRA). Significant expansion of supported architectures.
- 40x+ throughput improvement over grammar-constrained LLM classification.
- Reuses existing Engine[T], attention, normalization, embedding, training, and serving
  infrastructure. Estimated ~70% code reuse.
- FinBERT is well-validated in financial NLP; reduces risk vs. training from scratch.
- WordPiece tokenizer in ztoken benefits any future BERT-family use cases.

**Negative:**
- Bidirectional attention cannot reuse KV cache or CUDA graph capture optimized for
  causal decoding. Encoder models need separate optimization paths.
- BERT-to-GGUF conversion in zonnx requires new tensor mapping logic.
- WordPiece is a separate tokenizer algorithm from BPE; adds maintenance surface in ztoken.
- Testing requires downloading FinBERT weights (~440 MB for BERT-base).
