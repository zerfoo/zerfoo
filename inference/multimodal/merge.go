// Package multimodal provides image preprocessing and embedding merge for
// vision-language model inference.
package multimodal

import "fmt"

// MergeConfig controls how text and vision embeddings are merged.
type MergeConfig struct {
	// ImageTokenID is the token ID used as a placeholder for image patches
	// in the text token sequence.
	ImageTokenID int
	// MaxImageTokens is the maximum number of image tokens allowed in a
	// single sequence. Zero means no limit.
	MaxImageTokens int
	// EmbedDim is the embedding dimension shared by text and vision embeddings.
	EmbedDim int
}

// MergeResult holds the merged embedding sequence.
type MergeResult struct {
	// Embeddings is a flat [SeqLen, EmbedDim] float32 slice containing the
	// merged text and vision embeddings.
	Embeddings []float32
	// SeqLen is the sequence length of the merged output.
	SeqLen int
	// EmbedDim is the embedding dimension of the merged output.
	EmbedDim int
}

// NumImageTokens counts how many entries in tokenIDs equal imageTokenID.
func NumImageTokens(tokenIDs []int, imageTokenID int) int {
	n := 0
	for _, id := range tokenIDs {
		if id == imageTokenID {
			n++
		}
	}
	return n
}

// MergeEmbeddings replaces image-token positions in the text embedding
// sequence with consecutive vision embeddings. textEmbeds has shape
// [seqLen, EmbedDim], visionEmbeds has shape [numVisionTokens, EmbedDim]
// (already projected to text dimension space), and tokenIDs has length
// seqLen. Each position where tokenIDs[i] == cfg.ImageTokenID is replaced
// by the next vision embedding vector.
func MergeEmbeddings(textEmbeds []float32, visionEmbeds []float32, tokenIDs []int, cfg MergeConfig) (MergeResult, error) {
	seqLen := len(tokenIDs)
	if cfg.EmbedDim <= 0 {
		return MergeResult{}, fmt.Errorf("multimodal: EmbedDim must be positive, got %d", cfg.EmbedDim)
	}
	if len(textEmbeds) != seqLen*cfg.EmbedDim {
		return MergeResult{}, fmt.Errorf("multimodal: textEmbeds length %d does not match seqLen(%d) * EmbedDim(%d) = %d",
			len(textEmbeds), seqLen, cfg.EmbedDim, seqLen*cfg.EmbedDim)
	}

	numImg := NumImageTokens(tokenIDs, cfg.ImageTokenID)
	if cfg.MaxImageTokens > 0 && numImg > cfg.MaxImageTokens {
		return MergeResult{}, fmt.Errorf("multimodal: found %d image tokens, exceeds MaxImageTokens %d",
			numImg, cfg.MaxImageTokens)
	}

	numVision := len(visionEmbeds) / cfg.EmbedDim
	if len(visionEmbeds) != numVision*cfg.EmbedDim {
		return MergeResult{}, fmt.Errorf("multimodal: visionEmbeds length %d is not a multiple of EmbedDim %d",
			len(visionEmbeds), cfg.EmbedDim)
	}
	if numImg != numVision {
		return MergeResult{}, fmt.Errorf("multimodal: %d image token positions but %d vision embeddings provided",
			numImg, numVision)
	}

	out := make([]float32, seqLen*cfg.EmbedDim)
	vIdx := 0
	for i := 0; i < seqLen; i++ {
		dst := out[i*cfg.EmbedDim : (i+1)*cfg.EmbedDim]
		if tokenIDs[i] == cfg.ImageTokenID {
			copy(dst, visionEmbeds[vIdx*cfg.EmbedDim:(vIdx+1)*cfg.EmbedDim])
			vIdx++
		} else {
			copy(dst, textEmbeds[i*cfg.EmbedDim:(i+1)*cfg.EmbedDim])
		}
	}

	return MergeResult{
		Embeddings: out,
		SeqLen:     seqLen,
		EmbedDim:   cfg.EmbedDim,
	}, nil
}
