package multimodal

import (
	"testing"
)

func TestMergeEmbeddings(t *testing.T) {
	// 10 text tokens, image tokens at positions 3 and 4, embedDim=4.
	embedDim := 4
	seqLen := 10
	imageTokenID := 999

	tokenIDs := make([]int, seqLen)
	for i := range tokenIDs {
		tokenIDs[i] = i + 1 // regular token IDs 1..10
	}
	tokenIDs[3] = imageTokenID
	tokenIDs[4] = imageTokenID

	// Text embeddings: each position i has all values = float32(i+1).
	textEmbeds := make([]float32, seqLen*embedDim)
	for i := 0; i < seqLen; i++ {
		for d := 0; d < embedDim; d++ {
			textEmbeds[i*embedDim+d] = float32(i + 1)
		}
	}

	// Vision embeddings: 2 vectors, first = -1.0, second = -2.0.
	visionEmbeds := make([]float32, 2*embedDim)
	for d := 0; d < embedDim; d++ {
		visionEmbeds[d] = -1.0
		visionEmbeds[embedDim+d] = -2.0
	}

	cfg := MergeConfig{
		ImageTokenID: imageTokenID,
		EmbedDim:     embedDim,
	}

	result, err := MergeEmbeddings(textEmbeds, visionEmbeds, tokenIDs, cfg)
	if err != nil {
		t.Fatalf("MergeEmbeddings returned error: %v", err)
	}

	// Position 3 should have vision embed -1.0.
	for d := 0; d < embedDim; d++ {
		got := result.Embeddings[3*embedDim+d]
		if got != -1.0 {
			t.Errorf("position 3, dim %d: got %f, want -1.0", d, got)
		}
	}

	// Position 4 should have vision embed -2.0.
	for d := 0; d < embedDim; d++ {
		got := result.Embeddings[4*embedDim+d]
		if got != -2.0 {
			t.Errorf("position 4, dim %d: got %f, want -2.0", d, got)
		}
	}
}

func TestMergePreservesText(t *testing.T) {
	embedDim := 4
	seqLen := 6
	imageTokenID := 999

	tokenIDs := []int{1, 2, imageTokenID, 4, 5, 6}

	textEmbeds := make([]float32, seqLen*embedDim)
	for i := 0; i < seqLen; i++ {
		for d := 0; d < embedDim; d++ {
			textEmbeds[i*embedDim+d] = float32(i*embedDim + d)
		}
	}

	visionEmbeds := make([]float32, 1*embedDim)
	for d := 0; d < embedDim; d++ {
		visionEmbeds[d] = -99.0
	}

	cfg := MergeConfig{
		ImageTokenID: imageTokenID,
		EmbedDim:     embedDim,
	}

	result, err := MergeEmbeddings(textEmbeds, visionEmbeds, tokenIDs, cfg)
	if err != nil {
		t.Fatalf("MergeEmbeddings returned error: %v", err)
	}

	// All non-image positions should be unchanged.
	nonImagePositions := []int{0, 1, 3, 4, 5}
	for _, pos := range nonImagePositions {
		for d := 0; d < embedDim; d++ {
			want := textEmbeds[pos*embedDim+d]
			got := result.Embeddings[pos*embedDim+d]
			if got != want {
				t.Errorf("position %d, dim %d: got %f, want %f", pos, d, got, want)
			}
		}
	}
}

func TestMergeShape(t *testing.T) {
	embedDim := 8
	seqLen := 5
	imageTokenID := 42

	tokenIDs := []int{1, imageTokenID, 3, imageTokenID, 5}

	textEmbeds := make([]float32, seqLen*embedDim)
	visionEmbeds := make([]float32, 2*embedDim)

	cfg := MergeConfig{
		ImageTokenID: imageTokenID,
		EmbedDim:     embedDim,
	}

	result, err := MergeEmbeddings(textEmbeds, visionEmbeds, tokenIDs, cfg)
	if err != nil {
		t.Fatalf("MergeEmbeddings returned error: %v", err)
	}

	if result.SeqLen != seqLen {
		t.Errorf("SeqLen: got %d, want %d", result.SeqLen, seqLen)
	}
	if result.EmbedDim != embedDim {
		t.Errorf("EmbedDim: got %d, want %d", result.EmbedDim, embedDim)
	}
	if len(result.Embeddings) != seqLen*embedDim {
		t.Errorf("Embeddings length: got %d, want %d", len(result.Embeddings), seqLen*embedDim)
	}
}

func TestNumImageTokens(t *testing.T) {
	tests := []struct {
		name     string
		tokenIDs []int
		imgID    int
		want     int
	}{
		{"none", []int{1, 2, 3, 4, 5}, 99, 0},
		{"all", []int{99, 99, 99}, 99, 3},
		{"mixed", []int{1, 99, 2, 99, 3}, 99, 2},
		{"empty", []int{}, 99, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NumImageTokens(tt.tokenIDs, tt.imgID)
			if got != tt.want {
				t.Errorf("NumImageTokens: got %d, want %d", got, tt.want)
			}
		})
	}
}
