package multimodal

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// generateChartImage creates a synthetic 224x224 PNG chart image.
// Bull charts are filled with green; bear charts are filled with red.
// The fill color dominates so that the (G - R) signal in the encoder
// clearly distinguishes the two classes.
func generateChartImage(bull bool) []byte {
	const size = 224
	img := image.NewNRGBA(image.Rect(0, 0, size, size))

	var fillColor color.NRGBA
	if bull {
		// Green fill — high G, low R.
		fillColor = color.NRGBA{R: 20, G: 200, B: 20, A: 255}
	} else {
		// Red fill — high R, low G.
		fillColor = color.NRGBA{R: 200, G: 20, B: 20, A: 255}
	}

	// Fill background with the trend color.
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			img.SetNRGBA(x, y, fillColor)
		}
	}

	// Draw a contrasting trend line so it looks like a chart.
	for x := 0; x < size; x++ {
		var lineY int
		if bull {
			lineY = size - 20 - int(float64(x)*float64(size-40)/float64(size))
		} else {
			lineY = 20 + int(float64(x)*float64(size-40)/float64(size))
		}
		for dy := -3; dy <= 3; dy++ {
			py := lineY + dy
			if py >= 0 && py < size {
				img.SetNRGBA(x, py, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
			}
		}
	}

	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}

// classifyFromMerged determines bull/bear from merged embeddings.
// It sums hidden dimension 0 across all vision-token positions.
// With the engineered encoder weights, dim 0 holds the accumulated
// (G - R) signal: positive means bull (green line), negative means bear (red line).
func classifyFromMerged(result MergeResult, tokenIDs []int, imageTokenID int) string {
	var sum float64
	for i, id := range tokenIDs {
		if id == imageTokenID {
			// Read dim 0 of this token's embedding.
			sum += float64(result.Embeddings[i*result.EmbedDim])
		}
	}
	if sum >= 0 {
		return "bull"
	}
	return "bear"
}

// TestEarningsChartInference exercises the full vision-language pipeline:
// image generation -> preprocessing -> encoding -> projection -> merge -> classification.
func TestEarningsChartInference(t *testing.T) {
	const (
		imageSize    = 224
		patchSize    = 16
		hiddenDim    = 32 // small dims for fast tests
		textDim      = 32
		imageTokenID = 99999
	)

	patchCfg := PatchConfig{
		PatchSize: patchSize,
		ImageSize: imageSize,
		NormMean:  [3]float32{0.485, 0.456, 0.406},
		NormStd:   [3]float32{0.229, 0.224, 0.225},
	}

	numPatches := NumPatches(patchCfg) // 14*14 = 196

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Build encoder with deterministic weights that extract green-minus-red
	// signal. Patches are interleaved [R,G,B,R,G,B,...] with patchDim values
	// per patch. We set weights so hidden dim 0 accumulates (G - R) for each
	// pixel in the patch. Bull charts have green lines (high G), bear charts
	// have red lines (high R), so the sign of hidden dim 0 distinguishes them.
	encCfg := EncoderConfig{
		HiddenDim: hiddenDim,
		NumHeads:  4,
		NumLayers: 1,
		PatchCfg:  patchCfg,
	}
	encoder := NewSigLIPEncoder[float32](encCfg, engine)

	// Override encoder weights: [patchDim, hiddenDim].
	patchDim := PatchDim(patchCfg) // 16*16*3 = 768
	encWeights := make([]float32, patchDim*hiddenDim)
	for px := 0; px < patchSize*patchSize; px++ {
		// Weight is [patchDim, hiddenDim], row-major.
		// Row (px*3+0) = R channel of pixel px, col 0 = hidden dim 0.
		encWeights[(px*3+0)*hiddenDim+0] = -1.0 // R -> dim 0 with weight -1
		encWeights[(px*3+1)*hiddenDim+0] = 1.0  // G -> dim 0 with weight +1
		// B channel and other hidden dims remain 0.
	}
	encW, _ := tensor.New[float32]([]int{patchDim, hiddenDim}, encWeights)
	encoder.weight = encW

	// Build projection connector (identity-like: hiddenDim == textDim).
	connCfg := ConnectorConfig{VisionDim: hiddenDim, TextDim: textDim}
	connector := NewProjectionConnector[float32](connCfg, engine)

	// Set projection weights to scaled identity so signal passes through.
	projWeights := make([]float32, hiddenDim*textDim)
	for i := 0; i < hiddenDim && i < textDim; i++ {
		projWeights[i*textDim+i] = 1.0
	}
	if err := connector.LoadWeights(projWeights); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}

	// Merge config.
	mergeCfg := MergeConfig{
		ImageTokenID:   imageTokenID,
		MaxImageTokens: numPatches,
		EmbedDim:       textDim,
	}

	// Build token sequence: [BOS, <img>, <img>, ..., <img>, EOS].
	tokenIDs := make([]int, numPatches+2)
	tokenIDs[0] = 1 // BOS
	for i := 1; i <= numPatches; i++ {
		tokenIDs[i] = imageTokenID
	}
	tokenIDs[numPatches+1] = 2 // EOS

	seqLen := len(tokenIDs)

	// Stub text embeddings (zeros — only image tokens matter).
	textEmbeds := make([]float32, seqLen*textDim)

	type testCase struct {
		name     string
		bull     bool
		wantTrend string
	}

	cases := []testCase{
		{"bull_1", true, "bull"},
		{"bull_2", true, "bull"},
		{"bull_3", true, "bull"},
		{"bull_4", true, "bull"},
		{"bull_5", true, "bull"},
		{"bear_1", false, "bear"},
		{"bear_2", false, "bear"},
		{"bear_3", false, "bear"},
		{"bear_4", false, "bear"},
		{"bear_5", false, "bear"},
	}

	correct := 0
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Step 1: Generate synthetic chart image.
			imgData := generateChartImage(tc.bull)

			// Step 2: Preprocess image into patches.
			patches, err := PreprocessImage(imgData, PNG, patchCfg)
			if err != nil {
				t.Fatalf("PreprocessImage: %v", err)
			}
			wantPatchLen := numPatches * PatchDim(patchCfg)
			if len(patches) != wantPatchLen {
				t.Fatalf("patches length = %d, want %d", len(patches), wantPatchLen)
			}

			// Step 3: Encode patches through vision encoder.
			encoded, err := encoder.Encode(patches, patchCfg)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
			wantEncLen := numPatches * hiddenDim
			if len(encoded) != wantEncLen {
				t.Fatalf("encoded length = %d, want %d", len(encoded), wantEncLen)
			}

			// Step 4: Project vision embeddings to text space.
			projected, err := connector.Project(encoded, numPatches)
			if err != nil {
				t.Fatalf("Project: %v", err)
			}
			wantProjLen := numPatches * textDim
			if len(projected) != wantProjLen {
				t.Fatalf("projected length = %d, want %d", len(projected), wantProjLen)
			}

			// Step 5: Merge text and vision embeddings.
			visionEmbeds := make([]float32, len(projected))
			copy(visionEmbeds, projected)

			merged, err := MergeEmbeddings(textEmbeds, visionEmbeds, tokenIDs, mergeCfg)
			if err != nil {
				t.Fatalf("MergeEmbeddings: %v", err)
			}
			if merged.SeqLen != seqLen {
				t.Fatalf("merged SeqLen = %d, want %d", merged.SeqLen, seqLen)
			}
			if merged.EmbedDim != textDim {
				t.Fatalf("merged EmbedDim = %d, want %d", merged.EmbedDim, textDim)
			}

			// Verify merged embeddings are not all zeros (vision signal present).
			hasNonZero := false
			for _, v := range merged.Embeddings {
				if v != 0 {
					hasNonZero = true
					break
				}
			}
			if !hasNonZero {
				t.Fatal("merged embeddings are all zeros — vision signal lost")
			}

			// Step 6: Classify trend from merged embeddings.
			got := classifyFromMerged(merged, tokenIDs, imageTokenID)
			if got == tc.wantTrend {
				correct++
			} else {
				t.Logf("misclassified: got %s, want %s", got, tc.wantTrend)
			}
		})
	}

	// Accuracy gate: > 80% (at least 9 of 10).
	accuracy := float64(correct) / float64(len(cases))
	t.Logf("classification accuracy: %.0f%% (%d/%d)", accuracy*100, correct, len(cases))
	if accuracy <= 0.8 {
		t.Errorf("accuracy %.0f%% does not meet >80%% threshold", accuracy*100)
	}
}

// TestEarningsChartPipelineShapes verifies that each pipeline stage produces
// correctly-shaped outputs for both bull and bear charts.
func TestEarningsChartPipelineShapes(t *testing.T) {
	const (
		imageSize = 224
		patchSize = 16
		hiddenDim = 16
		textDim   = 16
	)

	patchCfg := PatchConfig{
		PatchSize: patchSize,
		ImageSize: imageSize,
		NormMean:  [3]float32{0.5, 0.5, 0.5},
		NormStd:   [3]float32{0.5, 0.5, 0.5},
	}
	numPatches := NumPatches(patchCfg)
	patchDim := PatchDim(patchCfg)

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	for _, bull := range []bool{true, false} {
		label := "bear"
		if bull {
			label = "bull"
		}
		t.Run(label, func(t *testing.T) {
			imgData := generateChartImage(bull)

			// Preprocess.
			patches, err := PreprocessImage(imgData, PNG, patchCfg)
			if err != nil {
				t.Fatalf("PreprocessImage: %v", err)
			}
			if len(patches) != numPatches*patchDim {
				t.Fatalf("patches: got %d, want %d", len(patches), numPatches*patchDim)
			}

			// Verify normalized values are finite.
			for i, v := range patches {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("patches[%d] is not finite: %f", i, v)
				}
			}

			// Encode.
			enc := NewSigLIPEncoder[float32](EncoderConfig{
				HiddenDim: hiddenDim,
				NumHeads:  4,
				NumLayers: 1,
				PatchCfg:  patchCfg,
			}, engine)
			encoded, err := enc.Encode(patches, patchCfg)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
			if len(encoded) != numPatches*hiddenDim {
				t.Fatalf("encoded: got %d, want %d", len(encoded), numPatches*hiddenDim)
			}

			// Project.
			conn := NewProjectionConnector[float32](ConnectorConfig{
				VisionDim: hiddenDim,
				TextDim:   textDim,
			}, engine)
			projected, err := conn.Project(encoded, numPatches)
			if err != nil {
				t.Fatalf("Project: %v", err)
			}
			if len(projected) != numPatches*textDim {
				t.Fatalf("projected: got %d, want %d", len(projected), numPatches*textDim)
			}
		})
	}
}
