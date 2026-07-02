package multimodal

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"testing"
)

func syntheticImage(w, h int, c color.Color) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

func encodeJPEG(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatalf("encode JPEG: %v", err)
	}
	return buf.Bytes()
}

func encodePNG(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode PNG: %v", err)
	}
	return buf.Bytes()
}

func defaultCfg() PatchConfig {
	return PatchConfig{
		PatchSize: 16,
		ImageSize: 64,
		NormMean:  [3]float32{0.5, 0.5, 0.5},
		NormStd:   [3]float32{0.5, 0.5, 0.5},
	}
}

func TestPreprocessJPEG(t *testing.T) {
	img := syntheticImage(64, 64, color.NRGBA{R: 128, G: 64, B: 32, A: 255})
	data := encodeJPEG(t, img)
	cfg := defaultCfg()

	out, err := PreprocessImage(data, JPEG, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}

	np := NumPatches(cfg)
	pd := PatchDim(cfg)
	wantLen := np * pd
	if len(out) != wantLen {
		t.Errorf("output length = %d, want %d (patches=%d, dim=%d)", len(out), wantLen, np, pd)
	}
}

func TestPreprocessPNG(t *testing.T) {
	img := syntheticImage(64, 64, color.NRGBA{R: 200, G: 100, B: 50, A: 255})
	data := encodePNG(t, img)
	cfg := defaultCfg()

	out, err := PreprocessImage(data, PNG, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}

	np := NumPatches(cfg)
	pd := PatchDim(cfg)
	wantLen := np * pd
	if len(out) != wantLen {
		t.Errorf("output length = %d, want %d (patches=%d, dim=%d)", len(out), wantLen, np, pd)
	}
}

func TestNormalization(t *testing.T) {
	// Use a uniform color image so we can predict normalized values.
	img := syntheticImage(64, 64, color.NRGBA{R: 128, G: 128, B: 128, A: 255})
	data := encodePNG(t, img)
	cfg := PatchConfig{
		PatchSize: 16,
		ImageSize: 64,
		NormMean:  [3]float32{0.5, 0.5, 0.5},
		NormStd:   [3]float32{0.5, 0.5, 0.5},
	}

	out, err := PreprocessImage(data, PNG, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}

	// 128/255 ≈ 0.502; normalized = (0.502 - 0.5) / 0.5 ≈ 0.004
	// All values should be close to 0 and well within [-1, 1].
	for i, v := range out {
		if v < -1.0 || v > 1.0 {
			t.Fatalf("out[%d] = %f, want in [-1, 1]", i, v)
		}
		if math.Abs(float64(v)) > 0.1 {
			t.Fatalf("out[%d] = %f, expected near 0 for uniform gray image", i, v)
		}
	}
}

func TestPatchCount(t *testing.T) {
	tests := []struct {
		name      string
		patchSize int
		imageSize int
		want      int
	}{
		{"224/16", 16, 224, 196},
		{"64/16", 16, 64, 16},
		{"64/32", 32, 64, 4},
		{"384/16", 16, 384, 576},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := PatchConfig{PatchSize: tt.patchSize, ImageSize: tt.imageSize}
			got := NumPatches(cfg)
			if got != tt.want {
				t.Errorf("NumPatches(%+v) = %d, want %d", cfg, got, tt.want)
			}
		})
	}
}

func TestPatchDim(t *testing.T) {
	cfg := PatchConfig{PatchSize: 16}
	got := PatchDim(cfg)
	want := 16 * 16 * 3
	if got != want {
		t.Errorf("PatchDim = %d, want %d", got, want)
	}
}

func TestUnsupportedFormat(t *testing.T) {
	_, err := PreprocessImage([]byte{0}, ImageFormat(99), defaultCfg())
	if err == nil {
		t.Fatal("expected error for unsupported format")
	}
}

func TestNonSquareInput(t *testing.T) {
	// Non-square input should be resized to square.
	img := syntheticImage(128, 32, color.NRGBA{R: 100, G: 100, B: 100, A: 255})
	data := encodePNG(t, img)
	cfg := defaultCfg()

	out, err := PreprocessImage(data, PNG, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}

	wantLen := NumPatches(cfg) * PatchDim(cfg)
	if len(out) != wantLen {
		t.Errorf("output length = %d, want %d", len(out), wantLen)
	}
}
