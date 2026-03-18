// Package multimodal provides image preprocessing for vision-language model inference.
package multimodal

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"

	"golang.org/x/image/draw"
)

// ImageFormat identifies the encoding format of an input image.
type ImageFormat int

const (
	JPEG ImageFormat = 1
	PNG  ImageFormat = 2
)

// PatchConfig specifies how an image should be resized, normalized, and
// divided into patches for a vision encoder.
type PatchConfig struct {
	PatchSize int
	ImageSize int
	NormMean  [3]float32
	NormStd   [3]float32
}

// NumPatches returns the number of patches the image is divided into.
func NumPatches(cfg PatchConfig) int {
	n := cfg.ImageSize / cfg.PatchSize
	return n * n
}

// PatchDim returns the dimensionality of each patch embedding (PatchSize*PatchSize*3).
func PatchDim(cfg PatchConfig) int {
	return cfg.PatchSize * cfg.PatchSize * 3
}

// PreprocessImage decodes an image from raw bytes, resizes it to
// cfg.ImageSize x cfg.ImageSize using bilinear interpolation, normalizes
// pixel values per channel, and returns the result as flattened patch
// embeddings of shape [num_patches, patch_dim].
func PreprocessImage(data []byte, format ImageFormat, cfg PatchConfig) ([]float32, error) {
	img, err := decodeImage(data, format)
	if err != nil {
		return nil, fmt.Errorf("multimodal: decode image: %w", err)
	}

	resized := resizeImage(img, cfg.ImageSize)
	pixels := normalizePixels(resized, cfg)
	patches := extractPatches(pixels, cfg)
	return patches, nil
}

func decodeImage(data []byte, format ImageFormat) (image.Image, error) {
	r := bytes.NewReader(data)
	switch format {
	case JPEG:
		return jpeg.Decode(r)
	case PNG:
		return png.Decode(r)
	default:
		return nil, fmt.Errorf("unsupported image format: %d", format)
	}
}

func resizeImage(img image.Image, size int) *image.NRGBA {
	dst := image.NewNRGBA(image.Rect(0, 0, size, size))
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)
	return dst
}

// normalizePixels converts the resized image to a [3][H][W] float32 array
// normalized by (pixel/255 - mean) / std per channel.
func normalizePixels(img *image.NRGBA, cfg PatchConfig) [3][]float32 {
	h := cfg.ImageSize
	w := cfg.ImageSize
	var channels [3][]float32
	for c := 0; c < 3; c++ {
		channels[c] = make([]float32, h*w)
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := y*img.Stride + x*4
			r := float32(img.Pix[off]) / 255.0
			g := float32(img.Pix[off+1]) / 255.0
			b := float32(img.Pix[off+2]) / 255.0

			idx := y*w + x
			channels[0][idx] = (r - cfg.NormMean[0]) / cfg.NormStd[0]
			channels[1][idx] = (g - cfg.NormMean[1]) / cfg.NormStd[1]
			channels[2][idx] = (b - cfg.NormMean[2]) / cfg.NormStd[2]
		}
	}
	return channels
}

// extractPatches rearranges normalized pixel data into patch embeddings.
// Output is flattened [num_patches, patch_dim] where each patch contains
// PatchSize*PatchSize pixels for all 3 channels interleaved as [R,G,B,...].
func extractPatches(pixels [3][]float32, cfg PatchConfig) []float32 {
	np := NumPatches(cfg)
	pd := PatchDim(cfg)
	out := make([]float32, np*pd)

	patchesPerRow := cfg.ImageSize / cfg.PatchSize

	for py := 0; py < patchesPerRow; py++ {
		for px := 0; px < patchesPerRow; px++ {
			patchIdx := py*patchesPerRow + px
			base := patchIdx * pd
			i := 0
			for dy := 0; dy < cfg.PatchSize; dy++ {
				for dx := 0; dx < cfg.PatchSize; dx++ {
					imgY := py*cfg.PatchSize + dy
					imgX := px*cfg.PatchSize + dx
					pixIdx := imgY*cfg.ImageSize + imgX
					out[base+i] = pixels[0][pixIdx]
					out[base+i+1] = pixels[1][pixIdx]
					out[base+i+2] = pixels[2][pixIdx]
					i += 3
				}
			}
		}
	}
	return out
}
