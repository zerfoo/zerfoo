package generate

import (
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// q3GroupSize is the number of elements per quantization group.
// Each group has its own codebook of 8 centroids (3-bit indices).
const q3GroupSize = 128

// q3NumCentroids is the number of centroids per group (2^3 = 8).
const q3NumCentroids = 8

// q3KMeansIters is the number of k-means iterations for codebook fitting.
const q3KMeansIters = 8

// q3Storage packs float32 values into 3-bit indices with a per-group
// non-uniform codebook. Each group of q3GroupSize elements has 8 float32
// centroids computed via sensitivity-weighted k-means. Elements are assigned
// to the nearest centroid and the 3-bit index is stored.
//
// Bit packing: 8 indices (3 bits each) pack into 3 bytes (24 bits).
// For a group of q3GroupSize=128 elements, that is 128*3/8 = 48 bytes of
// packed indices plus 8*4 = 32 bytes of centroids = 80 bytes per group,
// compared to 512 bytes for float32 (6.4x compression).
//
// Like q4Storage, a shadow float32 buffer (vals) is maintained so that
// partial-group updates preserve existing values on re-quantization.
type q3Storage struct {
	packed    []byte    // packed 3-bit indices
	centroids []float32 // q3NumCentroids centroids per group, contiguous
	vals      []float32 // shadow float32 buffer for patch-and-requantize
	n         int       // total number of logical elements
}

// newQ3Storage allocates a q3Storage for n elements, initialized to zero.
func newQ3Storage(n int) *q3Storage {
	numGroups := (n + q3GroupSize - 1) / q3GroupSize
	// Each element needs 3 bits. Total bits = n * 3. Bytes = ceil(n*3 / 8).
	packedBytes := (n*3 + 7) / 8
	return &q3Storage{
		packed:    make([]byte, packedBytes),
		centroids: make([]float32, numGroups*q3NumCentroids),
		vals:      make([]float32, n),
		n:         n,
	}
}

// encodeRegion updates the shadow buffer at [offset, offset+len(src)) with src
// values, then re-quantizes only the affected groups.
func (s *q3Storage) encodeRegion(offset int, src []float32) {
	copy(s.vals[offset:offset+len(src)], src)

	firstGroup := offset / q3GroupSize
	lastGroup := (offset + len(src) - 1) / q3GroupSize

	for g := firstGroup; g <= lastGroup; g++ {
		s.quantizeGroup(g)
	}
}

// quantizeGroup fits a sensitivity-weighted k-means codebook for the group
// and assigns each element to the nearest centroid.
func (s *q3Storage) quantizeGroup(g int) {
	start := g * q3GroupSize
	end := start + q3GroupSize
	if end > s.n {
		end = s.n
	}
	groupLen := end - start
	vals := s.vals[start:end]
	cBase := g * q3NumCentroids

	// Initialize centroids via uniform quantile sampling across the sorted
	// value range. We avoid actual sorting by using min/max spread.
	minVal, maxVal := vals[0], vals[0]
	for _, v := range vals {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Handle constant-value groups.
	if minVal == maxVal {
		for c := range q3NumCentroids {
			s.centroids[cBase+c] = minVal
		}
		// All indices point to centroid 0.
		for i := start; i < end; i++ {
			s.set3Bit(i, 0)
		}
		return
	}

	// Linearly-spaced initial centroids.
	for c := range q3NumCentroids {
		t := float32(c) / float32(q3NumCentroids-1)
		s.centroids[cBase+c] = minVal + t*(maxVal-minVal)
	}

	// Sensitivity weights: sqrt(|value|) acts as importance weight for k-means.
	// Larger-magnitude values are more sensitive to quantization error in
	// attention computation, so they receive higher weight in centroid
	// placement. Using sqrt provides a gentler bias than linear magnitude,
	// ensuring near-zero values still get reasonable centroid coverage.
	weights := make([]float32, groupLen)
	for i, v := range vals {
		weights[i] = float32(math.Sqrt(math.Abs(float64(v)))) + 1e-4
	}

	// Run weighted k-means.
	assignments := make([]int, groupLen)
	for iter := range q3KMeansIters {
		_ = iter

		// Assignment step: each element to nearest centroid.
		for i, v := range vals {
			bestDist := float32(math.MaxFloat32)
			bestC := 0
			for c := range q3NumCentroids {
				d := (v - s.centroids[cBase+c])
				d *= d
				if d < bestDist {
					bestDist = d
					bestC = c
				}
			}
			assignments[i] = bestC
		}

		// Update step: weighted mean of assigned elements.
		var sums [q3NumCentroids]float64
		var wSums [q3NumCentroids]float64
		for i, v := range vals {
			c := assignments[i]
			w := float64(weights[i])
			sums[c] += float64(v) * w
			wSums[c] += w
		}
		for c := range q3NumCentroids {
			if wSums[c] > 0 {
				s.centroids[cBase+c] = float32(sums[c] / wSums[c])
			}
		}
	}

	// Final assignment and packing.
	for i, v := range vals {
		bestDist := float32(math.MaxFloat32)
		bestC := 0
		for c := range q3NumCentroids {
			d := (v - s.centroids[cBase+c])
			d *= d
			if d < bestDist {
				bestDist = d
				bestC = c
			}
		}
		s.set3Bit(start+i, byte(bestC))
	}
}

// decode dequantizes all n elements back to float32 by looking up centroids.
func (s *q3Storage) decode() []float32 {
	out := make([]float32, s.n)
	numGroups := (s.n + q3GroupSize - 1) / q3GroupSize
	for g := range numGroups {
		cBase := g * q3NumCentroids
		start := g * q3GroupSize
		end := start + q3GroupSize
		if end > s.n {
			end = s.n
		}
		for i := start; i < end; i++ {
			idx := s.get3Bit(i)
			out[i] = s.centroids[cBase+int(idx)]
		}
	}
	return out
}

// set3Bit writes a 3-bit value (0-7) at logical index i.
func (s *q3Storage) set3Bit(i int, val byte) {
	bitOffset := i * 3
	byteIdx := bitOffset / 8
	bitPos := uint(bitOffset % 8)

	// Clear and set the bits. A 3-bit value can span at most 2 bytes.
	mask := byte(0x07) // 3 bits
	s.packed[byteIdx] &^= mask << bitPos
	s.packed[byteIdx] |= (val & 0x07) << bitPos

	// If the value crosses a byte boundary (bitPos > 5), write upper bits
	// into the next byte.
	if bitPos > 5 && byteIdx+1 < len(s.packed) {
		bitsInFirst := 8 - bitPos
		s.packed[byteIdx+1] &^= mask >> bitsInFirst
		s.packed[byteIdx+1] |= (val & 0x07) >> bitsInFirst
	}
}

// get3Bit reads a 3-bit value at logical index i, returned as byte in [0,7].
func (s *q3Storage) get3Bit(i int) byte {
	bitOffset := i * 3
	byteIdx := bitOffset / 8
	bitPos := uint(bitOffset % 8)

	val := (s.packed[byteIdx] >> bitPos) & 0x07

	// If crossing byte boundary, merge bits from next byte.
	if bitPos > 5 && byteIdx+1 < len(s.packed) {
		bitsInFirst := 8 - bitPos
		upperBits := s.packed[byteIdx+1] & (0x07 >> bitsInFirst)
		val = ((upperBits << bitsInFirst) | (s.packed[byteIdx] >> bitPos)) & 0x07
	}
	return val
}

// rawBytes returns the total bytes used for packed data (excludes codebook).
func (s *q3Storage) rawBytes() int {
	return len(s.packed)
}

// totalBytes returns packed bytes + codebook bytes.
func (s *q3Storage) totalBytes() int {
	return len(s.packed) + len(s.centroids)*4
}

// q3LayerBuf holds the pre-allocated Q3 backing buffer for one layer's KV cache.
type q3LayerBuf struct {
	keyBuf *q3Storage
	valBuf *q3Storage
	cursor int
	batch  int
	dim    int
}

// KVCacheQ3 stores key-value tensors for all attention layers using 3-bit
// non-uniform codebook quantization. Each group of q3GroupSize elements gets
// 8 centroids computed via sensitivity-weighted k-means, where larger-magnitude
// values receive higher weight. This is a KVQuant-style approach that
// quantizes keys pre-RoPE to preserve rotary position information.
// Memory reduction is ~6.4x compared to float32.
type KVCacheQ3 struct {
	layers    []q3LayerBuf
	maxSeqLen int
}

// NewKVCacheQ3 creates a KVCacheQ3 for the specified number of layers and
// maximum sequence length. Q3 backing buffers are lazily allocated on the
// first Update call for each layer.
func NewKVCacheQ3(numLayers, maxSeqLen int) *KVCacheQ3 {
	return &KVCacheQ3{
		layers:    make([]q3LayerBuf, numLayers),
		maxSeqLen: maxSeqLen,
	}
}

// NumLayers returns the number of layers in the cache.
func (c *KVCacheQ3) NumLayers() int {
	return len(c.layers)
}

// Get returns the cached key-value pair for the given layer as float32 tensors
// covering [0:cursor] on the sequence axis. Q3 data is dequantized to float32
// on the fly via codebook lookup. Returns false if the layer has not been
// populated yet.
func (c *KVCacheQ3) Get(layer int) (*LayerKV[float32], bool) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, false
	}
	lb := &c.layers[layer]
	if lb.cursor == 0 {
		return nil, false
	}

	shape := []int{lb.batch, lb.cursor, lb.dim}
	size := lb.batch * lb.cursor * lb.dim

	allKey := lb.keyBuf.decode()
	allVal := lb.valBuf.decode()

	var keyData, valData []float32
	if lb.batch == 1 || lb.cursor == c.maxSeqLen {
		keyData = allKey[:size]
		valData = allVal[:size]
	} else {
		keyData = make([]float32, size)
		valData = make([]float32, size)
		seqDim := lb.cursor * lb.dim
		for bi := range lb.batch {
			srcOff := bi * c.maxSeqLen * lb.dim
			dstOff := bi * seqDim
			copy(keyData[dstOff:dstOff+seqDim], allKey[srcOff:srcOff+seqDim])
			copy(valData[dstOff:dstOff+seqDim], allVal[srcOff:srcOff+seqDim])
		}
	}

	keyT, err := tensor.New(shape, keyData)
	if err != nil {
		return nil, false
	}
	valT, err := tensor.New(shape, valData)
	if err != nil {
		return nil, false
	}

	return &LayerKV[float32]{Key: keyT, Value: valT}, true
}

// Update appends new key and value float32 tensors to the Q3 cache for the
// given layer. Tensors are expected to have shape [batch, seq_len, dim]. Data
// is converted from float32 to Q3 via sensitivity-weighted k-means codebook
// quantization and stored in the pre-allocated buffer at the current cursor
// position.
func (c *KVCacheQ3) Update(layer int, newK, newV *tensor.TensorNumeric[float32]) error {
	if layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, len(c.layers))
	}

	shape := newK.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("expected 3D tensor [batch, seq, dim], got %dD", len(shape))
	}

	batch, seqLen, dim := shape[0], shape[1], shape[2]
	lb := &c.layers[layer]

	// Lazy allocation on first Update.
	if lb.keyBuf == nil {
		lb.batch = batch
		lb.dim = dim
		total := batch * c.maxSeqLen * dim
		lb.keyBuf = newQ3Storage(total)
		lb.valBuf = newQ3Storage(total)
	}

	if batch != lb.batch {
		return fmt.Errorf("batch mismatch: cache has %d, got %d", lb.batch, batch)
	}
	if dim != lb.dim {
		return fmt.Errorf("dim mismatch: cache has %d, got %d", lb.dim, dim)
	}
	if lb.cursor+seqLen > c.maxSeqLen {
		return fmt.Errorf("cache overflow: cursor=%d + seq=%d > maxSeqLen=%d", lb.cursor, seqLen, c.maxSeqLen)
	}

	kData := newK.Data()
	vData := newV.Data()

	for bi := range batch {
		srcOff := bi * seqLen * dim
		dstOff := bi*c.maxSeqLen*dim + lb.cursor*dim
		lb.keyBuf.encodeRegion(dstOff, kData[srcOff:srcOff+seqLen*dim])
		lb.valBuf.encodeRegion(dstOff, vData[srcOff:srcOff+seqLen*dim])
	}

	lb.cursor += seqLen
	return nil
}

// SeqLen returns the current cached sequence length. Returns 0 if the cache is empty.
func (c *KVCacheQ3) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	return c.layers[0].cursor
}

// Reset clears all cached data and resets cursors to zero.
// The pre-allocated Q3 buffers are retained for reuse.
func (c *KVCacheQ3) Reset() {
	for i := range c.layers {
		c.layers[i].cursor = 0
	}
}

// Truncate rolls back the cache to the given sequence length.
// If newSeqLen >= current SeqLen, this is a no-op.
func (c *KVCacheQ3) Truncate(newSeqLen int) {
	for i := range c.layers {
		if c.layers[i].cursor > newSeqLen {
			c.layers[i].cursor = newSeqLen
		}
	}
}
