package generate

import (
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// q4GroupSize is the number of elements per quantization group.
// Each group shares a single float32 absmax scale factor.
const q4GroupSize = 128

// q4Storage packs float32 values into 4-bit integers with per-group scaling.
// Each element is quantized to the range [-8, 7] (signed 4-bit). Two elements
// are packed into a single byte (low nibble first). Each group of q4GroupSize
// elements has one float32 scale factor.
//
// To avoid corrupting previously quantized data when a new update touches a
// group, the storage maintains a shadow float32 buffer (vals) that is patched
// on each update. Only the groups that contain modified elements are
// re-quantized, ensuring existing values in those groups are preserved exactly.
type q4Storage struct {
	packed []byte    // ceil(n/2) bytes, two 4-bit values per byte
	scales []float32 // one scale per group of q4GroupSize elements
	vals   []float32 // shadow float32 buffer for patch-and-requantize
	n      int       // total number of logical elements
}

// newQ4Storage allocates a q4Storage for n elements, initialized to zero.
func newQ4Storage(n int) *q4Storage {
	numGroups := (n + q4GroupSize - 1) / q4GroupSize
	return &q4Storage{
		packed: make([]byte, (n+1)/2),
		scales: make([]float32, numGroups),
		vals:   make([]float32, n),
		n:      n,
	}
}

// encodeRegion updates the shadow buffer at [offset, offset+len(src)) with src
// values, then re-quantizes only the affected groups.
func (s *q4Storage) encodeRegion(offset int, src []float32) {
	copy(s.vals[offset:offset+len(src)], src)

	// Determine which groups are affected.
	firstGroup := offset / q4GroupSize
	lastGroup := (offset + len(src) - 1) / q4GroupSize

	for g := firstGroup; g <= lastGroup; g++ {
		s.quantizeGroup(g)
	}
}

// quantizeGroup re-quantizes a single group from the shadow buffer.
func (s *q4Storage) quantizeGroup(g int) {
	start := g * q4GroupSize
	end := start + q4GroupSize
	if end > s.n {
		end = s.n
	}

	// Find absmax for this group.
	var absmax float32
	for i := start; i < end; i++ {
		a := float32(math.Abs(float64(s.vals[i])))
		if a > absmax {
			absmax = a
		}
	}

	// Scale: absmax / 7 (symmetric [-7, 7]).
	var scale float32
	if absmax > 0 {
		scale = absmax / 7.0
	}
	s.scales[g] = scale

	// Quantize and pack.
	var invScale float32
	if scale > 0 {
		invScale = 1.0 / scale
	}
	for i := start; i < end; i++ {
		q := int(math.Round(float64(s.vals[i] * invScale)))
		if q < -8 {
			q = -8
		} else if q > 7 {
			q = 7
		}
		s.setNibble(i, byte(q&0x0F))
	}
}

// decode dequantizes all n elements back to float32.
func (s *q4Storage) decode() []float32 {
	out := make([]float32, s.n)
	for g := range s.scales {
		scale := s.scales[g]
		start := g * q4GroupSize
		end := start + q4GroupSize
		if end > s.n {
			end = s.n
		}
		for i := start; i < end; i++ {
			q := s.getNibble(i)
			// Sign-extend 4-bit to int8.
			if q >= 8 {
				q -= 16
			}
			out[i] = float32(q) * scale
		}
	}
	return out
}

// setNibble writes a 4-bit value at logical index i.
func (s *q4Storage) setNibble(i int, val byte) {
	byteIdx := i / 2
	if i%2 == 0 {
		s.packed[byteIdx] = (s.packed[byteIdx] & 0xF0) | (val & 0x0F)
	} else {
		s.packed[byteIdx] = (s.packed[byteIdx] & 0x0F) | (val << 4)
	}
}

// getNibble reads a 4-bit value at logical index i, returned as int8 range [0,15].
func (s *q4Storage) getNibble(i int) int8 {
	byteIdx := i / 2
	if i%2 == 0 {
		return int8(s.packed[byteIdx] & 0x0F)
	}
	return int8(s.packed[byteIdx] >> 4)
}

// rawBytes returns the total bytes used for packed data (excludes scale storage).
func (s *q4Storage) rawBytes() int {
	return len(s.packed)
}

// totalBytes returns packed bytes + scale bytes.
func (s *q4Storage) totalBytes() int {
	return len(s.packed) + len(s.scales)*4
}

// q4LayerBuf holds the pre-allocated Q4 backing buffer for one layer's KV cache.
type q4LayerBuf struct {
	keyBuf *q4Storage
	valBuf *q4Storage
	cursor int
	batch  int
	dim    int
}

// KVCacheQ4 stores key-value tensors for all attention layers using 4-bit
// group quantization, reducing memory by ~8x compared to float32.
// On Update, float32 values are quantized to Q4 with per-group (group_size=128)
// absmax scaling. On Get, Q4 values are dequantized back to float32.
type KVCacheQ4 struct {
	layers    []q4LayerBuf
	maxSeqLen int
}

// NewKVCacheQ4 creates a KVCacheQ4 for the specified number of layers and
// maximum sequence length. Q4 backing buffers are lazily allocated on the
// first Update call for each layer.
func NewKVCacheQ4(numLayers, maxSeqLen int) *KVCacheQ4 {
	return &KVCacheQ4{
		layers:    make([]q4LayerBuf, numLayers),
		maxSeqLen: maxSeqLen,
	}
}

// NumLayers returns the number of layers in the cache.
func (c *KVCacheQ4) NumLayers() int {
	return len(c.layers)
}

// Get returns the cached key-value pair for the given layer as float32 tensors
// covering [0:cursor] on the sequence axis. Q4 data is dequantized to float32
// on the fly. Returns false if the layer has not been populated yet.
func (c *KVCacheQ4) Get(layer int) (*LayerKV[float32], bool) {
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

// Update appends new key and value float32 tensors to the Q4 cache for the
// given layer. Tensors are expected to have shape [batch, seq_len, dim]. Data
// is converted from float32 to Q4 and stored in the pre-allocated buffer
// at the current cursor position.
func (c *KVCacheQ4) Update(layer int, newK, newV *tensor.TensorNumeric[float32]) error {
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
		lb.keyBuf = newQ4Storage(total)
		lb.valBuf = newQ4Storage(total)
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

	// Patch the shadow buffer and re-quantize only affected groups.
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
func (c *KVCacheQ4) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	return c.layers[0].cursor
}

// Reset clears all cached data and resets cursors to zero.
// The pre-allocated Q4 buffers are retained for reuse.
func (c *KVCacheQ4) Reset() {
	for i := range c.layers {
		c.layers[i].cursor = 0
	}
}

// Truncate rolls back the cache to the given sequence length.
// If newSeqLen >= current SeqLen, this is a no-op.
func (c *KVCacheQ4) Truncate(newSeqLen int) {
	for i := range c.layers {
		if c.layers[i].cursor > newSeqLen {
			c.layers[i].cursor = newSeqLen
		}
	}
}
