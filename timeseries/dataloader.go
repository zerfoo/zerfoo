package timeseries

import (
	"math/rand/v2"

	"github.com/zerfoo/ztensor/tensor"
)

// DataLoader iterates over windowed time-series data in mini-batches,
// producing tensors suitable for training loops. It converts [][][]float64
// windows and []float64 labels into float32 tensors.
type DataLoader struct {
	windows   [][][]float64 // [nSamples][channels][inputLen]
	labels    []float64     // [nSamples * outputDim]
	batchSize int
	shuffle   bool
	outputDim int
	indices   []int
	pos       int // current position in indices
}

// NewDataLoader creates a DataLoader that yields mini-batches from the given
// windows and labels. windows has shape [nSamples][channels][inputLen] and
// labels has shape [nSamples * outputDim]. When shuffle is true, sample order
// is randomized using Fisher-Yates on each Reset.
func NewDataLoader(windows [][][]float64, labels []float64, batchSize int, shuffle bool) *DataLoader {
	n := len(windows)
	outputDim := 1
	if n > 0 {
		outputDim = len(labels) / n
	}

	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	dl := &DataLoader{
		windows:   windows,
		labels:    labels,
		batchSize: batchSize,
		shuffle:   shuffle,
		outputDim: outputDim,
		indices:   indices,
	}

	if shuffle {
		dl.shuffleIndices()
	}

	return dl
}

// Len returns the number of batches per epoch, including a possible
// partial final batch.
func (dl *DataLoader) Len() int {
	n := len(dl.windows)
	if n == 0 || dl.batchSize <= 0 {
		return 0
	}
	return (n + dl.batchSize - 1) / dl.batchSize
}

// Next returns the next batch of input and label tensors. The input tensor
// has shape [batchSize, channels, inputLen] and the label tensor has shape
// [batchSize, outputDim]. The final batch may have fewer than batchSize
// samples. ok is false when all batches have been consumed.
func (dl *DataLoader) Next() (inputBatch *tensor.TensorNumeric[float32], labelBatch *tensor.TensorNumeric[float32], ok bool) {
	n := len(dl.windows)
	if dl.pos >= n {
		return nil, nil, false
	}

	end := dl.pos + dl.batchSize
	if end > n {
		end = n
	}
	batchIndices := dl.indices[dl.pos:end]
	bs := len(batchIndices)
	dl.pos = end

	// Derive dimensions from the first sample.
	channels := len(dl.windows[0])
	inputLen := 0
	if channels > 0 {
		inputLen = len(dl.windows[0][0])
	}

	// Build input tensor data: [bs, channels, inputLen].
	inputData := make([]float32, bs*channels*inputLen)
	for i, idx := range batchIndices {
		for c := 0; c < channels; c++ {
			for t := 0; t < inputLen; t++ {
				inputData[i*channels*inputLen+c*inputLen+t] = float32(dl.windows[idx][c][t])
			}
		}
	}

	// Build label tensor data: [bs, outputDim].
	labelData := make([]float32, bs*dl.outputDim)
	for i, idx := range batchIndices {
		for d := 0; d < dl.outputDim; d++ {
			labelData[i*dl.outputDim+d] = float32(dl.labels[idx*dl.outputDim+d])
		}
	}

	inputT, err := tensor.New[float32]([]int{bs, channels, inputLen}, inputData)
	if err != nil {
		return nil, nil, false
	}

	labelT, err := tensor.New[float32]([]int{bs, dl.outputDim}, labelData)
	if err != nil {
		return nil, nil, false
	}

	return inputT, labelT, true
}

// NextIndices returns the indices of the next batch of samples without
// constructing tensors. ok is false when all batches have been consumed.
func (dl *DataLoader) NextIndices() (indices []int, ok bool) {
	n := len(dl.windows)
	if dl.pos >= n {
		return nil, false
	}

	end := dl.pos + dl.batchSize
	if end > n {
		end = n
	}
	batch := make([]int, end-dl.pos)
	copy(batch, dl.indices[dl.pos:end])
	dl.pos = end
	return batch, true
}

// Reset reshuffles (if enabled) and restarts iteration from the beginning.
func (dl *DataLoader) Reset() {
	dl.pos = 0
	if dl.shuffle {
		dl.shuffleIndices()
	}
}

// shuffleIndices performs a Fisher-Yates shuffle on the index slice.
func (dl *DataLoader) shuffleIndices() {
	for i := len(dl.indices) - 1; i > 0; i-- {
		j := rand.IntN(i + 1)
		dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
	}
}
