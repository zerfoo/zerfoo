package timeseries

import (
	"testing"
)

// makeWindows creates synthetic windows [nSamples][nChannels][inputLen]
// and labels [nSamples * outputDim] with deterministic values.
func makeWindows(nSamples, nChannels, inputLen, outputDim int) ([][][]float64, []float64) {
	windows := make([][][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, nChannels)
		for c := 0; c < nChannels; c++ {
			windows[i][c] = make([]float64, inputLen)
			for t := 0; t < inputLen; t++ {
				windows[i][c][t] = float64(i*1000 + c*100 + t)
			}
		}
	}
	labels := make([]float64, nSamples*outputDim)
	for i := range labels {
		labels[i] = float64(i)
	}
	return windows, labels
}

func TestDataLoader_AllSamplesVisited(t *testing.T) {
	tests := []struct {
		name      string
		nSamples  int
		batchSize int
	}{
		{"exact_division", 10, 5},
		{"partial_batch", 10, 3},
		{"single_batch", 4, 10},
		{"batch_of_one", 5, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			windows, labels := makeWindows(tt.nSamples, 2, 8, 1)
			dl := NewDataLoader(windows, labels, tt.batchSize, false)

			seen := make(map[int]bool)
			totalSamples := 0
			for {
				inputT, _, ok := dl.Next()
				if !ok {
					break
				}
				shape := inputT.Shape()
				bs := shape[0]
				totalSamples += bs

				// Recover original sample index from the deterministic encoding.
				data := inputT.Data()
				channels := shape[1]
				inputLen := shape[2]
				for i := 0; i < bs; i++ {
					// First element of first channel encodes i*1000.
					val := int(data[i*channels*inputLen])
					sampleIdx := val / 1000
					seen[sampleIdx] = true
				}
			}

			if totalSamples != tt.nSamples {
				t.Errorf("total samples = %d, want %d", totalSamples, tt.nSamples)
			}
			if len(seen) != tt.nSamples {
				t.Errorf("unique samples = %d, want %d", len(seen), tt.nSamples)
			}
		})
	}
}

func TestDataLoader_ShuffleDifferentOrder(t *testing.T) {
	nSamples := 20
	windows, labels := makeWindows(nSamples, 1, 4, 1)

	// Collect order across multiple resets. With 20 samples the probability
	// of identical random permutations is negligible.
	collectOrder := func(dl *DataLoader) []int {
		var order []int
		for {
			inputT, _, ok := dl.Next()
			if !ok {
				break
			}
			data := inputT.Data()
			shape := inputT.Shape()
			bs := shape[0]
			inputLen := shape[2]
			for i := 0; i < bs; i++ {
				order = append(order, int(data[i*inputLen])/1000)
			}
		}
		return order
	}

	dl := NewDataLoader(windows, labels, 5, true)
	order1 := collectOrder(dl)
	dl.Reset()
	order2 := collectOrder(dl)

	if len(order1) != nSamples || len(order2) != nSamples {
		t.Fatalf("order lengths = %d, %d; want %d", len(order1), len(order2), nSamples)
	}

	same := true
	for i := range order1 {
		if order1[i] != order2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("shuffle produced identical order across two resets")
	}
}

func TestDataLoader_PartialBatch(t *testing.T) {
	tests := []struct {
		name          string
		nSamples      int
		batchSize     int
		wantBatches   int
		wantLastBatch int
	}{
		{"remainder_1", 7, 3, 3, 1},
		{"remainder_2", 8, 3, 3, 2},
		{"exact", 6, 3, 2, 3},
		{"all_partial", 2, 5, 1, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			windows, labels := makeWindows(tt.nSamples, 1, 4, 1)
			dl := NewDataLoader(windows, labels, tt.batchSize, false)

			if dl.Len() != tt.wantBatches {
				t.Errorf("Len() = %d, want %d", dl.Len(), tt.wantBatches)
			}

			var lastBS int
			count := 0
			for {
				inputT, _, ok := dl.Next()
				if !ok {
					break
				}
				lastBS = inputT.Shape()[0]
				count++
			}

			if count != tt.wantBatches {
				t.Errorf("batch count = %d, want %d", count, tt.wantBatches)
			}
			if lastBS != tt.wantLastBatch {
				t.Errorf("last batch size = %d, want %d", lastBS, tt.wantLastBatch)
			}
		})
	}
}

func TestDataLoader_ShapeCorrectness(t *testing.T) {
	tests := []struct {
		name      string
		nChannels int
		inputLen  int
		outputDim int
	}{
		{"1ch_8len_1out", 1, 8, 1},
		{"3ch_16len_1out", 3, 16, 1},
		{"5ch_4len_3out", 5, 4, 3},
		{"2ch_32len_2out", 2, 32, 2},
		{"20ch_8len_1out", 20, 8, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nSamples := 6
			batchSize := 4
			windows, labels := makeWindows(nSamples, tt.nChannels, tt.inputLen, tt.outputDim)
			dl := NewDataLoader(windows, labels, batchSize, false)

			// First batch should be full.
			inputT, labelT, ok := dl.Next()
			if !ok {
				t.Fatal("expected first batch")
			}

			inputShape := inputT.Shape()
			wantInputShape := []int{batchSize, tt.nChannels, tt.inputLen}
			if !equalInts(inputShape, wantInputShape) {
				t.Errorf("input shape = %v, want %v", inputShape, wantInputShape)
			}

			labelShape := labelT.Shape()
			wantLabelShape := []int{batchSize, tt.outputDim}
			if !equalInts(labelShape, wantLabelShape) {
				t.Errorf("label shape = %v, want %v", labelShape, wantLabelShape)
			}

			// Second (partial) batch.
			inputT, labelT, ok = dl.Next()
			if !ok {
				t.Fatal("expected second batch")
			}

			partialBS := nSamples - batchSize
			inputShape = inputT.Shape()
			if inputShape[0] != partialBS {
				t.Errorf("partial input batch dim = %d, want %d", inputShape[0], partialBS)
			}
			labelShape = labelT.Shape()
			if labelShape[0] != partialBS {
				t.Errorf("partial label batch dim = %d, want %d", labelShape[0], partialBS)
			}
		})
	}
}

func TestDataLoader_ResetRestartsIteration(t *testing.T) {
	windows, labels := makeWindows(5, 2, 4, 1)
	dl := NewDataLoader(windows, labels, 3, false)

	// Drain all batches.
	count := 0
	for {
		_, _, ok := dl.Next()
		if !ok {
			break
		}
		count++
	}
	if count != 2 {
		t.Fatalf("first pass: got %d batches, want 2", count)
	}

	// After exhaustion, Next returns false.
	_, _, ok := dl.Next()
	if ok {
		t.Error("expected ok=false after exhaustion")
	}

	// Reset and drain again.
	dl.Reset()
	count = 0
	for {
		_, _, ok := dl.Next()
		if !ok {
			break
		}
		count++
	}
	if count != 2 {
		t.Errorf("second pass after Reset: got %d batches, want 2", count)
	}
}

func TestDataLoader_LenEmpty(t *testing.T) {
	dl := NewDataLoader(nil, nil, 4, false)
	if dl.Len() != 0 {
		t.Errorf("Len() = %d, want 0 for empty loader", dl.Len())
	}
}

func TestDataLoader_DataValues(t *testing.T) {
	nSamples := 4
	channels := 2
	inputLen := 3
	outputDim := 2
	windows, labels := makeWindows(nSamples, channels, inputLen, outputDim)
	dl := NewDataLoader(windows, labels, nSamples, false)

	inputT, labelT, ok := dl.Next()
	if !ok {
		t.Fatal("expected batch")
	}

	inputData := inputT.Data()
	for i := 0; i < nSamples; i++ {
		for c := 0; c < channels; c++ {
			for tt := 0; tt < inputLen; tt++ {
				got := inputData[i*channels*inputLen+c*inputLen+tt]
				want := float32(windows[i][c][tt])
				if got != want {
					t.Errorf("input[%d][%d][%d] = %v, want %v", i, c, tt, got, want)
				}
			}
		}
	}

	labelData := labelT.Data()
	for i := 0; i < nSamples; i++ {
		for d := 0; d < outputDim; d++ {
			got := labelData[i*outputDim+d]
			want := float32(labels[i*outputDim+d])
			if got != want {
				t.Errorf("label[%d][%d] = %v, want %v", i, d, got, want)
			}
		}
	}
}

func TestDataLoader_NextIndices(t *testing.T) {
	nSamples := 10
	windows, labels := makeWindows(nSamples, 1, 4, 1)

	tests := []struct {
		name      string
		batchSize int
		wantCount int // expected number of batches
	}{
		{"exact_division", 5, 2},
		{"partial_batch", 3, 4},
		{"single_batch", 10, 1},
		{"batch_of_one", 1, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dl := NewDataLoader(windows, labels, tt.batchSize, false)

			seen := make(map[int]bool)
			batchCount := 0
			for {
				indices, ok := dl.NextIndices()
				if !ok {
					break
				}
				batchCount++
				for _, idx := range indices {
					if idx < 0 || idx >= nSamples {
						t.Errorf("index %d out of range [0, %d)", idx, nSamples)
					}
					if seen[idx] {
						t.Errorf("duplicate index %d", idx)
					}
					seen[idx] = true
				}
			}

			if batchCount != tt.wantCount {
				t.Errorf("got %d batches, want %d", batchCount, tt.wantCount)
			}
			if len(seen) != nSamples {
				t.Errorf("visited %d samples, want %d", len(seen), nSamples)
			}
		})
	}
}

func TestDataLoader_NextIndicesDoesNotAlias(t *testing.T) {
	windows, labels := makeWindows(6, 1, 4, 1)
	dl := NewDataLoader(windows, labels, 3, false)

	first, ok := dl.NextIndices()
	if !ok {
		t.Fatal("expected first batch")
	}
	firstCopy := make([]int, len(first))
	copy(firstCopy, first)

	second, ok := dl.NextIndices()
	if !ok {
		t.Fatal("expected second batch")
	}

	// Mutating second should not affect first.
	second[0] = -999
	if first[0] == -999 {
		t.Error("NextIndices returned aliased slices")
	}
	if !equalInts(first, firstCopy) {
		t.Error("first batch was mutated by second call")
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
