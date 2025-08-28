package training

import (
	"math/rand/v2"
	"sort"

	"github.com/zerfoo/zerfoo/data"
)

// EraSequencer generates sequences of consecutive eras for curriculum learning.
type EraSequencer struct {
	maxSeqLen int
	rand      *rand.Rand
}

// NewEraSequencer creates a new era sequencer with the given maximum sequence length.
func NewEraSequencer(maxSeqLen int) *EraSequencer {
	return &EraSequencer{
		maxSeqLen: maxSeqLen,
		rand:      rand.New(rand.NewPCG(0, 0)), // Use PCG with default seed
	}
}

// GenerateSequences generates random sequences of consecutive eras from the dataset.
// Each sequence contains between 1 and maxSeqLen consecutive eras.
func (s *EraSequencer) GenerateSequences(dataset *data.Dataset, numSequences int) []*data.Dataset {
	if len(dataset.Eras) == 0 {
		return []*data.Dataset{}
	}

	// Sort eras chronologically
	sortedEras := make([]data.EraData, len(dataset.Eras))
	copy(sortedEras, dataset.Eras)
	sort.Slice(sortedEras, func(i, j int) bool {
		return sortedEras[i].Era < sortedEras[j].Era
	})

	sequences := make([]*data.Dataset, numSequences)

	for i := 0; i < numSequences; i++ {
		// Pick a random sequence length between 1 and maxSeqLen
		seqLen := s.rand.IntN(s.maxSeqLen) + 1

		// Make sure sequence length doesn't exceed available eras
		if seqLen > len(sortedEras) {
			seqLen = len(sortedEras)
		}

		// Pick a random starting era that allows for the full sequence
		maxStartIdx := len(sortedEras) - seqLen
		if maxStartIdx < 0 {
			maxStartIdx = 0
		}

		startIdx := s.rand.IntN(maxStartIdx + 1)

		// Create sequence
		sequenceEras := make([]data.EraData, seqLen)
		copy(sequenceEras, sortedEras[startIdx:startIdx+seqLen])

		sequences[i] = &data.Dataset{
			Eras: sequenceEras,
		}
	}

	return sequences
}

// GenerateTrainValidationSplit splits the dataset into training and validation sets.
// The validation set contains the last 'validationEras' eras chronologically.
func (s *EraSequencer) GenerateTrainValidationSplit(dataset *data.Dataset, validationEras int) (*data.Dataset, *data.Dataset) {
	if len(dataset.Eras) == 0 {
		return &data.Dataset{Eras: []data.EraData{}}, &data.Dataset{Eras: []data.EraData{}}
	}

	// Sort eras chronologically
	sortedEras := make([]data.EraData, len(dataset.Eras))
	copy(sortedEras, dataset.Eras)
	sort.Slice(sortedEras, func(i, j int) bool {
		return sortedEras[i].Era < sortedEras[j].Era
	})

	// Ensure we don't request more validation eras than available
	if validationEras > len(sortedEras) {
		validationEras = len(sortedEras)
	}

	splitIdx := len(sortedEras) - validationEras
	if splitIdx < 0 {
		splitIdx = 0
	}

	trainEras := make([]data.EraData, splitIdx)
	copy(trainEras, sortedEras[:splitIdx])

	validEras := make([]data.EraData, validationEras)
	copy(validEras, sortedEras[splitIdx:])

	trainData := &data.Dataset{Eras: trainEras}
	validData := &data.Dataset{Eras: validEras}

	return trainData, validData
}

// SetSeed sets the random seed for reproducible sequence generation.
func (s *EraSequencer) SetSeed(seed1, seed2 uint64) {
	s.rand = rand.New(rand.NewPCG(seed1, seed2))
}
