package helpers

// Wire your real implementations here in a follow-up commit.
// Provide concrete types that satisfy these interfaces, then set Impl to them in init().

type ZerfooAPI interface {
	SetSeed(seed int)
	Tokenize(text string) ([]int, error)
	RefTokenize(text string) ([]int, error) // reference tokenizer
	Logits(prompt string, maxNewTokens int) ([]float32, error) // model under test
	RefLogits(prompt string, maxNewTokens int) ([]float32, error) // reference model
	DecodeGreedy(ids []int) (string, error)
	DecodeTopP(ids []int, p float64) (string, error)
	DeviceName() string // "cpu" or "gpu"
}

type NumericsAPI interface {
	// Forward computes y = f(x) and caches context for Backward.
	Forward(x []float64) ([]float64, interface{}, error)
	// Backward returns dy/dx given upstream gradient g.
	Backward(ctx interface{}, g []float64) ([]float64, error)
}

type PipelineAPI interface {
	EraSplits(eras []string, k int) ([][]int, error) // k folds of indices
	LeakageOK(eras []string, folds [][]int) bool
	Bag(preds [][]float64) []float64
	ClipToTarget(preds []float64, targetMean, targetStd float64) []float64
	Neutralize(preds []float64, exposures [][]float64, strength float64) []float64
	Turnover(prev, curr []float64) float64
}

type PerfAPI interface {
	InferLatency(batchSize int) (ms float64, err error)
}

var (
	ImplZerfoo  ZerfooAPI
	ImplNumerics NumericsAPI
	ImplPipeline PipelineAPI
	ImplPerf     PerfAPI
)
