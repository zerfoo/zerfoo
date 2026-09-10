package tabular

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/model/dsl"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// ClassifierConfig describes a numeric classifier. No hidden layers selects a
// linear classifier; hidden layers use ReLU. Labels are ordered by class ID.
// This API is separate from the legacy three-direction Model API.
type ClassifierConfig struct {
	Definition *dsl.Definition `json:"definition,omitempty"`
	InputDim   int             `json:"input_dim"`
	ClassCount int             `json:"class_count"`
	Labels     []string        `json:"labels"`
	HiddenDims []int           `json:"hidden_dims,omitempty"`
	Seed       uint64          `json:"seed"`
}

// Prediction contains every class probability in the persisted label order.
type Prediction struct {
	ClassID       int       `json:"class_id"`
	Label         string    `json:"label"`
	Probabilities []float64 `json:"probabilities"`
}

// Classifier is a linear or ReLU MLP numeric classifier. Tensor operations use
// the supplied engine. Model, Train, and Direction retain their legacy API.
// DSL-backed classifiers require sequential calls. Legacy classifiers require
// an engine supporting concurrent inference for concurrent calls.
type Classifier[T tensor.Float] struct {
	compiled *dsl.Executable
	config   ClassifierConfig
	engine   compute.Engine[T]
	params   []*graph.Parameter[T] // unique DSL parameters, or alternating legacy weights and biases
}

// maxClassifierParameters bounds allocations from an untrusted configuration.
// The reference recipes are far below this 16-million-parameter limit.
const maxClassifierParameters = 1 << 24

// NewClassifier constructs a reproducibly initialized classifier. The same seed
// and configuration produce the same weights without changing global RNG state.
func NewClassifier[T tensor.Float](config ClassifierConfig, engine compute.Engine[T]) (*Classifier[T], error) {
	if engine == nil {
		return nil, fmt.Errorf("tabular: classifier: engine is required")
	}
	if err := validateClassifierConfig(config); err != nil {
		return nil, err
	}
	c := &Classifier[T]{config: cloneClassifierConfig(config), engine: engine}
	if config.Definition != nil {
		typed, ok := any(engine).(compute.Engine[float32])
		if !ok {
			return nil, fmt.Errorf("tabular: DSL execution currently requires float32")
		}
		compiled, err := dsl.Compile(context.Background(), *config.Definition, typed, config.Seed)
		if err != nil {
			return nil, err
		}
		c.compiled = compiled
		canonical := compiled.Definition()
		c.config.Definition = &canonical
		for _, p := range compiled.Parameters() {
			parameter, ok := any(p).(*graph.Parameter[T])
			if !ok {
				return nil, fmt.Errorf("tabular: DSL parameter dtype mismatch")
			}
			c.params = append(c.params, parameter)
		}
		return c, nil
	}

	rng := rand.New(rand.NewPCG(config.Seed, config.Seed^0x9e3779b97f4a7c15))
	dims := append([]int{config.InputDim}, config.HiddenDims...)
	dims = append(dims, config.ClassCount)
	for i := 0; i+1 < len(dims); i++ {
		name := fmt.Sprintf("layer%d", i)
		if i == len(dims)-2 {
			name = "head"
		}
		weights := make([]T, dims[i]*dims[i+1])
		scale := math.Sqrt(2 / float64(dims[i]))
		for j := range weights {
			weights[j] = T(rng.NormFloat64() * scale)
		}
		if err := c.addParameter(name+".weights", []int{dims[i], dims[i+1]}, weights); err != nil {
			return nil, err
		}
		if err := c.addParameter(name+".biases", []int{1, dims[i+1]}, make([]T, dims[i+1])); err != nil {
			return nil, err
		}
	}
	return c, nil
}

func validateClassifierConfig(config ClassifierConfig) error {
	if config.InputDim <= 0 || config.ClassCount < 2 {
		return fmt.Errorf("tabular: classifier: input_dim must be positive and class_count must be at least 2")
	}
	if len(config.Labels) != config.ClassCount {
		return fmt.Errorf("tabular: classifier: got %d labels for %d classes", len(config.Labels), config.ClassCount)
	}
	seen := make(map[string]bool, len(config.Labels))
	for i, label := range config.Labels {
		if strings.TrimSpace(label) == "" || !utf8.ValidString(label) || seen[label] {
			return fmt.Errorf("tabular: classifier: label %d must be nonempty, valid UTF-8 and unique", i)
		}
		seen[label] = true
	}
	dims := append([]int{config.InputDim}, config.HiddenDims...)
	dims = append(dims, config.ClassCount)
	total := 0
	for i := 0; i+1 < len(dims); i++ {
		in, out := dims[i], dims[i+1]
		if in <= 0 || out <= 0 || out > maxClassifierParameters || in > (maxClassifierParameters-out)/out {
			return fmt.Errorf("tabular: classifier: invalid or oversized layer %d dimensions [%d,%d]", i, in, out)
		}
		count := in*out + out
		if total > maxClassifierParameters-count {
			return fmt.Errorf("tabular: classifier: exceeds %d parameters", maxClassifierParameters)
		}
		total += count
	}
	if config.Definition != nil {
		return validateClassifierDefinition(config)
	}
	return nil
}

func cloneClassifierConfig(config ClassifierConfig) ClassifierConfig {
	if config.Definition != nil {
		copy := dsl.CloneDefinition(*config.Definition)
		config.Definition = &copy
	}
	config.Labels = slices.Clone(config.Labels)
	config.HiddenDims = slices.Clone(config.HiddenDims)
	return config
}

// Config returns a copy; changing it cannot mutate model topology or labels.
func (c *Classifier[T]) Config() ClassifierConfig { return cloneClassifierConfig(c.config) }

func (c *Classifier[T]) addParameter(name string, shape []int, data []T) error {
	value, err := tensor.New[T](shape, data)
	if err != nil {
		return fmt.Errorf("tabular: classifier: tensor %s: %w", name, err)
	}
	param, err := graph.NewParameter[T](name, value, tensor.New[T])
	if err != nil {
		return fmt.Errorf("tabular: classifier: parameter %s: %w", name, err)
	}
	c.params = append(c.params, param)
	return nil
}

func (c *Classifier[T]) inputTensor(ctx context.Context, rows [][]float64) (*tensor.TensorNumeric[T], error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	width := max(c.config.InputDim, c.config.ClassCount)
	for _, hidden := range c.config.HiddenDims {
		width = max(width, hidden)
	}
	if len(rows) == 0 || len(rows) > maxClassifierParameters/width {
		return nil, fmt.Errorf("tabular: classifier: batch is empty or exceeds %d input values", maxClassifierParameters)
	}
	data := make([]T, len(rows)*c.config.InputDim)
	for i, row := range rows {
		if len(row) != c.config.InputDim {
			return nil, fmt.Errorf("tabular: classifier: row %d has %d features, want %d", i, len(row), c.config.InputDim)
		}
		for j, value := range row {
			converted := T(value)
			if math.IsNaN(value) || math.IsInf(value, 0) || math.IsInf(float64(converted), 0) {
				return nil, fmt.Errorf("tabular: classifier: row %d feature %d must be finite and representable", i, j)
			}
			data[i*c.config.InputDim+j] = converted
		}
	}
	input, err := tensor.New[T]([]int{len(rows), c.config.InputDim}, data)
	if err != nil {
		return nil, fmt.Errorf("tabular: classifier: input: %w", err)
	}
	return input, nil
}

func (c *Classifier[T]) forward(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if c.compiled != nil {
		definition := c.config.Definition
		typed, ok := any(input).(*tensor.TensorNumeric[float32])
		if !ok {
			return nil, fmt.Errorf("tabular: DSL input dtype mismatch")
		}
		outputs, err := c.compiled.Forward(ctx, map[string]*tensor.TensorNumeric[float32]{definition.Inputs[0].Name: typed})
		if err != nil {
			return nil, err
		}
		output, ok := any(outputs[definition.Outputs[0].Name]).(*tensor.TensorNumeric[T])
		if !ok {
			return nil, fmt.Errorf("tabular: DSL output dtype mismatch")
		}
		return output, nil
	}

	x := input
	for i := 0; i < len(c.params); i += 2 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		weight, err := c.engine.Transpose(ctx, c.params[i].Value, []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("tabular: classifier: transpose: %w", err)
		}
		x, err = functional.Linear(ctx, c.engine, x, weight, c.params[i+1].Value)
		if err != nil {
			return nil, fmt.Errorf("tabular: classifier: linear layer %d: %w", i/2, err)
		}
		if i+2 < len(c.params) {
			x, err = functional.ReLU(ctx, c.engine, c.engine.Ops(), x)
			if err != nil {
				return nil, fmt.Errorf("tabular: classifier: ReLU layer %d: %w", i/2, err)
			}
		}
	}
	for _, value := range x.Data() {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("tabular: classifier: nonfinite logits")
		}
	}
	return x, nil
}

// Predict classifies one numeric feature vector.
func (c *Classifier[T]) Predict(ctx context.Context, features []float64) (Prediction, error) {
	predictions, err := c.PredictBatch(ctx, [][]float64{features})
	if err != nil {
		return Prediction{}, err
	}
	return predictions[0], nil
}

// PredictBatch rejects malformed/nonfinite inputs and returns one prediction per
// row. Ties choose the lowest class ID. Input rows are already preprocessed.
func (c *Classifier[T]) PredictBatch(ctx context.Context, rows [][]float64) ([]Prediction, error) {
	input, err := c.inputTensor(ctx, rows)
	if err != nil {
		return nil, err
	}
	logits, err := c.forward(ctx, input)
	if err != nil {
		return nil, err
	}
	probabilities, err := c.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return nil, fmt.Errorf("tabular: classifier: softmax: %w", err)
	}
	data := probabilities.Data()
	predictions := make([]Prediction, len(rows))
	for i := range predictions {
		values := make([]float64, c.config.ClassCount)
		best, sum := 0, 0.0
		for j := range values {
			value := float64(data[i*c.config.ClassCount+j])
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
				return nil, fmt.Errorf("tabular: classifier: invalid probability at row %d class %d", i, j)
			}
			values[j] = value
			sum += value
			if values[j] > values[best] {
				best = j
			}
		}
		if math.Abs(sum-1) > 1e-5 {
			return nil, fmt.Errorf("tabular: classifier: row %d probabilities sum to %g", i, sum)
		}
		predictions[i] = Prediction{ClassID: best, Label: c.config.Labels[best], Probabilities: values}
	}
	return predictions, nil
}

// Loss returns mean logit cross entropy for explicit class IDs. Unlike the
// clipped reporting metric, this training loss remains stable for large logits.
func (c *Classifier[T]) Loss(ctx context.Context, rows [][]float64, targets []int) (float64, error) {
	labels, err := c.targetTensor(len(rows), targets)
	if err != nil {
		return 0, err
	}
	input, err := c.inputTensor(ctx, rows)
	if err != nil {
		return 0, err
	}
	logits, err := c.forward(ctx, input)
	if err != nil {
		return 0, err
	}
	objective := loss.NewCrossEntropyLoss[T](c.engine)
	result, err := objective.Forward(ctx, logits, labels)
	if err != nil {
		return 0, fmt.Errorf("tabular: classifier: loss: %w", err)
	}
	value := float64(result.Data()[0])
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("tabular: classifier: nonfinite loss")
	}
	return value, nil
}

func (c *Classifier[T]) targetTensor(rows int, targets []int) (*tensor.TensorNumeric[T], error) {
	if rows == 0 || len(targets) != rows {
		return nil, fmt.Errorf("tabular: classifier: need one target per nonempty row")
	}
	values := make([]T, rows)
	for i, target := range targets {
		if target < 0 || target >= c.config.ClassCount {
			return nil, fmt.Errorf("tabular: classifier: target %d at row %d out of range", target, i)
		}
		values[i] = T(target)
	}
	labels, err := tensor.New[T]([]int{rows}, values)
	if err != nil {
		return nil, fmt.Errorf("tabular: classifier: targets: %w", err)
	}
	return labels, nil
}
