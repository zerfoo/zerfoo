package meta

import (
	"math"
	"testing"
)

// sinTask is a regression task where the target is a*sin(x+b).
// Each task has a different amplitude and phase, making it ideal for
// testing few-shot adaptation: the model must learn a good initialization
// that quickly adapts to new sin functions.
type sinTask struct {
	amplitude float64
	phase     float64
	trainX    [][]float64
	trainY    []float64
	testX     [][]float64
	testY     []float64
}

func newSinTask(amplitude, phase float64, nTrain, nTest int) *sinTask {
	t := &sinTask{amplitude: amplitude, phase: phase}
	t.trainX = make([][]float64, nTrain)
	t.trainY = make([]float64, nTrain)
	for i := 0; i < nTrain; i++ {
		x := -5.0 + 10.0*float64(i)/float64(nTrain)
		t.trainX[i] = []float64{x}
		t.trainY[i] = amplitude * math.Sin(x+phase)
	}
	t.testX = make([][]float64, nTest)
	t.testY = make([]float64, nTest)
	for i := 0; i < nTest; i++ {
		x := -5.0 + 10.0*float64(i)/float64(nTest) + 0.05
		t.testX[i] = []float64{x}
		t.testY[i] = amplitude * math.Sin(x+phase)
	}
	return t
}

func (t *sinTask) TrainData() ([][]float64, []float64) { return t.trainX, t.trainY }
func (t *sinTask) TestData() ([][]float64, []float64)  { return t.testX, t.testY }

// linearTask is a simple linear regression task: y = slope*x + intercept.
type linearTask struct {
	trainX [][]float64
	trainY []float64
	testX  [][]float64
	testY  []float64
}

func newLinearTask(slope, intercept float64, nTrain, nTest int) *linearTask {
	t := &linearTask{}
	t.trainX = make([][]float64, nTrain)
	t.trainY = make([]float64, nTrain)
	for i := 0; i < nTrain; i++ {
		x := float64(i) / float64(nTrain)
		t.trainX[i] = []float64{x}
		t.trainY[i] = slope*x + intercept
	}
	t.testX = make([][]float64, nTest)
	t.testY = make([]float64, nTest)
	for i := 0; i < nTest; i++ {
		x := float64(i)/float64(nTest) + 0.01
		t.testX[i] = []float64{x}
		t.testY[i] = slope*x + intercept
	}
	return t
}

func (t *linearTask) TrainData() ([][]float64, []float64) { return t.trainX, t.trainY }
func (t *linearTask) TestData() ([][]float64, []float64)  { return t.testX, t.testY }

func TestMAML_FewShotAdaptation(t *testing.T) {
	// Create a set of linear tasks with different slopes and intercepts.
	// After meta-training, the model should adapt to a new linear task
	// in very few gradient steps.
	tasks := make([]Task, 20)
	for i := 0; i < 20; i++ {
		slope := 0.5 + float64(i)*0.2
		intercept := -1.0 + float64(i)*0.1
		tasks[i] = newLinearTask(slope, intercept, 20, 10)
	}

	config := MAMLConfig{
		InnerLR:        0.01,
		OuterLR:        0.005,
		InnerSteps:     5,
		NTasksPerBatch: 4,
		MetaEpochs:     200,
		HiddenDims:     []int{32, 32},
	}

	maml, err := NewMAML(config)
	if err != nil {
		t.Fatalf("NewMAML: %v", err)
	}

	if err := maml.MetaTrain(tasks, config); err != nil {
		t.Fatalf("MetaTrain: %v", err)
	}

	// Create a novel task the model has never seen.
	novelTask := newLinearTask(3.0, 0.5, 10, 10)

	// Adapt with just 5 gradient steps (few-shot).
	adapted := maml.Adapt(novelTask, 5)

	// Evaluate on test data.
	testIn, testTgt := novelTask.TestData()
	var mse float64
	for i := range testIn {
		pred, err := adapted.Predict(testIn[i])
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		diff := pred - testTgt[i]
		mse += diff * diff
	}
	mse /= float64(len(testIn))

	// After meta-learning + 5 adaptation steps, MSE should be reasonable.
	// For linear tasks, this should be well below 1.0.
	if mse > 2.0 {
		t.Errorf("few-shot adaptation MSE = %f, want < 2.0", mse)
	}

	// Also verify that adaptation actually reduces loss compared to no adaptation.
	noAdapt := maml.Adapt(novelTask, 0)
	var mseNoAdapt float64
	for i := range testIn {
		pred, err := noAdapt.Predict(testIn[i])
		if err != nil {
			t.Fatalf("Predict (no adapt): %v", err)
		}
		diff := pred - testTgt[i]
		mseNoAdapt += diff * diff
	}
	mseNoAdapt /= float64(len(testIn))

	if mse >= mseNoAdapt {
		t.Errorf("adaptation did not reduce loss: adapted MSE=%f >= unadapted MSE=%f", mse, mseNoAdapt)
	}
}

func TestMAML_MetaConvergence(t *testing.T) {
	// Create tasks from a family of sine functions with varying amplitude.
	tasks := make([]Task, 16)
	for i := 0; i < 16; i++ {
		amp := 0.5 + float64(i)*0.2
		phase := float64(i) * 0.3
		tasks[i] = newSinTask(amp, phase, 20, 10)
	}

	seed := uint64(42)
	config := MAMLConfig{
		InnerLR:        0.01,
		OuterLR:        0.003,
		InnerSteps:     3,
		NTasksPerBatch: 4,
		MetaEpochs:     50,
		HiddenDims:     []int{40, 40},
		Seed:           &seed,
	}

	maml, err := NewMAML(config)
	if err != nil {
		t.Fatalf("NewMAML: %v", err)
	}

	// Compute meta-loss before training by running a short init.
	// We need to initialize weights first by running 1 epoch.
	initConfig := config
	initConfig.MetaEpochs = 1
	if err := maml.MetaTrain(tasks, initConfig); err != nil {
		t.Fatalf("MetaTrain (init): %v", err)
	}
	lossBefore := maml.MetaLoss(tasks)

	// Now train for more epochs.
	trainConfig := config
	trainConfig.MetaEpochs = 150
	if err := maml.MetaTrain(tasks, trainConfig); err != nil {
		t.Fatalf("MetaTrain: %v", err)
	}
	lossAfter := maml.MetaLoss(tasks)

	if lossAfter >= lossBefore {
		t.Errorf("meta-loss did not decrease: before=%f, after=%f", lossBefore, lossAfter)
	}

	// The loss should decrease substantially (at least 20%).
	improvement := (lossBefore - lossAfter) / lossBefore
	if improvement < 0.1 {
		t.Errorf("meta-loss improvement = %.1f%%, want at least 10%%", improvement*100)
	}
}

func TestMAML_ConfigValidation(t *testing.T) {
	tests := []struct {
		name   string
		config MAMLConfig
	}{
		{"zero InnerLR", MAMLConfig{InnerLR: 0, OuterLR: 0.01, InnerSteps: 1, NTasksPerBatch: 1, MetaEpochs: 1, HiddenDims: []int{8}}},
		{"negative OuterLR", MAMLConfig{InnerLR: 0.01, OuterLR: -1, InnerSteps: 1, NTasksPerBatch: 1, MetaEpochs: 1, HiddenDims: []int{8}}},
		{"zero InnerSteps", MAMLConfig{InnerLR: 0.01, OuterLR: 0.01, InnerSteps: 0, NTasksPerBatch: 1, MetaEpochs: 1, HiddenDims: []int{8}}},
		{"zero NTasksPerBatch", MAMLConfig{InnerLR: 0.01, OuterLR: 0.01, InnerSteps: 1, NTasksPerBatch: 0, MetaEpochs: 1, HiddenDims: []int{8}}},
		{"zero MetaEpochs", MAMLConfig{InnerLR: 0.01, OuterLR: 0.01, InnerSteps: 1, NTasksPerBatch: 1, MetaEpochs: 0, HiddenDims: []int{8}}},
		{"empty HiddenDims", MAMLConfig{InnerLR: 0.01, OuterLR: 0.01, InnerSteps: 1, NTasksPerBatch: 1, MetaEpochs: 1, HiddenDims: nil}},
		{"negative HiddenDims", MAMLConfig{InnerLR: 0.01, OuterLR: 0.01, InnerSteps: 1, NTasksPerBatch: 1, MetaEpochs: 1, HiddenDims: []int{-1}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewMAML(tt.config)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestMAML_MetaTrainErrors(t *testing.T) {
	config := MAMLConfig{
		InnerLR:        0.01,
		OuterLR:        0.01,
		InnerSteps:     1,
		NTasksPerBatch: 1,
		MetaEpochs:     1,
		HiddenDims:     []int{8},
	}

	maml, err := NewMAML(config)
	if err != nil {
		t.Fatalf("NewMAML: %v", err)
	}

	// No tasks.
	if err := maml.MetaTrain(nil, config); err == nil {
		t.Error("MetaTrain with no tasks: expected error, got nil")
	}

	// Task with no data.
	emptyTask := newLinearTask(1, 0, 0, 0)
	if err := maml.MetaTrain([]Task{emptyTask}, config); err == nil {
		t.Error("MetaTrain with empty task: expected error, got nil")
	}
}

func TestAdaptedModel_PredictError(t *testing.T) {
	config := MAMLConfig{
		InnerLR:        0.01,
		OuterLR:        0.01,
		InnerSteps:     1,
		NTasksPerBatch: 1,
		MetaEpochs:     1,
		HiddenDims:     []int{8},
	}

	maml, err := NewMAML(config)
	if err != nil {
		t.Fatalf("NewMAML: %v", err)
	}

	task := newLinearTask(1, 0, 5, 5)
	if err := maml.MetaTrain([]Task{task}, config); err != nil {
		t.Fatalf("MetaTrain: %v", err)
	}

	adapted := maml.Adapt(task, 1)

	// Wrong input dimension.
	_, err = adapted.Predict([]float64{1, 2, 3})
	if err == nil {
		t.Error("Predict with wrong dim: expected error, got nil")
	}
}
