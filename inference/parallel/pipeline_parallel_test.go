package parallel

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestPipelineParallelConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     PipelineParallelConfig
		wantErr bool
	}{
		{
			name:    "valid config",
			cfg:     PipelineParallelConfig{NumStages: 4, NumLayers: 32, MicroBatchSize: 8},
			wantErr: false,
		},
		{
			name:    "single stage",
			cfg:     PipelineParallelConfig{NumStages: 1, NumLayers: 1, MicroBatchSize: 1},
			wantErr: false,
		},
		{
			name:    "zero stages",
			cfg:     PipelineParallelConfig{NumStages: 0, NumLayers: 32, MicroBatchSize: 8},
			wantErr: true,
		},
		{
			name:    "zero layers",
			cfg:     PipelineParallelConfig{NumStages: 4, NumLayers: 0, MicroBatchSize: 8},
			wantErr: true,
		},
		{
			name:    "more stages than layers",
			cfg:     PipelineParallelConfig{NumStages: 8, NumLayers: 4, MicroBatchSize: 8},
			wantErr: true,
		},
		{
			name:    "zero micro-batches",
			cfg:     PipelineParallelConfig{NumStages: 4, NumLayers: 32, MicroBatchSize: 0},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestAssignLayers(t *testing.T) {
	tests := []struct {
		name      string
		numLayers int
		numStages int
		wantStage [][]int // expected layers per stage
	}{
		{
			name:      "even distribution 8 layers 4 stages",
			numLayers: 8,
			numStages: 4,
			wantStage: [][]int{{0, 1}, {2, 3}, {4, 5}, {6, 7}},
		},
		{
			name:      "even distribution 32 layers 4 stages",
			numLayers: 32,
			numStages: 4,
			wantStage: [][]int{
				{0, 1, 2, 3, 4, 5, 6, 7},
				{8, 9, 10, 11, 12, 13, 14, 15},
				{16, 17, 18, 19, 20, 21, 22, 23},
				{24, 25, 26, 27, 28, 29, 30, 31},
			},
		},
		{
			name:      "uneven distribution 7 layers 3 stages",
			numLayers: 7,
			numStages: 3,
			wantStage: [][]int{{0, 1, 2}, {3, 4}, {5, 6}},
		},
		{
			name:      "single stage",
			numLayers: 4,
			numStages: 1,
			wantStage: [][]int{{0, 1, 2, 3}},
		},
		{
			name:      "one layer per stage",
			numLayers: 4,
			numStages: 4,
			wantStage: [][]int{{0}, {1}, {2}, {3}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sa := AssignLayers(tc.numLayers, tc.numStages)

			if len(sa.LayersPerStage) != tc.numStages {
				t.Fatalf("got %d stages, want %d", len(sa.LayersPerStage), tc.numStages)
			}

			for stage, want := range tc.wantStage {
				got := sa.LayersPerStage[stage]
				if len(got) != len(want) {
					t.Errorf("stage %d: got %d layers %v, want %d layers %v", stage, len(got), got, len(want), want)
					continue
				}
				for i := range want {
					if got[i] != want[i] {
						t.Errorf("stage %d: layer[%d] = %d, want %d", stage, i, got[i], want[i])
					}
				}
			}

			// Verify StageForLayer is consistent.
			for layer, stage := range sa.StageForLayer {
				found := false
				for _, l := range sa.LayersPerStage[stage] {
					if l == layer {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("layer %d mapped to stage %d but not in LayersPerStage[%d]", layer, stage, stage)
				}
			}

			// Verify all layers assigned.
			total := 0
			for _, layers := range sa.LayersPerStage {
				total += len(layers)
			}
			if total != tc.numLayers {
				t.Errorf("total assigned layers = %d, want %d", total, tc.numLayers)
			}
		})
	}
}

func TestForwardSchedule(t *testing.T) {
	cfg := PipelineParallelConfig{NumStages: 3, NumLayers: 6, MicroBatchSize: 4}
	sched, err := NewPipelineScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}

	schedule := sched.ForwardSchedule()
	// 3 stages + 4 micro-batches - 1 = 6 clock cycles.
	wantClocks := 6
	if len(schedule) != wantClocks {
		t.Fatalf("got %d clock cycles, want %d", len(schedule), wantClocks)
	}

	// Verify each (stage, micro-batch) pair appears exactly once.
	seen := make(map[string]bool)
	for _, steps := range schedule {
		for _, step := range steps {
			key := fmt.Sprintf("s%d-mb%d", step.Stage, step.MicroBatch)
			if seen[key] {
				t.Errorf("duplicate step: %s", key)
			}
			seen[key] = true
		}
	}
	wantPairs := cfg.NumStages * cfg.MicroBatchSize
	if len(seen) != wantPairs {
		t.Errorf("got %d (stage, micro-batch) pairs, want %d", len(seen), wantPairs)
	}

	// Clock 0: only stage 0, mb 0.
	if len(schedule[0]) != 1 || schedule[0][0].Stage != 0 || schedule[0][0].MicroBatch != 0 {
		t.Errorf("clock 0: got %v, want [{Stage:0, MicroBatch:0}]", schedule[0])
	}
}

func TestBubbleRatio(t *testing.T) {
	tests := []struct {
		name      string
		numStages int
		numMB     int
		wantMax   float64 // bubble ratio must be <= this
	}{
		{
			name:      "4 stages 16 micro-batches",
			numStages: 4,
			numMB:     16,
			wantMax:   0.20, // 3/19 ~ 0.158
		},
		{
			name:      "4 stages 4 micro-batches",
			numStages: 4,
			numMB:     4,
			wantMax:   0.50, // 3/7 ~ 0.429
		},
		{
			name:      "2 stages 8 micro-batches",
			numStages: 2,
			numMB:     8,
			wantMax:   0.15, // 1/9 ~ 0.111
		},
		{
			name:      "1 stage any micro-batches",
			numStages: 1,
			numMB:     4,
			wantMax:   0.001, // 0/3 = 0
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sched, err := NewPipelineScheduler(PipelineParallelConfig{
				NumStages:      tc.numStages,
				NumLayers:      tc.numStages * 8,
				MicroBatchSize: tc.numMB,
			})
			if err != nil {
				t.Fatal(err)
			}
			ratio := sched.BubbleRatio()
			if ratio > tc.wantMax {
				t.Errorf("bubble ratio = %f, want <= %f", ratio, tc.wantMax)
			}

			// Verify exact formula: (S-1)/(S+M-1).
			exact := float64(tc.numStages-1) / float64(tc.numStages+tc.numMB-1)
			if math.Abs(ratio-exact) > 1e-10 {
				t.Errorf("bubble ratio = %f, want exact %f", ratio, exact)
			}
		})
	}
}

func TestBubbleRatioUnder20PercentAt4GPUs(t *testing.T) {
	// Acceptance criterion: bubble ratio < 20% at 4 GPUs.
	// With 16 micro-batches: (4-1)/(4+16-1) = 3/19 ~ 15.8%.
	sched, err := NewPipelineScheduler(PipelineParallelConfig{
		NumStages:      4,
		NumLayers:      32,
		MicroBatchSize: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	ratio := sched.BubbleRatio()
	if ratio >= 0.20 {
		t.Errorf("bubble ratio at 4 GPUs with 16 micro-batches = %.4f, want < 0.20", ratio)
	}
}

func TestPipelineParallel(t *testing.T) {
	// End-to-end test: 4 stages, 8 layers (2 per stage), 4 micro-batches.
	// Each layer multiplies the activation by 2.0, so after 8 layers the
	// activation should be multiplied by 2^8 = 256.
	numStages := 4
	numLayers := 8
	numMB := 4

	cfg := PipelineParallelConfig{
		NumStages:      numStages,
		NumLayers:      numLayers,
		MicroBatchSize: numMB,
	}

	sched, err := NewPipelineScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Create one CPU engine per stage.
	engines := make([]compute.Engine[float32], numStages)
	for i := range numStages {
		engines[i] = compute.NewCPUEngine[float32](numeric.Float32Ops{})
	}

	// Layer function: multiply each element by 2.
	layerFn := func(layerIdx int, input *tensor.TensorNumeric[float32], eng compute.Engine[float32]) (*tensor.TensorNumeric[float32], error) {
		data := input.Data()
		out := make([]float32, len(data))
		for i, v := range data {
			out[i] = v * 2.0
		}
		return tensor.New[float32](input.Shape(), out)
	}

	exec, err := NewPipelineExecutor(sched, engines, layerFn)
	if err != nil {
		t.Fatal(err)
	}

	// Create 4 micro-batches, each [2, 4] filled with 1.0.
	microBatches := make([]*tensor.TensorNumeric[float32], numMB)
	for i := range numMB {
		data := make([]float32, 8)
		for j := range data {
			data[j] = 1.0
		}
		mb, err := tensor.New[float32]([]int{2, 4}, data)
		if err != nil {
			t.Fatal(err)
		}
		microBatches[i] = mb
	}

	ctx := context.Background()
	outputs, err := exec.Execute(ctx, microBatches)
	if err != nil {
		t.Fatal(err)
	}

	if len(outputs) != numMB {
		t.Fatalf("got %d outputs, want %d", len(outputs), numMB)
	}

	expected := float32(256.0) // 2^8
	for i, out := range outputs {
		data := out.Data()
		for j, v := range data {
			if math.Abs(float64(v-expected)) > 1e-3 {
				t.Errorf("micro-batch %d, element %d = %f, want %f", i, j, v, expected)
			}
		}
	}
}

func TestPipelineExecutorContextCancellation(t *testing.T) {
	cfg := PipelineParallelConfig{NumStages: 2, NumLayers: 4, MicroBatchSize: 2}
	sched, err := NewPipelineScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}

	engines := []compute.Engine[float32]{
		compute.NewCPUEngine[float32](numeric.Float32Ops{}),
		compute.NewCPUEngine[float32](numeric.Float32Ops{}),
	}

	layerFn := func(layerIdx int, input *tensor.TensorNumeric[float32], eng compute.Engine[float32]) (*tensor.TensorNumeric[float32], error) {
		return input, nil
	}

	exec, err := NewPipelineExecutor(sched, engines, layerFn)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	mbs := make([]*tensor.TensorNumeric[float32], 2)
	for i := range mbs {
		mbs[i], _ = tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	}

	_, err = exec.Execute(ctx, mbs)
	if err == nil {
		t.Error("expected error from cancelled context, got nil")
	}
}

func TestPipelineExecutorWrongEngineCount(t *testing.T) {
	cfg := PipelineParallelConfig{NumStages: 4, NumLayers: 8, MicroBatchSize: 2}
	sched, err := NewPipelineScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}

	engines := []compute.Engine[float32]{
		compute.NewCPUEngine[float32](numeric.Float32Ops{}),
	}

	_, err = NewPipelineExecutor(sched, engines, func(int, *tensor.TensorNumeric[float32], compute.Engine[float32]) (*tensor.TensorNumeric[float32], error) {
		return nil, nil
	})
	if err == nil {
		t.Error("expected error for wrong engine count, got nil")
	}
}

func TestPipelineExecutorWrongMicroBatchCount(t *testing.T) {
	cfg := PipelineParallelConfig{NumStages: 2, NumLayers: 4, MicroBatchSize: 4}
	sched, err := NewPipelineScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}

	engines := []compute.Engine[float32]{
		compute.NewCPUEngine[float32](numeric.Float32Ops{}),
		compute.NewCPUEngine[float32](numeric.Float32Ops{}),
	}

	exec, err := NewPipelineExecutor(sched, engines, func(int, *tensor.TensorNumeric[float32], compute.Engine[float32]) (*tensor.TensorNumeric[float32], error) {
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Pass 2 micro-batches when 4 are expected.
	mbs := make([]*tensor.TensorNumeric[float32], 2)
	for i := range mbs {
		mbs[i], _ = tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	}

	_, err = exec.Execute(context.Background(), mbs)
	if err == nil {
		t.Error("expected error for wrong micro-batch count, got nil")
	}
}

func TestSplitMicroBatches(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	batch, err := tensor.New[float32]([]int{4, 6}, data)
	if err != nil {
		t.Fatal(err)
	}

	mbs, err := SplitMicroBatches(batch, 2, eng)
	if err != nil {
		t.Fatal(err)
	}

	if len(mbs) != 2 {
		t.Fatalf("got %d micro-batches, want 2", len(mbs))
	}

	for i, mb := range mbs {
		shape := mb.Shape()
		if shape[0] != 2 || shape[1] != 6 {
			t.Errorf("micro-batch %d shape = %v, want [2, 6]", i, shape)
		}
	}

	// Verify data is correct.
	mb0Data := mbs[0].Data()
	for i, v := range mb0Data {
		if v != float32(i) {
			t.Errorf("micro-batch 0, element %d = %f, want %f", i, v, float32(i))
		}
	}
	mb1Data := mbs[1].Data()
	for i, v := range mb1Data {
		if v != float32(i+12) {
			t.Errorf("micro-batch 1, element %d = %f, want %f", i, v, float32(i+12))
		}
	}
}

func TestSplitMicroBatchesIndivisible(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	batch, _ := tensor.New[float32]([]int{5, 3}, make([]float32, 15))

	_, err := SplitMicroBatches(batch, 2, eng)
	if err == nil {
		t.Error("expected error for indivisible batch size, got nil")
	}
}

func TestConcatMicroBatches(t *testing.T) {
	mb0, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	mb1, _ := tensor.New[float32]([]int{2, 3}, []float32{7, 8, 9, 10, 11, 12})

	result, err := ConcatMicroBatches([]*tensor.TensorNumeric[float32]{mb0, mb1})
	if err != nil {
		t.Fatal(err)
	}

	shape := result.Shape()
	if shape[0] != 4 || shape[1] != 3 {
		t.Errorf("shape = %v, want [4, 3]", shape)
	}

	data := result.Data()
	for i, v := range data {
		if v != float32(i+1) {
			t.Errorf("element %d = %f, want %f", i, v, float32(i+1))
		}
	}
}

func TestConcatMicroBatchesEmpty(t *testing.T) {
	_, err := ConcatMicroBatches[float32](nil)
	if err == nil {
		t.Error("expected error for empty slice, got nil")
	}
}

func TestConcatMicroBatchesSingle(t *testing.T) {
	mb, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result, err := ConcatMicroBatches([]*tensor.TensorNumeric[float32]{mb})
	if err != nil {
		t.Fatal(err)
	}
	if result != mb {
		t.Error("single micro-batch should return the same tensor")
	}
}

func TestPipelineExecutorLayerError(t *testing.T) {
	cfg := PipelineParallelConfig{NumStages: 2, NumLayers: 4, MicroBatchSize: 1}
	sched, err := NewPipelineScheduler(cfg)
	if err != nil {
		t.Fatal(err)
	}

	engines := []compute.Engine[float32]{
		compute.NewCPUEngine[float32](numeric.Float32Ops{}),
		compute.NewCPUEngine[float32](numeric.Float32Ops{}),
	}

	layerFn := func(layerIdx int, input *tensor.TensorNumeric[float32], eng compute.Engine[float32]) (*tensor.TensorNumeric[float32], error) {
		if layerIdx == 2 {
			return nil, fmt.Errorf("simulated failure at layer %d", layerIdx)
		}
		return input, nil
	}

	exec, err := NewPipelineExecutor(sched, engines, layerFn)
	if err != nil {
		t.Fatal(err)
	}

	mbs := make([]*tensor.TensorNumeric[float32], 1)
	mbs[0], _ = tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})

	_, err = exec.Execute(context.Background(), mbs)
	if err == nil {
		t.Error("expected error from failing layer, got nil")
	}
}

func TestSchedulerAssignment(t *testing.T) {
	sched, err := NewPipelineScheduler(PipelineParallelConfig{
		NumStages:      4,
		NumLayers:      32,
		MicroBatchSize: 8,
	})
	if err != nil {
		t.Fatal(err)
	}

	a := sched.Assignment()
	if len(a.StageForLayer) != 32 {
		t.Errorf("got %d layer assignments, want 32", len(a.StageForLayer))
	}
	if len(a.LayersPerStage) != 4 {
		t.Errorf("got %d stages, want 4", len(a.LayersPerStage))
	}
	for stage, layers := range a.LayersPerStage {
		if len(layers) != 8 {
			t.Errorf("stage %d has %d layers, want 8", stage, len(layers))
		}
	}
}
