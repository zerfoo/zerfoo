package sequence

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func makeEngine() compute.Engine[float32] {
	return compute.NewCPUEngine(numeric.Float32Ops{})
}

func TestNewS4_ValidParams(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}
	if s4.OpType() != "S4" {
		t.Errorf("OpType = %q, want %q", s4.OpType(), "S4")
	}
}

func TestNewS4_InvalidParams(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name     string
		layerName string
		inputDim int
		stateDim int
	}{
		{"empty name", "", 4, 8},
		{"zero input dim", "s4", 0, 8},
		{"negative input dim", "s4", -1, 8},
		{"zero state dim", "s4", 4, 0},
		{"negative state dim", "s4", 4, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewS4[float32](tt.layerName, engine, ops, tt.inputDim, tt.stateDim)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestS4_Attributes(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	attrs := s4.Attributes()
	if attrs["input_dim"] != 4 {
		t.Errorf("input_dim = %v, want 4", attrs["input_dim"])
	}
	if attrs["state_dim"] != 8 {
		t.Errorf("state_dim = %v, want 8", attrs["state_dim"])
	}
}

func TestS4_OutputShape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	shape := s4.OutputShape()
	if len(shape) != 3 {
		t.Fatalf("OutputShape len = %d, want 3", len(shape))
	}
	// [-1, -1, 4] where -1 = dynamic batch and seq dims
	if shape[2] != 4 {
		t.Errorf("OutputShape[2] = %d, want 4 (input_dim)", shape[2])
	}
}

func TestS4_Parameters(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	params := s4.Parameters()
	if len(params) != 4 {
		t.Fatalf("Parameters len = %d, want 4 (aLog, b, c, d)", len(params))
	}

	// Check parameter names and shapes.
	wantNames := []string{"test_s4_a_log", "test_s4_b", "test_s4_c", "test_s4_d"}
	wantShapes := [][]int{{4, 8}, {4, 8}, {4, 8}, {4}}
	for i, p := range params {
		if p.Name != wantNames[i] {
			t.Errorf("param[%d].Name = %q, want %q", i, p.Name, wantNames[i])
		}
		shape := p.Value.Shape()
		if len(shape) != len(wantShapes[i]) {
			t.Errorf("param[%d].Shape = %v, want %v", i, shape, wantShapes[i])
			continue
		}
		for j := range shape {
			if shape[j] != wantShapes[i][j] {
				t.Errorf("param[%d].Shape[%d] = %d, want %d", i, j, shape[j], wantShapes[i][j])
			}
		}
	}
}

func TestS4_Forward_Shape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Input: [batch=2, seq=5, dim=4]
	input, err := tensor.New[float32]([]int{2, 5, 4}, make([]float32, 2*5*4))
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 2 || shape[1] != 5 || shape[2] != 4 {
		t.Errorf("output shape = %v, want [2, 5, 4]", shape)
	}
}

func TestS4_Forward_NonZeroOutput(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 2, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Non-zero input should produce non-zero output (due to skip connection D).
	inputData := make([]float32, 1*3*2)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}
	input, err := tensor.New[float32]([]int{1, 3, 2}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	hasNonZero := false
	for _, v := range output.Data() {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("expected non-zero output for non-zero input")
	}
}

func TestS4_Forward_FiniteOutput(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	inputData := make([]float32, 2*10*4)
	for i := range inputData {
		inputData[i] = float32(i%7) * 0.1
	}
	input, err := tensor.New[float32]([]int{2, 10, 4}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] = %f (not finite)", i, v)
		}
	}
}

func TestS4_Forward_WrongInputCount(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	ctx := context.Background()
	_, err = s4.Forward(ctx)
	if err == nil {
		t.Error("expected error for no inputs")
	}
}

func TestS4_Forward_WrongInputRank(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// 2D input instead of 3D
	input, _ := tensor.New[float32]([]int{2, 4}, make([]float32, 8))
	ctx := context.Background()
	_, err = s4.Forward(ctx, input)
	if err == nil {
		t.Error("expected error for 2D input")
	}
}

func TestS4_Forward_EdgeCase_SingleStep(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 2, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// seq_len=1, batch=1
	input, _ := tensor.New[float32]([]int{1, 1, 2}, []float32{1, 2})
	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if output.Shape()[0] != 1 || output.Shape()[1] != 1 || output.Shape()[2] != 2 {
		t.Errorf("shape = %v, want [1, 1, 2]", output.Shape())
	}
}

func TestS4_Forward_EdgeCase_Dim1(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 1, 1)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 1}, []float32{1, 2, 3})
	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if output.Shape()[2] != 1 {
		t.Errorf("output dim = %d, want 1", output.Shape()[2])
	}
}

func TestS4_Backward_Shape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Run forward first.
	inputData := make([]float32, 2*5*4)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	input, _ := tensor.New[float32]([]int{2, 5, 4}, inputData)
	ctx := context.Background()
	_, err = s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Backward with output gradient.
	gradData := make([]float32, 2*5*4)
	for i := range gradData {
		gradData[i] = 1.0
	}
	grad, _ := tensor.New[float32]([]int{2, 5, 4}, gradData)
	inputGrads, err := s4.Backward(ctx, types.OneStepApproximation, grad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(inputGrads) != 1 {
		t.Fatalf("Backward returned %d gradients, want 1", len(inputGrads))
	}
	gradShape := inputGrads[0].Shape()
	if len(gradShape) != 3 || gradShape[0] != 2 || gradShape[1] != 5 || gradShape[2] != 4 {
		t.Errorf("input gradient shape = %v, want [2, 5, 4]", gradShape)
	}
}

func TestS4_Backward_FiniteGradients(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 2, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	inputData := make([]float32, 1*3*2)
	for i := range inputData {
		inputData[i] = float32(i) * 0.1
	}
	input, _ := tensor.New[float32]([]int{1, 3, 2}, inputData)
	ctx := context.Background()
	_, err = s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	gradData := make([]float32, 1*3*2)
	for i := range gradData {
		gradData[i] = 1.0
	}
	grad, _ := tensor.New[float32]([]int{1, 3, 2}, gradData)
	inputGrads, err := s4.Backward(ctx, types.OneStepApproximation, grad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for i, v := range inputGrads[0].Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("input gradient[%d] = %f (not finite)", i, v)
		}
	}

	// Check parameter gradients are finite.
	for _, p := range s4.Parameters() {
		for i, v := range p.Gradient.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("param %s gradient[%d] = %f (not finite)", p.Name, i, v)
			}
		}
	}
}

func TestS4_Backward_WrongInputCount(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 2, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 2}, make([]float32, 6))
	ctx := context.Background()
	_, err = s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	grad, _ := tensor.New[float32]([]int{1, 3, 2}, make([]float32, 6))

	// Zero inputs
	_, err = s4.Backward(ctx, types.OneStepApproximation, grad)
	if err == nil {
		t.Error("expected error for zero inputs")
	}

	// Two inputs
	_, err = s4.Backward(ctx, types.OneStepApproximation, grad, input, input)
	if err == nil {
		t.Error("expected error for two inputs")
	}
}

func TestS4_Forward_WrongInputCountMultiple(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 2, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	input1, _ := tensor.New[float32]([]int{1, 3, 2}, make([]float32, 6))
	input2, _ := tensor.New[float32]([]int{1, 3, 2}, make([]float32, 6))
	ctx := context.Background()
	_, err = s4.Forward(ctx, input1, input2)
	if err == nil {
		t.Error("expected error for two inputs")
	}
}

func TestS4_RegistryBuilder_Valid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	attrs := map[string]interface{}{
		"input_dim": 4,
		"state_dim": 8,
	}
	builder, err := model.GetLayerBuilder[float32]("S4")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}
	node, err := builder(engine, ops, "registry_s4", nil, attrs)
	if err != nil {
		t.Fatalf("builder: %v", err)
	}
	if node.OpType() != "S4" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "S4")
	}
}

func TestS4_RegistryBuilder_MissingInputDim(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	attrs := map[string]interface{}{
		"state_dim": 8,
	}
	builder, err := model.GetLayerBuilder[float32]("S4")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}
	_, err = builder(engine, ops, "registry_s4", nil, attrs)
	if err == nil {
		t.Error("expected error for missing input_dim")
	}
}

func TestS4_RegistryBuilder_MissingStateDim(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	attrs := map[string]interface{}{
		"input_dim": 4,
	}
	builder, err := model.GetLayerBuilder[float32]("S4")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}
	_, err = builder(engine, ops, "registry_s4", nil, attrs)
	if err == nil {
		t.Error("expected error for missing state_dim")
	}
}

func TestS4_RegistryBuilder_WrongTypeDim(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name  string
		attrs map[string]interface{}
	}{
		{"input_dim string", map[string]interface{}{"input_dim": "four", "state_dim": 8}},
		{"state_dim string", map[string]interface{}{"input_dim": 4, "state_dim": "eight"}},
	}

	builder, err := model.GetLayerBuilder[float32]("S4")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := builder(engine, ops, "registry_s4", nil, tt.attrs)
			if err == nil {
				t.Error("expected error for wrong type attribute")
			}
		})
	}
}

func TestS4_Forward_LargerDimensions(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 32, 16)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// batch=4, seq=10, dim=32
	inputData := make([]float32, 4*10*32)
	for i := range inputData {
		inputData[i] = float32(i%11) * 0.01
	}
	input, _ := tensor.New[float32]([]int{4, 10, 32}, inputData)
	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := output.Shape()
	if shape[0] != 4 || shape[1] != 10 || shape[2] != 32 {
		t.Errorf("output shape = %v, want [4, 10, 32]", shape)
	}
}

// TestS4_NumericalGradientCheck verifies analytical gradients against finite-difference
// approximations for all parameters and the input.
func TestS4_NumericalGradientCheck(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	const (
		inputDim = 2
		stateDim = 3
		batch    = 1
		seqLen   = 3
		eps      = 1e-3
		tol      = 0.1 // relative tolerance for float32
	)

	// Helper: create a fresh S4 with specific parameter values.
	makeS4WithParams := func(aLogData, bData, cData []float32, dData []float32) *S4[float32] {
		s, err := NewS4[float32]("grad_check", engine, ops, inputDim, stateDim)
		if err != nil {
			t.Fatalf("NewS4: %v", err)
		}
		copy(s.aLog.Value.Data(), aLogData)
		copy(s.b.Value.Data(), bData)
		copy(s.c.Value.Data(), cData)
		copy(s.d.Value.Data(), dData)
		return s
	}

	// Fixed parameter values for reproducibility.
	aLogData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	bData := []float32{0.1, -0.1, 0.2, -0.2, 0.1, -0.1}
	cData := []float32{0.3, 0.1, -0.2, 0.2, -0.1, 0.3}
	dData := []float32{1.0, 0.5}

	inputData := []float32{0.5, -0.3, 0.2, 0.4, -0.1, 0.6}

	// Compute analytical gradients.
	s4 := makeS4WithParams(aLogData, bData, cData, dData)
	input, _ := tensor.New[float32]([]int{batch, seqLen, inputDim}, inputData)
	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Use sum of outputs as scalar loss.
	gradData := make([]float32, batch*seqLen*inputDim)
	for i := range gradData {
		gradData[i] = 1.0
	}
	outGrad, _ := tensor.New[float32]([]int{batch, seqLen, inputDim}, gradData)
	inputGrads, err := s4.Backward(ctx, types.OneStepApproximation, outGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Helper: compute scalar loss (sum of all outputs).
	computeLoss := func(s *S4[float32], in *tensor.TensorNumeric[float32]) float64 {
		out, err := s.Forward(ctx, in)
		if err != nil {
			t.Fatalf("Forward in loss: %v", err)
		}
		var sum float64
		for _, v := range out.Data() {
			sum += float64(v)
		}
		return sum
	}

	_ = output // used indirectly via forward

	// Table-driven parameter gradient checks.
	type paramGradCase struct {
		name     string
		data     []float32
		anaGrads []float32
		makeS4   func(perturbed []float32) *S4[float32]
	}

	paramCases := []paramGradCase{
		{
			name: "d_gradient", data: dData,
			anaGrads: s4.d.Gradient.Data(),
			makeS4:   func(p []float32) *S4[float32] { return makeS4WithParams(aLogData, bData, cData, p) },
		},
		{
			name: "b_gradient", data: bData,
			anaGrads: s4.b.Gradient.Data(),
			makeS4:   func(p []float32) *S4[float32] { return makeS4WithParams(aLogData, p, cData, dData) },
		},
		{
			name: "c_gradient", data: cData,
			anaGrads: s4.c.Gradient.Data(),
			makeS4:   func(p []float32) *S4[float32] { return makeS4WithParams(aLogData, bData, p, dData) },
		},
		{
			name: "a_log_gradient", data: aLogData,
			anaGrads: s4.aLog.Gradient.Data(),
			makeS4:   func(p []float32) *S4[float32] { return makeS4WithParams(p, bData, cData, dData) },
		},
	}

	// Check input gradient via finite differences.
	t.Run("input_gradient", func(t *testing.T) {
		for i := range inputData {
			pertPlus := make([]float32, len(inputData))
			pertMinus := make([]float32, len(inputData))
			copy(pertPlus, inputData)
			copy(pertMinus, inputData)
			pertPlus[i] += eps
			pertMinus[i] -= eps

			inPlus, _ := tensor.New[float32]([]int{batch, seqLen, inputDim}, pertPlus)
			inMinus, _ := tensor.New[float32]([]int{batch, seqLen, inputDim}, pertMinus)

			sPlus := makeS4WithParams(aLogData, bData, cData, dData)
			sMinus := makeS4WithParams(aLogData, bData, cData, dData)

			lPlus := computeLoss(sPlus, inPlus)
			lMinus := computeLoss(sMinus, inMinus)

			numGrad := (lPlus - lMinus) / (2 * eps)
			anaGrad := float64(inputGrads[0].Data()[i])

			if math.Abs(numGrad) > 1e-6 || math.Abs(anaGrad) > 1e-6 {
				relErr := math.Abs(numGrad-anaGrad) / (math.Abs(numGrad) + math.Abs(anaGrad) + 1e-8)
				if relErr > tol {
					t.Errorf("input grad[%d]: analytical=%.6f, numerical=%.6f, relErr=%.4f",
						i, anaGrad, numGrad, relErr)
				}
			}
		}
	})

	// Check parameter gradients via finite differences (table-driven).
	for _, pc := range paramCases {
		t.Run(pc.name, func(t *testing.T) {
			for i := range pc.data {
				plus := make([]float32, len(pc.data))
				minus := make([]float32, len(pc.data))
				copy(plus, pc.data)
				copy(minus, pc.data)
				plus[i] += eps
				minus[i] -= eps

				in, _ := tensor.New[float32]([]int{batch, seqLen, inputDim}, inputData)
				sPlus := pc.makeS4(plus)
				sMinus := pc.makeS4(minus)

				lPlus := computeLoss(sPlus, in)
				lMinus := computeLoss(sMinus, in)
				_ = in

				numGrad := (lPlus - lMinus) / (2 * eps)
				anaGrad := float64(pc.anaGrads[i])

				if math.Abs(numGrad) > 1e-6 || math.Abs(anaGrad) > 1e-6 {
					relErr := math.Abs(numGrad-anaGrad) / (math.Abs(numGrad) + math.Abs(anaGrad) + 1e-8)
					if relErr > tol {
						t.Errorf("%s grad[%d]: analytical=%.6f, numerical=%.6f, relErr=%.4f",
							pc.name, i, anaGrad, numGrad, relErr)
					}
				}
			}
		})
	}
}

// TestS4_Forward_Parity verifies that the engine-based Forward produces
// identical results to a reference scalar implementation (S96.11.1).
func TestS4_Forward_Parity(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name     string
		batch    int
		seqLen   int
		inputDim int
		stateDim int
	}{
		{"tiny", 1, 1, 1, 1},
		{"small", 1, 3, 2, 4},
		{"batched", 2, 5, 4, 8},
		{"large_state", 1, 4, 3, 16},
		{"multi_batch", 3, 2, 2, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s4, err := NewS4[float32]("parity", engine, ops, tt.inputDim, tt.stateDim)
			if err != nil {
				t.Fatalf("NewS4: %v", err)
			}

			// Snapshot parameter data for the reference implementation.
			aLogData := make([]float32, len(s4.aLog.Value.Data()))
			copy(aLogData, s4.aLog.Value.Data())
			bData := make([]float32, len(s4.b.Value.Data()))
			copy(bData, s4.b.Value.Data())
			cData := make([]float32, len(s4.c.Value.Data()))
			copy(cData, s4.c.Value.Data())
			dData := make([]float32, len(s4.d.Value.Data()))
			copy(dData, s4.d.Value.Data())

			// Build deterministic input.
			n := tt.batch * tt.seqLen * tt.inputDim
			inputData := make([]float32, n)
			for i := range inputData {
				inputData[i] = float32(i%7)*0.1 - 0.3
			}
			input, err := tensor.New[float32]([]int{tt.batch, tt.seqLen, tt.inputDim}, inputData)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			// Run the engine-based Forward.
			ctx := context.Background()
			got, err := s4.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Reference scalar implementation (original raw-data logic).
			aDisc := make([]float32, tt.inputDim*tt.stateDim)
			for i, v := range aLogData {
				aDisc[i] = float32(math.Exp(-math.Exp(float64(v))))
			}
			wantData := make([]float32, tt.batch*tt.seqLen*tt.inputDim)
			state := make([]float32, tt.batch*tt.inputDim*tt.stateDim)
			for batch := range tt.batch {
				for step := range tt.seqLen {
					for d := range tt.inputDim {
						u := inputData[batch*tt.seqLen*tt.inputDim+step*tt.inputDim+d]
						var y float32
						for sn := range tt.stateDim {
							idx := d*tt.stateDim + sn
							si := batch*tt.inputDim*tt.stateDim + idx
							state[si] = aDisc[idx]*state[si] + bData[idx]*u
							y += cData[idx] * state[si]
						}
						y += dData[d] * u
						wantData[batch*tt.seqLen*tt.inputDim+step*tt.inputDim+d] = y
					}
				}
			}

			gotData := got.Data()
			if len(gotData) != len(wantData) {
				t.Fatalf("output length = %d, want %d", len(gotData), len(wantData))
			}
			for i := range wantData {
				diff := float64(gotData[i] - wantData[i])
				if diff < 0 {
					diff = -diff
				}
				if diff > 1e-5 {
					t.Errorf("output[%d] = %g, want %g (diff %g)", i, gotData[i], wantData[i], diff)
				}
			}
		})
	}
}

// Compile-time interface check.
var _ = (*S4[float32])(nil)
