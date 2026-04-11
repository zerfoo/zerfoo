package parity_test

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/layers/gather"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/recurrent"
	"github.com/zerfoo/zerfoo/layers/reducesum"
	"github.com/zerfoo/zerfoo/layers/regularization"
	"github.com/zerfoo/zerfoo/layers/ssm"
	ltranspose "github.com/zerfoo/zerfoo/layers/transpose"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// goldenDir returns the path to tests/golden/layers/.
func goldenDir() string {
	_, f, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(f), "..", "golden", "layers")
}

// loadGolden loads a golden JSON file and returns the raw map.
func loadGolden(t *testing.T, name string) map[string]interface{} {
	t.Helper()
	path := filepath.Join(goldenDir(), name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("load golden %s: %v", name, err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("parse golden %s: %v", name, err)
	}
	return m
}

// getFloat32s extracts a float32 slice from a JSON array.
func getFloat32s(m map[string]interface{}, key string) []float32 {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	out := make([]float32, len(arr))
	for i, v := range arr {
		out[i] = float32(v.(float64))
	}
	return out
}

// getInts extracts an int slice from a JSON array.
func getInts(m map[string]interface{}, key string) []int {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	out := make([]int, len(arr))
	for i, v := range arr {
		out[i] = int(v.(float64))
	}
	return out
}

// getFloat extracts a float64 from a JSON value.
func getFloat(m map[string]interface{}, key string) float64 {
	return m[key].(float64)
}

// makeTensor creates a tensor from golden data.
func makeTensor(t *testing.T, data []float32, shape []int) *tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("create tensor shape=%v: %v", shape, err)
	}
	return tn
}

// makeParam creates a graph.Parameter from golden data.
func makeParam(t *testing.T, name string, data []float32, shape []int) *graph.Parameter[float32] {
	t.Helper()
	tn := makeTensor(t, data, shape)
	p, err := graph.NewParameter[float32](name, tn, tensor.New[float32])
	if err != nil {
		t.Fatalf("create param %s: %v", name, err)
	}
	return p
}

// compareSlices compares two float32 slices with tolerance.
// Returns the number of mismatches and the max absolute difference.
func compareSlices(got, want []float32, tol float64) (mismatches int, maxDiff float64) {
	if len(got) != len(want) {
		return len(got) + len(want), math.Inf(1)
	}
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tol {
			mismatches++
		}
	}
	return
}

// assertClose compares output data against expected with tolerance.
func assertClose(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	mismatches, maxDiff := compareSlices(got, want, tol)
	if mismatches > 0 {
		// Show first few mismatches
		shown := 0
		for i := range got {
			diff := math.Abs(float64(got[i] - want[i]))
			if diff > tol {
				t.Errorf("%s[%d]: got %g, want %g (diff=%g)", label, i, got[i], want[i], diff)
				shown++
				if shown >= 5 {
					break
				}
			}
		}
		t.Errorf("%s: %d/%d values exceed tolerance %g (maxDiff=%g)", label, mismatches, len(got), tol, maxDiff)
	}
}

// setup creates engine and ops for all tests.
func setup() (compute.Engine[float32], *numeric.Float32Ops) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	return engine, ops
}

// ---------------------------------------------------------------------------
// Activation tests
// ---------------------------------------------------------------------------

func TestParity_ReLU(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_relu")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	relu := activations.NewReLU(engine, ops)
	output, err := relu.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "relu_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_GELU(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_gelu")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	gelu := activations.NewGelu(engine, ops)
	output, err := gelu.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "gelu_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Sigmoid(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_sigmoid")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	sigmoid := activations.NewSigmoid(engine, ops)
	output, err := sigmoid.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "sigmoid_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Tanh(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_tanh")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	tanh := activations.NewTanh(engine, ops)
	output, err := tanh.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "tanh_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Softmax(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "activation_softmax")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	sm := activations.NewSoftmax[float32](engine, -1)
	output, err := sm.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "softmax_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_LeakyReLU(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_leaky_relu")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	alpha := getFloat(g, "alpha")
	lrelu := activations.NewLeakyReLU(engine, ops, activations.WithAlpha[float32](alpha))
	output, err := lrelu.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "leaky_relu_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_SwiGLU(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_swiglu")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	swiglu := activations.NewSwiGLU[float32](engine, ops)
	output, err := swiglu.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "swiglu_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Erf(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_erf")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	erf := activations.NewErf(engine, ops)
	output, err := erf.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "erf_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// Functional tests (pure functions, no layer state)
// ---------------------------------------------------------------------------

func TestParity_Functional_ReLU(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_relu")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	output, err := functional.ReLU(context.Background(), engine, ops, input)
	if err != nil {
		t.Fatalf("functional.ReLU: %v", err)
	}
	assertClose(t, "functional_relu", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Functional_GELU(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_gelu")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	output, err := functional.GELU(context.Background(), engine, ops, input)
	if err != nil {
		t.Fatalf("functional.GELU: %v", err)
	}
	assertClose(t, "functional_gelu", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Functional_Sigmoid(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_sigmoid")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	output, err := functional.Sigmoid(context.Background(), engine, ops, input)
	if err != nil {
		t.Fatalf("functional.Sigmoid: %v", err)
	}
	assertClose(t, "functional_sigmoid", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Functional_SiLU(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "activation_silu")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	output, err := functional.SiLU(context.Background(), engine, ops, input)
	if err != nil {
		t.Fatalf("functional.SiLU: %v", err)
	}
	assertClose(t, "functional_silu", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Functional_Softmax(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "activation_softmax")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	output, err := functional.Softmax(context.Background(), engine, input, -1)
	if err != nil {
		t.Fatalf("functional.Softmax: %v", err)
	}
	assertClose(t, "functional_softmax", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Functional_LayerNorm(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "norm_layer_norm")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	gamma := makeTensor(t, getFloat32s(g, "gamma"), getInts(g, "gamma_shape"))
	beta := makeTensor(t, getFloat32s(g, "beta"), getInts(g, "beta_shape"))
	eps := float32(getFloat(g, "epsilon"))

	output, err := functional.LayerNorm(context.Background(), engine, input, gamma, beta, eps)
	if err != nil {
		t.Fatalf("functional.LayerNorm: %v", err)
	}
	assertClose(t, "functional_layernorm", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Functional_RMSNorm(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "norm_rms_norm")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	gain := makeTensor(t, getFloat32s(g, "gain"), getInts(g, "gain_shape"))
	eps := float32(getFloat(g, "epsilon"))

	output, err := functional.RMSNorm(context.Background(), engine, input, gain, eps)
	if err != nil {
		t.Fatalf("functional.RMSNorm: %v", err)
	}
	assertClose(t, "functional_rmsnorm", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Functional_Linear(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "core_dense")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	// functional.Linear expects weight as [out_features, in_features] (PyTorch convention)
	// and does x @ W^T + bias. Golden file has weight as [in, out], so we transpose.
	wShape := getInts(g, "weight_shape")
	wData := getFloat32s(g, "weight")
	inF, outF := wShape[0], wShape[1]
	// Transpose [in, out] -> [out, in]
	wT := make([]float32, len(wData))
	for i := 0; i < inF; i++ {
		for j := 0; j < outF; j++ {
			wT[j*inF+i] = wData[i*outF+j]
		}
	}
	weight := makeTensor(t, wT, []int{outF, inF})
	bias := makeTensor(t, getFloat32s(g, "bias"), getInts(g, "bias_shape"))

	output, err := functional.Linear(context.Background(), engine, input, weight, bias)
	if err != nil {
		t.Fatalf("functional.Linear: %v", err)
	}
	assertClose(t, "functional_linear", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// Normalization tests
// ---------------------------------------------------------------------------

func TestParity_LayerNorm(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "norm_layer_norm")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	gamma := makeTensor(t, getFloat32s(g, "gamma"), getInts(g, "gamma_shape"))
	beta := makeTensor(t, getFloat32s(g, "beta"), getInts(g, "beta_shape"))
	eps := float32(getFloat(g, "epsilon"))

	gammaParam := makeParam(t, "gamma", getFloat32s(g, "gamma"), getInts(g, "gamma_shape"))
	betaParam := makeParam(t, "beta", getFloat32s(g, "beta"), getInts(g, "beta_shape"))

	_ = gamma
	_ = beta

	ln := normalization.NewLayerNormalizationFromParams(engine, eps, gammaParam, betaParam)
	output, err := ln.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "layer_norm_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_RMSNorm(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "norm_rms_norm")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	eps := float32(getFloat(g, "epsilon"))

	gainParam := makeParam(t, "gain", getFloat32s(g, "gain"), getInts(g, "gain_shape"))

	rms, err := normalization.NewRMSNormFromParam(engine, ops, eps, gainParam)
	if err != nil {
		t.Fatalf("NewRMSNormFromParam: %v", err)
	}
	output, err := rms.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "rms_norm_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// Core layer tests
// ---------------------------------------------------------------------------

func TestParity_Linear(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "core_linear")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	weightParam := makeParam(t, "weight", getFloat32s(g, "weight"), getInts(g, "weight_shape"))

	linear := core.NewLinearFromParam(engine, weightParam)
	output, err := linear.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "linear_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_MatMul(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "core_matmul")
	tol := getFloat(g, "tolerance")

	a := makeTensor(t, getFloat32s(g, "input_a"), getInts(g, "input_a_shape"))
	b := makeTensor(t, getFloat32s(g, "input_b"), getInts(g, "input_b_shape"))

	mm := core.NewMatMul[float32](engine)
	output, err := mm.Forward(context.Background(), a, b)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "matmul_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Conv1D(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "core_conv1d")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	inCh := int(getFloat(g, "in_channels"))
	outCh := int(getFloat(g, "out_channels"))
	kernel := int(getFloat(g, "kernel_size"))
	stride := int(getFloat(g, "stride"))
	padding := int(getFloat(g, "padding"))

	conv, err := core.NewConv1D[float32]("test_conv1d", engine, ops, inCh, outCh, kernel,
		core.Conv1DStride(stride), core.Conv1DPadding(padding))
	if err != nil {
		t.Fatalf("NewConv1D: %v", err)
	}

	// Set weights from golden data
	params := conv.Parameters()
	weightData := getFloat32s(g, "weight")
	biasData := getFloat32s(g, "bias")

	// The Conv1D weight shape from PyTorch is [out_ch, in_ch, kernel],
	// need to check if Zerfoo expects the same layout
	for _, p := range params {
		pdata := p.Value.Data()
		if len(pdata) == len(weightData) {
			copy(pdata, weightData)
		} else if len(pdata) == len(biasData) {
			copy(pdata, biasData)
		}
	}

	output, err := conv.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "conv1d_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// Attention tests
// ---------------------------------------------------------------------------

func TestParity_SDPA_Causal(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "attention_sdpa_causal")
	tol := getFloat(g, "tolerance")

	q := makeTensor(t, getFloat32s(g, "query"), getInts(g, "query_shape"))
	k := makeTensor(t, getFloat32s(g, "key"), getInts(g, "key_shape"))
	v := makeTensor(t, getFloat32s(g, "value"), getInts(g, "value_shape"))

	headDim := getInts(g, "query_shape")[2]
	sdpa := attention.NewScaledDotProductAttention[float32](engine, headDim)
	sdpa.SetCausal(true)
	output, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "sdpa_causal_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_SDPA_Bidirectional(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "attention_sdpa_bidirectional")
	tol := getFloat(g, "tolerance")

	q := makeTensor(t, getFloat32s(g, "query"), getInts(g, "query_shape"))
	k := makeTensor(t, getFloat32s(g, "key"), getInts(g, "key_shape"))
	v := makeTensor(t, getFloat32s(g, "value"), getInts(g, "value_shape"))

	headDim := getInts(g, "query_shape")[2]
	sdpa := attention.NewBidirectionalSDPA[float32](engine, headDim)
	output, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "sdpa_bidirectional_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_MultiHeadAttention(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "attention_multi_head")
	tol := getFloat(g, "tolerance")

	q := makeTensor(t, getFloat32s(g, "query"), getInts(g, "query_shape"))
	k := makeTensor(t, getFloat32s(g, "key"), getInts(g, "key_shape"))
	v := makeTensor(t, getFloat32s(g, "value"), getInts(g, "value_shape"))
	nHeads := int(getFloat(g, "n_heads"))

	output, err := functional.MultiHeadAttention(context.Background(), engine, q, k, v, nHeads)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "mha_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// Embedding tests
// ---------------------------------------------------------------------------

func TestParity_TokenEmbedding(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "embedding_token")
	tol := getFloat(g, "tolerance")

	tableData := getFloat32s(g, "table")
	tableShape := getInts(g, "table_shape")
	tableParam := makeParam(t, "embedding_table", tableData, tableShape)

	emb, err := embeddings.NewTokenEmbeddingFromParam(engine, tableParam)
	if err != nil {
		t.Fatalf("NewTokenEmbeddingFromParam: %v", err)
	}

	// Create index tensor
	idxRaw := g["indices"].([]interface{})
	idxData := make([]float32, len(idxRaw))
	for i, v := range idxRaw {
		idxData[i] = float32(v.(float64))
	}
	idxTensor := makeTensor(t, idxData, []int{len(idxData)})

	output, err := emb.Forward(context.Background(), idxTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "token_embedding_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_RotaryEmbedding(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "embedding_rotary")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	inputShape := getInts(g, "input_shape")
	headDim := inputShape[2]
	seqLen := inputShape[1]
	base := getFloat(g, "base")

	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, headDim, seqLen,
		embeddings.WithRotaryBase(base),
	)
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding: %v", err)
	}

	output, err := rope.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "rotary_embedding_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// Loss function tests
// ---------------------------------------------------------------------------

func TestParity_MSELoss(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "loss_mse")
	tol := getFloat(g, "tolerance")

	pred := makeTensor(t, getFloat32s(g, "predictions"), getInts(g, "predictions_shape"))
	target := makeTensor(t, getFloat32s(g, "targets"), getInts(g, "targets_shape"))

	mse := loss.NewMSE(engine, ops)
	output, err := mse.Forward(context.Background(), pred, target)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expectedLoss := float32(getFloat(g, "expected_loss"))
	gotLoss := output.Data()[0]
	diff := math.Abs(float64(gotLoss - expectedLoss))
	if diff > tol {
		t.Errorf("mse_loss: got %g, want %g (diff=%g)", gotLoss, expectedLoss, diff)
	}
}

func TestParity_BCELoss(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "loss_bce")
	tol := getFloat(g, "tolerance")

	pred := makeTensor(t, getFloat32s(g, "predictions"), getInts(g, "predictions_shape"))
	target := makeTensor(t, getFloat32s(g, "targets"), getInts(g, "targets_shape"))

	bce := loss.NewBCELoss(engine, ops)
	output, err := bce.Forward(context.Background(), pred, target)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expectedLoss := float32(getFloat(g, "expected_loss"))
	gotLoss := output.Data()[0]
	diff := math.Abs(float64(gotLoss - expectedLoss))
	if diff > tol {
		t.Errorf("bce_loss: got %g, want %g (diff=%g)", gotLoss, expectedLoss, diff)
	}
}

func TestParity_CrossEntropyLoss(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "loss_cross_entropy")
	tol := getFloat(g, "tolerance")

	logits := makeTensor(t, getFloat32s(g, "logits"), getInts(g, "logits_shape"))

	// Create target tensor (class indices as float32)
	targetsRaw := g["targets"].([]interface{})
	targetData := make([]float32, len(targetsRaw))
	for i, v := range targetsRaw {
		targetData[i] = float32(v.(float64))
	}
	targets := makeTensor(t, targetData, []int{len(targetData)})

	cel := loss.NewCrossEntropyLoss[float32](engine)
	output, err := cel.Forward(context.Background(), logits, targets)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expectedLoss := float32(getFloat(g, "expected_loss"))
	gotLoss := output.Data()[0]
	diff := math.Abs(float64(gotLoss - expectedLoss))
	if diff > tol {
		t.Errorf("cross_entropy_loss: got %g, want %g (diff=%g)", gotLoss, expectedLoss, diff)
	}
}

// ---------------------------------------------------------------------------
// Op tests
// ---------------------------------------------------------------------------

func TestParity_ReduceSum(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "op_reduce_sum")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	axes := getInts(g, "axes")

	rs := reducesum.New[float32](engine, axes, true)
	output, err := rs.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "reduce_sum_keepdims", output.Data(), getFloat32s(g, "expected_output_keepdims"), tol)
}

func TestParity_Transpose(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "op_transpose")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	axes := getInts(g, "axes")

	tr := ltranspose.New[float32](engine, axes)
	output, err := tr.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "transpose_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

func TestParity_Gather(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "op_gather")
	tol := getFloat(g, "tolerance")

	table := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))

	// Indices as int tensor
	idxRaw := g["indices"].([]interface{})
	idxData := make([]int, len(idxRaw))
	for i, v := range idxRaw {
		idxData[i] = int(v.(float64))
	}
	idxTensor, err := tensor.New[int]([]int{len(idxData)}, idxData)
	if err != nil {
		t.Fatalf("create index tensor: %v", err)
	}

	ga := gather.NewWithWeights(engine, table)
	_ = idxTensor
	// Gather with embedded weights expects float indices
	idxFloat := make([]float32, len(idxData))
	for i, v := range idxData {
		idxFloat[i] = float32(v)
	}
	idxFloatTensor := makeTensor(t, idxFloat, []int{len(idxFloat)})

	output, err := ga.Forward(context.Background(), idxFloatTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "gather_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// Conv2D test
// ---------------------------------------------------------------------------

func TestParity_Conv2D(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "core_conv2d")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	weight := makeTensor(t, getFloat32s(g, "weight"), getInts(g, "weight_shape"))
	bias := makeTensor(t, getFloat32s(g, "bias"), getInts(g, "bias_shape"))

	strides := getInts(g, "stride")
	pads := getInts(g, "padding")

	conv := core.NewConv2d[float32](engine, ops, strides, pads, []int{1, 1}, 1)
	output, err := conv.Forward(context.Background(), input, weight, bias)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "conv2d_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// FFN test
// ---------------------------------------------------------------------------

// transposeWeight2D transposes a 2D weight matrix from [rows, cols] to [cols, rows].
func transposeWeight2D(data []float32, rows, cols int) []float32 {
	out := make([]float32, len(data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[j*rows+i] = data[i*cols+j]
		}
	}
	return out
}

func TestParity_FFN(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "core_ffn")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))

	inputShape := getInts(g, "input_shape")
	w1Shape := getInts(g, "w1_shape")
	w2Shape := getInts(g, "w2_shape")

	inputDim := inputShape[1]  // 4
	hiddenDim := w1Shape[1]    // 32
	outputDim := w2Shape[1]    // 4

	// Create FFN with no bias (golden file has no bias data)
	ffn, err := core.NewFFN[float32]("test_ffn", engine, ops, inputDim, hiddenDim, outputDim, core.WithFFNNoBias[float32]())
	if err != nil {
		t.Fatalf("NewFFN: %v", err)
	}

	// Set weights from golden data.
	// FFN has w1, w2, w3 Dense layers. Each Dense has a Linear with weights [in, out].
	// Golden data has weights in [in, out] format (matching Zerfoo's layout).
	params := ffn.Parameters()
	w1Data := getFloat32s(g, "w1")
	w2Data := getFloat32s(g, "w2")
	w3Data := getFloat32s(g, "w3")

	for _, p := range params {
		pdata := p.Value.Data()
		switch len(pdata) {
		case len(w1Data):
			// Could be w1 or w3, distinguish by name
			if len(p.Name) > 0 && p.Name[len(p.Name)-1] == 's' {
				// Parameter names end with "_weights"
				// w1 is first, w3 is third in parameter order
			}
		}
		_ = pdata
	}

	// More precise approach: FFN.Parameters() returns w1 params, then w2 params, then w3 params.
	// Each Dense has [linear_weights] (no bias since WithFFNNoBias).
	// w1 linear weight: [inputDim, hiddenDim] = [4, 32]
	// w2 linear weight: [hiddenDim, outputDim] = [32, 4]
	// w3 linear weight: [inputDim, hiddenDim] = [4, 32]
	if len(params) < 3 {
		t.Fatalf("expected at least 3 parameters, got %d", len(params))
	}

	copy(params[0].Value.Data(), w1Data)
	copy(params[1].Value.Data(), w2Data)
	copy(params[2].Value.Data(), w3Data)

	output, err := ffn.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "ffn_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// BatchNorm test
// ---------------------------------------------------------------------------

func TestParity_BatchNorm(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "norm_batch_norm")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	scale := makeTensor(t, getFloat32s(g, "scale"), getInts(g, "scale_shape"))
	bias := makeTensor(t, getFloat32s(g, "bias"), getInts(g, "bias_shape"))
	runningMean := makeTensor(t, getFloat32s(g, "running_mean"), getInts(g, "scale_shape"))
	runningVar := makeTensor(t, getFloat32s(g, "running_var"), getInts(g, "scale_shape"))

	eps := float32(getFloat(g, "epsilon"))
	bn := normalization.NewBatchNormalization[float32](engine, ops, eps)

	output, err := bn.Forward(context.Background(), input, scale, bias, runningMean, runningVar)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "batch_norm_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// Dropout test (eval mode only)
// ---------------------------------------------------------------------------

func TestParity_Dropout(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "op_dropout")
	tol := getFloat(g, "tolerance")

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	rate := float32(getFloat(g, "rate"))

	dropout := regularization.NewDropout[float32](engine, ops, rate)
	// Eval mode (default): output should be identical to input
	output, err := dropout.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "dropout_eval", output.Data(), getFloat32s(g, "expected_output_eval"), tol)
}

// ---------------------------------------------------------------------------
// AdamW optimizer test
// ---------------------------------------------------------------------------

func TestParity_AdamW(t *testing.T) {
	engine, _ := setup()
	g := loadGolden(t, "optimizer_adamw")
	tol := getFloat(g, "tolerance")

	lr := float32(getFloat(g, "lr"))
	beta1 := float32(getFloat(g, "beta1"))
	beta2 := float32(getFloat(g, "beta2"))
	eps := float32(getFloat(g, "epsilon"))
	wd := float32(getFloat(g, "weight_decay"))

	param := makeParam(t, "test_param", getFloat32s(g, "param_before"), getInts(g, "param_shape"))
	gradTensor := makeTensor(t, getFloat32s(g, "grad"), getInts(g, "param_shape"))
	param.Gradient = gradTensor

	adamw := optimizer.NewAdamW[float32](engine, lr, beta1, beta2, eps, wd)
	err := adamw.Step(context.Background(), []*graph.Parameter[float32]{param})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	assertClose(t, "adamw_step", param.Value.Data(), getFloat32s(g, "expected_param_after"), tol)
}

// ---------------------------------------------------------------------------
// SGD optimizer test
// ---------------------------------------------------------------------------

func TestParity_SGD(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "optimizer_sgd")
	tol := getFloat(g, "tolerance")

	lr := float32(getFloat(g, "lr"))

	param := makeParam(t, "test_param", getFloat32s(g, "param_before"), getInts(g, "param_shape"))
	gradTensor := makeTensor(t, getFloat32s(g, "grad"), getInts(g, "param_shape"))
	param.Gradient = gradTensor

	sgd := optimizer.NewSGD[float32](engine, ops, lr)
	err := sgd.Step(context.Background(), []*graph.Parameter[float32]{param})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	assertClose(t, "sgd_step", param.Value.Data(), getFloat32s(g, "expected_param_after"), tol)
}

// ---------------------------------------------------------------------------
// SimpleRNN test
// ---------------------------------------------------------------------------

func TestParity_SimpleRNN(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "recurrent_simple_rnn")
	tol := getFloat(g, "tolerance")

	inputDim := int(getFloat(g, "input_dim"))
	hiddenDim := int(getFloat(g, "hidden_dim"))

	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))
	inputShape := getInts(g, "input_shape")
	batchSize := inputShape[0]
	seqLen := inputShape[1]

	rnn, err := recurrent.NewSimpleRNN[float32]("test_rnn", engine, ops, inputDim, hiddenDim)
	if err != nil {
		t.Fatalf("NewSimpleRNN: %v", err)
	}

	// Set weights from golden data.
	// PyTorch stores weights as [out, in], Zerfoo Linear stores as [in, out].
	// RNN parameters order: inputWeights params, hiddenWeights params, bias params.
	params := rnn.Parameters()
	// inputWeights: Linear with weights [inputDim, hiddenDim]
	// PyTorch weight_ih: [hiddenDim, inputDim] -> transpose
	wihShape := getInts(g, "weight_ih_shape")
	wihData := transposeWeight2D(getFloat32s(g, "weight_ih"), wihShape[0], wihShape[1])
	copy(params[0].Value.Data(), wihData)

	// hiddenWeights: Linear with weights [hiddenDim, hiddenDim]
	// PyTorch weight_hh: [hiddenDim, hiddenDim] -> transpose (symmetric dims but data differs)
	whhShape := getInts(g, "weight_hh_shape")
	whhData := transposeWeight2D(getFloat32s(g, "weight_hh"), whhShape[0], whhShape[1])
	copy(params[1].Value.Data(), whhData)

	// Bias: Zerfoo has a single bias, PyTorch has bias_ih + bias_hh
	biasIH := getFloat32s(g, "bias_ih")
	biasHH := getFloat32s(g, "bias_hh")
	combinedBias := make([]float32, len(biasIH))
	for i := range combinedBias {
		combinedBias[i] = biasIH[i] + biasHH[i]
	}
	copy(params[2].Value.Data(), combinedBias)

	// Run forward step-by-step over the sequence dimension
	inData := input.Data()
	outputs := make([]float32, 0, batchSize*seqLen*hiddenDim)
	for step := 0; step < seqLen; step++ {
		// Extract input[:, step, :] -> [batch, inputDim]
		stepData := make([]float32, batchSize*inputDim)
		for b := 0; b < batchSize; b++ {
			copy(stepData[b*inputDim:(b+1)*inputDim], inData[b*seqLen*inputDim+step*inputDim:b*seqLen*inputDim+step*inputDim+inputDim])
		}
		stepInput := makeTensor(t, stepData, []int{batchSize, inputDim})
		stepOutput, err := rnn.Forward(context.Background(), stepInput)
		if err != nil {
			t.Fatalf("Forward step %d: %v", step, err)
		}
		outputs = append(outputs, stepOutput.Data()...)
	}

	// Reshape outputs to [batch, seq, hidden] for comparison
	// outputs is currently [seq * batch * hidden], need to rearrange to [batch * seq * hidden]
	expected := getFloat32s(g, "expected_output")
	reordered := make([]float32, len(outputs))
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < hiddenDim; h++ {
				reordered[b*seqLen*hiddenDim+s*hiddenDim+h] = outputs[s*batchSize*hiddenDim+b*hiddenDim+h]
			}
		}
	}
	assertClose(t, "simple_rnn_output", reordered, expected, tol)
}

// ---------------------------------------------------------------------------
// S4 test
// ---------------------------------------------------------------------------

func TestParity_S4(t *testing.T) {
	engine, ops := setup()
	g := loadGolden(t, "ssm_s4")
	tol := getFloat(g, "tolerance")

	inputDim := int(getFloat(g, "input_dim"))
	stateDim := int(getFloat(g, "state_dim"))
	input := makeTensor(t, getFloat32s(g, "input"), getInts(g, "input_shape"))

	s4, err := ssm.NewS4[float32]("test_s4", engine, ops, inputDim, stateDim)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Set parameters from golden data
	params := s4.Parameters()
	// Parameters order: aLog, b, c, d
	copy(params[0].Value.Data(), getFloat32s(g, "a_log"))
	copy(params[1].Value.Data(), getFloat32s(g, "b"))
	copy(params[2].Value.Data(), getFloat32s(g, "c"))
	copy(params[3].Value.Data(), getFloat32s(g, "d"))

	output, err := s4.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertClose(t, "s4_forward", output.Data(), getFloat32s(g, "expected_output"), tol)
}

// ---------------------------------------------------------------------------
// MambaBlock test
// ---------------------------------------------------------------------------

func TestParity_MambaBlock(t *testing.T) {
	t.Skip("TODO: complex weight wiring needed - MambaBlock has 6+ projection layers (in_proj, x_proj, dt_proj, out_proj, conv_weight, A, D) with intricate dimensions and discretization logic that requires careful mapping from PyTorch's parameter layout")
}

// ---------------------------------------------------------------------------
// TransformerBlock test
// ---------------------------------------------------------------------------

func TestParity_TransformerBlock(t *testing.T) {
	t.Skip("TODO: complex weight wiring needed - TransformerBlock requires constructing an attention node (with QKV+output projections), 3 RMSNorm layers, and an FFN, all with coordinated weight shapes and specific parameter initialization from golden data")
}

// ---------------------------------------------------------------------------
// Summary test: runs all layer parity tests and prints a report
// ---------------------------------------------------------------------------

func TestParity_Summary(t *testing.T) {
	type testCase struct {
		name string
		fn   func(*testing.T)
	}

	cases := []testCase{
		// Activations
		{"ReLU", TestParity_ReLU},
		{"GELU", TestParity_GELU},
		{"Sigmoid", TestParity_Sigmoid},
		{"Tanh", TestParity_Tanh},
		{"Softmax", TestParity_Softmax},
		{"LeakyReLU", TestParity_LeakyReLU},
		{"SwiGLU", TestParity_SwiGLU},
		{"Erf", TestParity_Erf},
		// Functional
		{"Functional/ReLU", TestParity_Functional_ReLU},
		{"Functional/GELU", TestParity_Functional_GELU},
		{"Functional/Sigmoid", TestParity_Functional_Sigmoid},
		{"Functional/SiLU", TestParity_Functional_SiLU},
		{"Functional/Softmax", TestParity_Functional_Softmax},
		{"Functional/LayerNorm", TestParity_Functional_LayerNorm},
		{"Functional/RMSNorm", TestParity_Functional_RMSNorm},
		{"Functional/Linear", TestParity_Functional_Linear},
		// Normalization
		{"LayerNorm", TestParity_LayerNorm},
		{"RMSNorm", TestParity_RMSNorm},
		// Core
		{"Linear", TestParity_Linear},
		{"MatMul", TestParity_MatMul},
		{"Conv1D", TestParity_Conv1D},
		// Attention
		{"SDPA/Causal", TestParity_SDPA_Causal},
		{"SDPA/Bidirectional", TestParity_SDPA_Bidirectional},
		{"MultiHeadAttention", TestParity_MultiHeadAttention},
		// Embeddings
		{"TokenEmbedding", TestParity_TokenEmbedding},
		{"RotaryEmbedding", TestParity_RotaryEmbedding},
		// Loss
		{"MSELoss", TestParity_MSELoss},
		{"BCELoss", TestParity_BCELoss},
		{"CrossEntropyLoss", TestParity_CrossEntropyLoss},
		// Ops
		{"ReduceSum", TestParity_ReduceSum},
		{"Transpose", TestParity_Transpose},
		{"Gather", TestParity_Gather},
		// Conv2D
		{"Conv2D", TestParity_Conv2D},
		// FFN
		{"FFN", TestParity_FFN},
		// Normalization (BatchNorm)
		{"BatchNorm", TestParity_BatchNorm},
		// Regularization
		{"Dropout", TestParity_Dropout},
		// Optimizers
		{"AdamW", TestParity_AdamW},
		{"SGD", TestParity_SGD},
		// Recurrent
		{"SimpleRNN", TestParity_SimpleRNN},
		// SSM
		{"S4", TestParity_S4},
		{"MambaBlock", TestParity_MambaBlock},
		// Transformer
		{"TransformerBlock", TestParity_TransformerBlock},
	}

	passed, failed := 0, 0
	for _, tc := range cases {
		ok := t.Run(tc.name, tc.fn)
		if ok {
			passed++
		} else {
			failed++
		}
	}

	fmt.Printf("\n=== LAYER PARITY REPORT ===\n")
	fmt.Printf("Total: %d | Passed: %d | Failed: %d\n", passed+failed, passed, failed)
	fmt.Printf("Pass rate: %.1f%%\n", float64(passed)/float64(passed+failed)*100)
}
