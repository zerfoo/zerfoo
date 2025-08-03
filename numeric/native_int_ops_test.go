package numeric_test

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestIntOps_Add(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 5, ops.Add(2, 3), "2 + 3 should be 5")
}

func TestIntOps_Sub(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, -1, ops.Sub(2, 3), "2 - 3 should be -1")
}

func TestIntOps_Mul(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 6, ops.Mul(2, 3), "2 * 3 should be 6")
}

func TestIntOps_Div(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 2, ops.Div(6, 3), "6 / 3 should be 2")
	testutils.AssertEqual(t, 0, ops.Div(1, 0), "1 / 0 should be 0")
}

func TestIntOps_FromFloat32(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 3, ops.FromFloat32(3.14), "float32(3.14) should be 3")
}

func TestIntOps_ToFloat32(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, float32(3), ops.ToFloat32(3), "int(3) should be 3.0")
}

func TestIntOps_Tanh(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, int(math.Tanh(2)), ops.Tanh(2), "tanh(2) should be correct")
}

func TestIntOps_Sigmoid(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, int(1.0/(1.0+math.Exp(-2.0))), ops.Sigmoid(2), "sigmoid(2) should be correct")
}

func TestIntOps_ReLU(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 2, ops.ReLU(2), "ReLU(2) should be 2")
	testutils.AssertEqual(t, 0, ops.ReLU(-2), "ReLU(-2) should be 0")
}

func TestIntOps_LeakyReLU(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 2, ops.LeakyReLU(2, 0.1), "LeakyReLU(2) should be 2")
	testutils.AssertEqual(t, -1, ops.LeakyReLU(-10, 0.1), "LeakyReLU(-10) should be -1")
}

func TestIntOps_TanhGrad(t *testing.T) {
	ops := numeric.IntOps{}
	tanhX := int(math.Tanh(2))
	expected := 1 - (tanhX * tanhX)
	testutils.AssertEqual(t, expected, ops.TanhGrad(2), "tanhgrad(2) should be correct")
}

func TestIntOps_SigmoidGrad(t *testing.T) {
	ops := numeric.IntOps{}
	sigX := int(1.0 / (1.0 + math.Exp(-2.0)))
	expected := sigX * (1 - sigX)
	testutils.AssertEqual(t, expected, ops.SigmoidGrad(2), "sigmoidgrad(2) should be correct")
}

func TestIntOps_ReLUGrad(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 1, ops.ReLUGrad(2), "ReLUGrad(2) should be 1")
	testutils.AssertEqual(t, 0, ops.ReLUGrad(-2), "ReLUGrad(-2) should be 0")
}

func TestIntOps_LeakyReLUGrad(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 1, ops.LeakyReLUGrad(2, 0.1), "LeakyReLUGrad(2) should be 1")
	testutils.AssertEqual(t, 0, ops.LeakyReLUGrad(-2, 0.1), "LeakyReLUGrad(-2) should be 0")
}

func TestIntOps_IsZero(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertTrue(t, ops.IsZero(0), "IsZero(0) should be true")
	testutils.AssertFalse(t, ops.IsZero(1), "IsZero(1) should be false")
}

func TestIntOps_Exp(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, int(math.Exp(2)), ops.Exp(2), "Exp(2) should be correct")
}

func TestIntOps_Log(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, int(math.Log(2)), ops.Log(2), "Log(2) should be correct")
}

func TestIntOps_Pow(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 8, ops.Pow(2, 3), "Pow(2, 3) should be 8")
}

func TestIntOps_Abs(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 2, ops.Abs(2), "Abs(2) should be 2")
	testutils.AssertEqual(t, 2, ops.Abs(-2), "Abs(-2) should be 2")
}

func TestIntOps_Sum(t *testing.T) {
	ops := numeric.IntOps{}
	testutils.AssertEqual(t, 6, ops.Sum([]int{1, 2, 3}), "Sum of [1, 2, 3] should be 6")
}
