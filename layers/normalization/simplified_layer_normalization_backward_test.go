package normalization_test

import (
    "context"
    "math"
    "testing"

    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/layers/normalization"
    "github.com/zerfoo/zerfoo/numeric"
    "github.com/zerfoo/zerfoo/tensor"
    "github.com/zerfoo/zerfoo/testing/testutils"
)

// TestSimplifiedLayerNormalization_Backward validates gradients via finite differences
func TestSimplifiedLayerNormalization_Backward(t *testing.T) {
    ctx := context.Background()
    ops := numeric.Float32Ops{}
    engine := compute.NewCPUEngine[float32](&ops)

    // Input tensor [batch=1, dim=4]
    inputData := []float32{1, 2, 3, 4}
    input, _ := tensor.New[float32]([]int{1, 4}, inputData)

    // Upstream gradient (same shape as output)
    dOutData := []float32{0.1, -0.2, 0.3, -0.4}
    dOut, _ := tensor.New[float32]([]int{1, 4}, dOutData)

    // Gain parameter matching the last dimension [4]
    gainData := []float32{1.5, 1.5, 1.5, 1.5}
    gain, _ := tensor.New[float32]([]int{4}, gainData)

    epsilon := float32(1e-5)

    sln, err := normalization.NewSimplifiedLayerNormalization[float32](engine, &ops, gain, epsilon)
    testutils.AssertNoError(t, err, "NewSimplifiedLayerNormalization failed")

    // Analytical gradients
    _, err = sln.Forward(ctx, input)
    testutils.AssertNoError(t, err, "forward failed")
    grads, err := sln.Backward(ctx, dOut, input)
    testutils.AssertNoError(t, err, "backward failed")
    dInput := grads[0]
    dGainAnalytical := sln.Parameters()[0].Gradient

    // Numerical gradient for input
    eps := float32(1e-3)
    dInputNum := make([]float32, len(inputData))
    for i := range inputData {
        // Perturb input
        irow, icol := 0, i // shape [1,4]
        idx := irow*4 + icol
        orig := input.Data()[idx]

        input.Data()[idx] = orig + eps
        outPos, err := sln.Forward(ctx, input)
        testutils.AssertNoError(t, err, "forward+ failed")
        lp := dot(outPos.Data(), dOutData)

        input.Data()[idx] = orig - eps
        outNeg, err := sln.Forward(ctx, input)
        testutils.AssertNoError(t, err, "forward- failed")
        ln := dot(outNeg.Data(), dOutData)

        num := (lp - ln) / (2 * eps)
        dInputNum[i] = num

        // Restore
        input.Data()[idx] = orig
    }

    if !approxSlice(dInput.Data(), dInputNum, 2e-2) { // finite diff tolerance
        t.Fatalf("dInput mismatch:\nanalytical=%v\nnumerical=%v", dInput.Data(), dInputNum)
    }

    // Numerical gradient for gain
    dGainNum := make([]float32, len(gainData))
    for i := range gainData {
        orig := gain.Data()[i]

        gain.Data()[i] = orig + eps
        outPos, err := sln.Forward(ctx, input)
        testutils.AssertNoError(t, err, "forward+ gain failed")
        lp := dot(outPos.Data(), dOutData)

        gain.Data()[i] = orig - eps
        outNeg, err := sln.Forward(ctx, input)
        testutils.AssertNoError(t, err, "forward- gain failed")
        ln := dot(outNeg.Data(), dOutData)

        dGainNum[i] = (lp - ln) / (2 * eps)
        gain.Data()[i] = orig
    }

    if !approxSlice(dGainAnalytical.Data(), dGainNum, 2e-2) {
        t.Fatalf("dGain mismatch:\nanalytical=%v\nnumerical=%v", dGainAnalytical.Data(), dGainNum)
    }
}

func dot(a, b []float32) float32 {
    var s float32
    for i := range a {
        s += a[i] * b[i]
    }
    return s
}

func approxSlice(a, b []float32, tol float32) bool {
    if len(a) != len(b) {
        return false
    }
    for i := range a {
        if float32(math.Abs(float64(a[i]-b[i]))) > tol {
            return false
        }
    }
    return true
}
