package numeric

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
)

// ---------- Int8Ops ----------

func TestInt8Ops_AllMethods(t *testing.T) {
	ops := Int8Ops{}

	t.Run("Mul", func(t *testing.T) {
		if ops.Mul(3, 4) != 12 {
			t.Errorf("Mul(3,4) = %d, want 12", ops.Mul(3, 4))
		}
	})

	t.Run("Div", func(t *testing.T) {
		if ops.Div(10, 2) != 5 {
			t.Errorf("Div(10,2) = %d, want 5", ops.Div(10, 2))
		}
		if ops.Div(5, 0) != 0 {
			t.Error("Div by zero should return 0")
		}
	})

	t.Run("Tanh", func(t *testing.T) {
		got := ops.Tanh(0)
		if got != 0 {
			t.Errorf("Tanh(0) = %d, want 0", got)
		}
	})

	t.Run("Sigmoid", func(t *testing.T) {
		got := ops.Sigmoid(0)
		if got != 0 { // int8(0.5) = 0
			t.Errorf("Sigmoid(0) = %d, want 0", got)
		}
	})

	t.Run("ReLU", func(t *testing.T) {
		if ops.ReLU(5) != 5 {
			t.Error("ReLU(5) should be 5")
		}
		if ops.ReLU(-3) != 0 {
			t.Error("ReLU(-3) should be 0")
		}
		if ops.ReLU(0) != 0 {
			t.Error("ReLU(0) should be 0")
		}
	})

	t.Run("LeakyReLU", func(t *testing.T) {
		if ops.LeakyReLU(5, 0.1) != 5 {
			t.Error("LeakyReLU(5,0.1) should be 5")
		}
		got := ops.LeakyReLU(-10, 0.1)
		want := int8(float64(-10) * 0.1)
		if got != want {
			t.Errorf("LeakyReLU(-10,0.1) = %d, want %d", got, want)
		}
	})

	t.Run("TanhGrad", func(t *testing.T) {
		got := ops.TanhGrad(0)
		if got != 1 { // 1 - tanh(0)^2 = 1
			t.Errorf("TanhGrad(0) = %d, want 1", got)
		}
	})

	t.Run("SigmoidGrad", func(t *testing.T) {
		_ = ops.SigmoidGrad(0)
		// Just verify no panic
	})

	t.Run("ReLUGrad", func(t *testing.T) {
		if ops.ReLUGrad(5) != 1 {
			t.Error("ReLUGrad(5) should be 1")
		}
		if ops.ReLUGrad(-3) != 0 {
			t.Error("ReLUGrad(-3) should be 0")
		}
	})

	t.Run("LeakyReLUGrad", func(t *testing.T) {
		if ops.LeakyReLUGrad(5, 0.1) != 1 {
			t.Error("LeakyReLUGrad(5,0.1) should be 1")
		}
		got := ops.LeakyReLUGrad(-3, 0.1)
		if got != 0 { // int8(0.1) truncates to 0
			t.Errorf("LeakyReLUGrad(-3,0.1) = %d, want 0", got)
		}
	})

	t.Run("FromFloat32", func(t *testing.T) {
		if ops.FromFloat32(42.9) != 42 {
			t.Errorf("FromFloat32(42.9) = %d", ops.FromFloat32(42.9))
		}
	})

	t.Run("FromFloat64", func(t *testing.T) {
		if ops.FromFloat64(42.9) != 42 {
			t.Errorf("FromFloat64(42.9) = %d", ops.FromFloat64(42.9))
		}
	})

	t.Run("One", func(t *testing.T) {
		if ops.One() != 1 {
			t.Error("One() should be 1")
		}
	})

	t.Run("IsZero", func(t *testing.T) {
		if !ops.IsZero(0) {
			t.Error("IsZero(0) should be true")
		}
		if ops.IsZero(1) {
			t.Error("IsZero(1) should be false")
		}
	})

	t.Run("Abs", func(t *testing.T) {
		if ops.Abs(-5) != 5 {
			t.Error("Abs(-5) should be 5")
		}
		if ops.Abs(5) != 5 {
			t.Error("Abs(5) should be 5")
		}
	})

	t.Run("Sum", func(t *testing.T) {
		if ops.Sum([]int8{1, 2, 3}) != 6 {
			t.Errorf("Sum([1,2,3]) = %d", ops.Sum([]int8{1, 2, 3}))
		}
	})

	t.Run("Exp", func(t *testing.T) {
		got := ops.Exp(0)
		if got != 1 {
			t.Errorf("Exp(0) = %d, want 1", got)
		}
	})

	t.Run("Log", func(t *testing.T) {
		got := ops.Log(1)
		if got != 0 {
			t.Errorf("Log(1) = %d, want 0", got)
		}
	})

	t.Run("Pow", func(t *testing.T) {
		if ops.Pow(2, 3) != 8 {
			t.Errorf("Pow(2,3) = %d, want 8", ops.Pow(2, 3))
		}
	})

	t.Run("Sqrt", func(t *testing.T) {
		if ops.Sqrt(9) != 3 {
			t.Errorf("Sqrt(9) = %d, want 3", ops.Sqrt(9))
		}
	})

	t.Run("GreaterThan", func(t *testing.T) {
		if !ops.GreaterThan(5, 3) {
			t.Error("5 > 3 should be true")
		}
		if ops.GreaterThan(3, 5) {
			t.Error("3 > 5 should be false")
		}
	})
}

// ---------- Uint8Ops ----------

func TestUint8Ops_AllMethods(t *testing.T) {
	ops := Uint8Ops{}

	t.Run("Tanh", func(t *testing.T) {
		_ = ops.Tanh(0)
		_ = ops.Tanh(1)
	})

	t.Run("Sigmoid", func(t *testing.T) {
		_ = ops.Sigmoid(0)
		_ = ops.Sigmoid(1)
	})

	t.Run("ReLU", func(t *testing.T) {
		if ops.ReLU(5) != 5 {
			t.Error("ReLU(5) should be 5")
		}
		if ops.ReLU(0) != 0 {
			t.Error("ReLU(0) should be 0")
		}
	})

	t.Run("LeakyReLU", func(t *testing.T) {
		if ops.LeakyReLU(5, 0.1) != 5 {
			t.Error("LeakyReLU(5,0.1) should be 5")
		}
		if ops.LeakyReLU(0, 0.1) != 0 {
			t.Error("LeakyReLU(0,0.1) should be 0")
		}
	})

	t.Run("TanhGrad", func(t *testing.T) {
		got := ops.TanhGrad(0)
		if got != 1 {
			t.Errorf("TanhGrad(0) = %d, want 1", got)
		}
	})

	t.Run("SigmoidGrad", func(t *testing.T) {
		_ = ops.SigmoidGrad(0)
	})

	t.Run("ReLUGrad", func(t *testing.T) {
		if ops.ReLUGrad(5) != 1 {
			t.Error("ReLUGrad(5) should be 1")
		}
		if ops.ReLUGrad(0) != 0 {
			t.Error("ReLUGrad(0) should be 0")
		}
	})

	t.Run("LeakyReLUGrad", func(t *testing.T) {
		if ops.LeakyReLUGrad(5, 0.1) != 1 {
			t.Error("LeakyReLUGrad(5,0.1) should be 1")
		}
		got := ops.LeakyReLUGrad(0, 0.1)
		if got != 0 { // uint8(0.1) truncates to 0
			t.Errorf("LeakyReLUGrad(0,0.1) = %d, want 0", got)
		}
	})

	t.Run("FromFloat32", func(t *testing.T) {
		if ops.FromFloat32(42.9) != 42 {
			t.Errorf("FromFloat32(42.9) = %d", ops.FromFloat32(42.9))
		}
	})

	t.Run("FromFloat64", func(t *testing.T) {
		if ops.FromFloat64(42.9) != 42 {
			t.Errorf("FromFloat64(42.9) = %d", ops.FromFloat64(42.9))
		}
	})

	t.Run("One", func(t *testing.T) {
		if ops.One() != 1 {
			t.Error("One() should be 1")
		}
	})

	t.Run("IsZero", func(t *testing.T) {
		if !ops.IsZero(0) {
			t.Error("IsZero(0) should be true")
		}
		if ops.IsZero(1) {
			t.Error("IsZero(1) should be false")
		}
	})

	t.Run("Abs", func(t *testing.T) {
		if ops.Abs(5) != 5 {
			t.Error("Abs(5) should be 5")
		}
		if ops.Abs(0) != 0 {
			t.Error("Abs(0) should be 0")
		}
	})

	t.Run("Sum", func(t *testing.T) {
		if ops.Sum([]uint8{1, 2, 3}) != 6 {
			t.Errorf("Sum([1,2,3]) = %d", ops.Sum([]uint8{1, 2, 3}))
		}
	})

	t.Run("Exp", func(t *testing.T) {
		got := ops.Exp(0)
		if got != 1 {
			t.Errorf("Exp(0) = %d, want 1", got)
		}
	})

	t.Run("Log", func(t *testing.T) {
		got := ops.Log(1)
		if got != 0 {
			t.Errorf("Log(1) = %d, want 0", got)
		}
	})

	t.Run("Pow", func(t *testing.T) {
		if ops.Pow(2, 3) != 8 {
			t.Errorf("Pow(2,3) = %d, want 8", ops.Pow(2, 3))
		}
	})

	t.Run("Sqrt", func(t *testing.T) {
		if ops.Sqrt(9) != 3 {
			t.Errorf("Sqrt(9) = %d, want 3", ops.Sqrt(9))
		}
	})
}

// ---------- Float16Ops missing methods ----------

func TestFloat16Ops_Sqrt(t *testing.T) {
	ops := Float16Ops{}
	x := float16.FromFloat32(4.0)
	got := ops.Sqrt(x).ToFloat32()
	if math.Abs(float64(got-2.0)) > 0.1 {
		t.Errorf("Sqrt(4) = %f, want ~2.0", got)
	}
}

func TestFloat16Ops_GreaterThan(t *testing.T) {
	ops := Float16Ops{}
	a := float16.FromFloat32(3.0)
	b := float16.FromFloat32(2.0)
	if !ops.GreaterThan(a, b) {
		t.Error("3 > 2 should be true")
	}
	if ops.GreaterThan(b, a) {
		t.Error("2 > 3 should be false")
	}
}

func TestFloat16Ops_One(t *testing.T) {
	ops := Float16Ops{}
	got := ops.One().ToFloat32()
	if got != 1.0 {
		t.Errorf("One() = %f, want 1.0", got)
	}
}

func TestFloat16Ops_FromFloat64(t *testing.T) {
	ops := Float16Ops{}
	got := ops.FromFloat64(3.14).ToFloat32()
	if math.Abs(float64(got-3.14)) > 0.01 {
		t.Errorf("FromFloat64(3.14) = %f", got)
	}
}

// ---------- Float8Ops missing methods ----------

func TestFloat8Ops_Sqrt(t *testing.T) {
	ops := Float8Ops{}
	x := float8.ToFloat8(4.0)
	got := ops.Sqrt(x).ToFloat32()
	if math.Abs(float64(got-2.0)) > 0.5 {
		t.Errorf("Sqrt(4) = %f, want ~2.0", got)
	}
}

func TestFloat8Ops_GreaterThan(t *testing.T) {
	ops := Float8Ops{}
	a := float8.ToFloat8(3.0)
	b := float8.ToFloat8(1.0)
	if !ops.GreaterThan(a, b) {
		t.Error("3 > 1 should be true")
	}
	if ops.GreaterThan(b, a) {
		t.Error("1 > 3 should be false")
	}
}

func TestFloat8Ops_One(t *testing.T) {
	ops := Float8Ops{}
	got := ops.One().ToFloat32()
	if got != 1.0 {
		t.Errorf("One() = %f, want 1.0", got)
	}
}

func TestFloat8Ops_FromFloat64(t *testing.T) {
	ops := Float8Ops{}
	got := ops.FromFloat64(2.0).ToFloat32()
	if math.Abs(float64(got-2.0)) > 0.5 {
		t.Errorf("FromFloat64(2.0) = %f", got)
	}
}

// ---------- Float64Ops ----------

func TestFloat64Ops_AllMethods(t *testing.T) {
	ops := Float64Ops{}

	t.Run("Add", func(t *testing.T) {
		if ops.Add(1.5, 2.5) != 4.0 {
			t.Error("Add failed")
		}
	})
	t.Run("Sub", func(t *testing.T) {
		if ops.Sub(5.0, 3.0) != 2.0 {
			t.Error("Sub failed")
		}
	})
	t.Run("Mul", func(t *testing.T) {
		if ops.Mul(2.0, 3.0) != 6.0 {
			t.Error("Mul failed")
		}
	})
	t.Run("Div", func(t *testing.T) {
		if ops.Div(6.0, 2.0) != 3.0 {
			t.Error("Div failed")
		}
		if ops.Div(1.0, 0.0) != 0 {
			t.Error("Div by zero should return 0")
		}
	})
	t.Run("Tanh", func(t *testing.T) {
		if ops.Tanh(0) != 0 {
			t.Error("Tanh(0) should be 0")
		}
	})
	t.Run("Sigmoid", func(t *testing.T) {
		if ops.Sigmoid(0) != 0.5 {
			t.Error("Sigmoid(0) should be 0.5")
		}
	})
	t.Run("TanhGrad", func(t *testing.T) {
		if ops.TanhGrad(0) != 1.0 {
			t.Errorf("TanhGrad(0) = %f, want 1.0", ops.TanhGrad(0))
		}
	})
	t.Run("SigmoidGrad", func(t *testing.T) {
		if ops.SigmoidGrad(0) != 0.25 {
			t.Errorf("SigmoidGrad(0) = %f, want 0.25", ops.SigmoidGrad(0))
		}
	})
	t.Run("ReLU", func(t *testing.T) {
		if ops.ReLU(5) != 5.0 {
			t.Error("ReLU(5) should be 5")
		}
		if ops.ReLU(-3) != 0 {
			t.Error("ReLU(-3) should be 0")
		}
	})
	t.Run("LeakyReLU", func(t *testing.T) {
		if ops.LeakyReLU(5, 0.1) != 5.0 {
			t.Error("LeakyReLU(5,0.1) should be 5")
		}
		got := ops.LeakyReLU(-3, 0.1)
		if math.Abs(got-(-0.3)) > 1e-10 {
			t.Errorf("LeakyReLU(-3,0.1) = %f, want ~-0.3", got)
		}
	})
	t.Run("ReLUGrad", func(t *testing.T) {
		if ops.ReLUGrad(5) != 1.0 {
			t.Error("ReLUGrad(5) should be 1")
		}
		if ops.ReLUGrad(-3) != 0 {
			t.Error("ReLUGrad(-3) should be 0")
		}
	})
	t.Run("LeakyReLUGrad", func(t *testing.T) {
		if ops.LeakyReLUGrad(5, 0.1) != 1.0 {
			t.Error("LeakyReLUGrad(5,0.1) should be 1")
		}
		if ops.LeakyReLUGrad(-3, 0.1) != 0.1 {
			t.Errorf("LeakyReLUGrad(-3,0.1) = %f", ops.LeakyReLUGrad(-3, 0.1))
		}
	})
	t.Run("FromFloat32", func(t *testing.T) {
		if ops.FromFloat32(3.14) != float64(float32(3.14)) {
			t.Error("FromFloat32 failed")
		}
	})
	t.Run("FromFloat64", func(t *testing.T) {
		if ops.FromFloat64(3.14) != 3.14 {
			t.Error("FromFloat64 failed")
		}
	})
	t.Run("ToFloat32", func(t *testing.T) {
		if ops.ToFloat32(3.14) != float32(3.14) {
			t.Error("ToFloat32 failed")
		}
	})
	t.Run("IsZero", func(t *testing.T) {
		if !ops.IsZero(0) {
			t.Error("IsZero(0) should be true")
		}
		if ops.IsZero(1) {
			t.Error("IsZero(1) should be false")
		}
	})
	t.Run("Exp", func(t *testing.T) {
		if ops.Exp(0) != 1.0 {
			t.Error("Exp(0) should be 1")
		}
	})
	t.Run("Log", func(t *testing.T) {
		if ops.Log(1) != 0 {
			t.Error("Log(1) should be 0")
		}
	})
	t.Run("Pow", func(t *testing.T) {
		if ops.Pow(2, 3) != 8.0 {
			t.Error("Pow(2,3) should be 8")
		}
	})
	t.Run("Sqrt", func(t *testing.T) {
		if ops.Sqrt(4) != 2.0 {
			t.Error("Sqrt(4) should be 2")
		}
	})
	t.Run("Abs", func(t *testing.T) {
		if ops.Abs(-5) != 5 {
			t.Error("Abs(-5) should be 5")
		}
		if ops.Abs(5) != 5 {
			t.Error("Abs(5) should be 5")
		}
	})
	t.Run("Sum", func(t *testing.T) {
		if ops.Sum([]float64{1, 2, 3}) != 6 {
			t.Error("Sum([1,2,3]) should be 6")
		}
	})
	t.Run("GreaterThan", func(t *testing.T) {
		if !ops.GreaterThan(5, 3) {
			t.Error("5 > 3 should be true")
		}
		if ops.GreaterThan(3, 5) {
			t.Error("3 > 5 should be false")
		}
	})
	t.Run("One", func(t *testing.T) {
		if ops.One() != 1.0 {
			t.Error("One() should be 1")
		}
	})
}

// ---------- IntOps ----------

func TestIntOps_AllMethods(t *testing.T) {
	ops := IntOps{}

	t.Run("Add", func(t *testing.T) {
		if ops.Add(1, 2) != 3 {
			t.Error("Add failed")
		}
	})
	t.Run("Sub", func(t *testing.T) {
		if ops.Sub(5, 3) != 2 {
			t.Error("Sub failed")
		}
	})
	t.Run("Mul", func(t *testing.T) {
		if ops.Mul(2, 3) != 6 {
			t.Error("Mul failed")
		}
	})
	t.Run("Div", func(t *testing.T) {
		if ops.Div(6, 2) != 3 {
			t.Error("Div failed")
		}
		if ops.Div(1, 0) != 0 {
			t.Error("Div by zero should return 0")
		}
	})
	t.Run("Tanh", func(t *testing.T) {
		if ops.Tanh(0) != 0 {
			t.Error("Tanh(0) should be 0")
		}
	})
	t.Run("Sigmoid", func(t *testing.T) {
		_ = ops.Sigmoid(0)
	})
	t.Run("ReLU", func(t *testing.T) {
		if ops.ReLU(5) != 5 {
			t.Error("ReLU(5) should be 5")
		}
		if ops.ReLU(-3) != 0 {
			t.Error("ReLU(-3) should be 0")
		}
	})
	t.Run("LeakyReLU", func(t *testing.T) {
		if ops.LeakyReLU(5, 0.1) != 5 {
			t.Error("LeakyReLU(5,0.1) should be 5")
		}
		_ = ops.LeakyReLU(-3, 0.1)
	})
	t.Run("TanhGrad", func(t *testing.T) {
		if ops.TanhGrad(0) != 1 {
			t.Errorf("TanhGrad(0) = %d, want 1", ops.TanhGrad(0))
		}
	})
	t.Run("SigmoidGrad", func(t *testing.T) {
		_ = ops.SigmoidGrad(0)
	})
	t.Run("ReLUGrad", func(t *testing.T) {
		if ops.ReLUGrad(5) != 1 {
			t.Error("ReLUGrad(5) should be 1")
		}
		if ops.ReLUGrad(-3) != 0 {
			t.Error("ReLUGrad(-3) should be 0")
		}
	})
	t.Run("LeakyReLUGrad", func(t *testing.T) {
		if ops.LeakyReLUGrad(5, 0.1) != 1 {
			t.Error("LeakyReLUGrad(5,0.1) should be 1")
		}
		_ = ops.LeakyReLUGrad(-3, 0.1)
	})
	t.Run("FromFloat32", func(t *testing.T) {
		if ops.FromFloat32(3.9) != 3 {
			t.Error("FromFloat32(3.9) should be 3")
		}
	})
	t.Run("FromFloat64", func(t *testing.T) {
		if ops.FromFloat64(3.9) != 3 {
			t.Error("FromFloat64(3.9) should be 3")
		}
	})
	t.Run("ToFloat32", func(t *testing.T) {
		if ops.ToFloat32(5) != 5.0 {
			t.Error("ToFloat32(5) should be 5.0")
		}
	})
	t.Run("IsZero", func(t *testing.T) {
		if !ops.IsZero(0) {
			t.Error("IsZero(0) should be true")
		}
		if ops.IsZero(1) {
			t.Error("IsZero(1) should be false")
		}
	})
	t.Run("Exp", func(t *testing.T) {
		if ops.Exp(0) != 1 {
			t.Error("Exp(0) should be 1")
		}
	})
	t.Run("Log", func(t *testing.T) {
		if ops.Log(1) != 0 {
			t.Error("Log(1) should be 0")
		}
	})
	t.Run("Pow", func(t *testing.T) {
		if ops.Pow(2, 3) != 8 {
			t.Error("Pow(2,3) should be 8")
		}
	})
	t.Run("Sqrt", func(t *testing.T) {
		if ops.Sqrt(9) != 3 {
			t.Error("Sqrt(9) should be 3")
		}
	})
	t.Run("Abs", func(t *testing.T) {
		if ops.Abs(-5) != 5 {
			t.Error("Abs(-5) should be 5")
		}
		if ops.Abs(5) != 5 {
			t.Error("Abs(5) should be 5")
		}
	})
	t.Run("Sum", func(t *testing.T) {
		if ops.Sum([]int{1, 2, 3}) != 6 {
			t.Error("Sum([1,2,3]) should be 6")
		}
	})
	t.Run("GreaterThan", func(t *testing.T) {
		if !ops.GreaterThan(5, 3) {
			t.Error("5 > 3 should be true")
		}
		if ops.GreaterThan(3, 5) {
			t.Error("3 > 5 should be false")
		}
	})
	t.Run("One", func(t *testing.T) {
		if ops.One() != 1 {
			t.Error("One() should be 1")
		}
	})
}

// ---------- Quantization edge cases ----------

func TestQuantize_Clamping(t *testing.T) {
	// Test value that would go below 0 after quantization
	qc, err := NewQuantizationConfig(0.01, 0, false)
	if err != nil {
		t.Fatal(err)
	}

	got := qc.Quantize(-10.0)
	if got != 0 {
		t.Errorf("Quantize(-10.0) with small scale = %d, want 0 (clamped)", got)
	}

	// Test value that would go above 255 after quantization
	got = qc.Quantize(10.0)
	if got != 255 {
		t.Errorf("Quantize(10.0) with small scale = %d, want 255 (clamped)", got)
	}
}

func TestComputeQuantizationParams_ZeroPointClamping(t *testing.T) {
	// When minVal is very negative, zero_point would be > 255
	// minVal=-1000, maxVal=0 -> scale=(0-(-1000))/255 ~= 3.92
	// zeroPoint = round(-(-1000)/3.92) = round(255.1) = 255 (clamped to 255)
	qc, err := ComputeQuantizationParams(-1000, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if qc.ZeroPoint > 255 {
		t.Errorf("ZeroPoint = %d, should be clamped to <= 255", qc.ZeroPoint)
	}

	// When maxVal is very positive and minVal is 0, zero_point would be 0
	// minVal=0, maxVal=1000 -> scale=1000/255 ~= 3.92
	// zeroPoint = round(-0/3.92) = 0
	qc2, err := ComputeQuantizationParams(0, 1000, false)
	if err != nil {
		t.Fatal(err)
	}
	if qc2.ZeroPoint < 0 {
		t.Errorf("ZeroPoint = %d, should be clamped to >= 0", qc2.ZeroPoint)
	}
}

// ---------- Float32Ops missing branches ----------

func TestFloat32Ops_Sqrt(t *testing.T) {
	ops := Float32Ops{}
	if ops.Sqrt(4) != 2 {
		t.Error("Sqrt(4) should be 2")
	}
}

func TestFloat32Ops_FromFloat64(t *testing.T) {
	ops := Float32Ops{}
	if ops.FromFloat64(3.14) != float32(3.14) {
		t.Error("FromFloat64 failed")
	}
}

func TestFloat32Ops_One(t *testing.T) {
	ops := Float32Ops{}
	if ops.One() != 1.0 {
		t.Error("One() should be 1.0")
	}
}

func TestFloat32Ops_GreaterThan(t *testing.T) {
	ops := Float32Ops{}
	if !ops.GreaterThan(5, 3) {
		t.Error("5 > 3 should be true")
	}
	if ops.GreaterThan(3, 5) {
		t.Error("3 > 5 should be false")
	}
}

func TestFloat32Ops_ToFloat32_Identity(t *testing.T) {
	ops := Float32Ops{}
	if ops.ToFloat32(3.14) != 3.14 {
		t.Error("ToFloat32 failed")
	}
}
