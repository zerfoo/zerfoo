package fp8

import (
	"math"
	"testing"
)

func TestLossScaler(t *testing.T) {
	t.Run("normal_step", func(t *testing.T) {
		ls := NewLossScaler(1024)

		scaled := ls.ScaleLoss(0.5)
		if scaled != 512 {
			t.Errorf("ScaleLoss(0.5) = %f, want 512", scaled)
		}

		grads := [][]float32{{0.1, 0.2, 0.3}}
		ok := ls.CheckGradients(grads)
		if !ok {
			t.Error("CheckGradients returned false for finite gradients")
		}
		if ls.Scale != 1024 {
			t.Errorf("scale = %f after clean step, want 1024", ls.Scale)
		}
	})

	t.Run("inf_gradient_halves_scale", func(t *testing.T) {
		ls := NewLossScaler(1024)

		grads := [][]float32{{0.1, float32(math.Inf(1)), 0.3}}
		ok := ls.CheckGradients(grads)
		if ok {
			t.Error("CheckGradients returned true for inf gradient")
		}
		if ls.Scale != 512 {
			t.Errorf("scale = %f after inf, want 512", ls.Scale)
		}
	})

	t.Run("nan_gradient_halves_scale", func(t *testing.T) {
		ls := NewLossScaler(1024)

		grads := [][]float32{{float32(math.NaN()), 0.2}}
		ok := ls.CheckGradients(grads)
		if ok {
			t.Error("CheckGradients returned true for NaN gradient")
		}
		if ls.Scale != 512 {
			t.Errorf("scale = %f after NaN, want 512", ls.Scale)
		}
	})

	t.Run("2000_clean_steps_doubles_scale", func(t *testing.T) {
		ls := NewLossScaler(1024)

		for i := 0; i < 2000; i++ {
			ls.Update(false)
		}
		if ls.Scale != 2048 {
			t.Errorf("scale = %f after 2000 clean steps, want 2048", ls.Scale)
		}
	})

	t.Run("overflow_resets_grow_counter", func(t *testing.T) {
		ls := NewLossScaler(1024)

		// 1999 clean steps, then overflow resets counter.
		for i := 0; i < 1999; i++ {
			ls.Update(false)
		}
		ls.Update(true)
		// One more clean step should not double (counter was reset).
		ls.Update(false)
		if ls.Scale != 1024 {
			t.Errorf("scale = %f after reset, want 1024", ls.Scale)
		}
	})

	t.Run("scale_floor", func(t *testing.T) {
		ls := NewLossScaler(2)

		// Halve repeatedly — should not go below 1.0.
		infGrads := [][]float32{{float32(math.Inf(1))}}
		ls.CheckGradients(infGrads) // 2 -> 1
		ls.CheckGradients(infGrads) // stays 1
		ls.CheckGradients(infGrads) // stays 1
		if ls.Scale != 1.0 {
			t.Errorf("scale = %f, want 1.0 (floor)", ls.Scale)
		}
	})

	t.Run("unscale_gradients", func(t *testing.T) {
		ls := NewLossScaler(4)

		grads := [][]float32{{8, 12}, {4}}
		ls.UnscaleGradients(grads)

		want := [][]float32{{2, 3}, {1}}
		for i := range want {
			for j := range want[i] {
				if grads[i][j] != want[i][j] {
					t.Errorf("grads[%d][%d] = %f, want %f", i, j, grads[i][j], want[i][j])
				}
			}
		}
	})

	t.Run("initial_scale_floor", func(t *testing.T) {
		ls := NewLossScaler(0.5)
		if ls.Scale != 1.0 {
			t.Errorf("initial scale = %f, want 1.0 (floor)", ls.Scale)
		}
	})
}
