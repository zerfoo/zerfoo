package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// trainingModel mirrors training.Model[float32] for compile-time verification.
// This avoids importing the training package (which would create a circular
// dependency risk) while still proving interface compliance.
type trainingModel interface {
	Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)
	Backward(ctx context.Context, grad *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error)
	Parameters() []*graph.Parameter[float32]
}

// Compile-time interface checks.
var (
	_ trainingModel = (*NBEATSAdapter)(nil)
	_ trainingModel = (*PatchTSTAdapter)(nil)
	_ trainingModel = (*TFTAdapter)(nil)
	_ trainingModel = (*TimeMixerAdapter)(nil)
)

func TestNBEATSAdapter(t *testing.T) {
	engine, ops := newTestEngine()

	m, err := NewNBEATS(NBEATSConfig{
		InputLength:     12,
		OutputLength:    6,
		StackTypes:      []StackType{StackTrend, StackSeasonality},
		NBlocksPerStack: 1,
		HiddenDim:       8,
		NHarmonics:      2,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewNBEATS: %v", err)
	}

	adapter, err := NewNBEATSAdapter(m)
	if err != nil {
		t.Fatalf("NewNBEATSAdapter: %v", err)
	}

	tests := []struct {
		name    string
		fn      func() error
		wantErr bool
	}{
		{
			name: "forward returns forecast tensor",
			fn: func() error {
				input, err := tensor.New[float32]([]int{2, 12}, make([]float32, 24))
				if err != nil {
					return err
				}
				out, err := adapter.Forward(context.Background(), input)
				if err != nil {
					return err
				}
				shape := out.Shape()
				if len(shape) != 2 || shape[0] != 2 || shape[1] != 6 {
					t.Errorf("expected shape [2, 6], got %v", shape)
				}
				return nil
			},
		},
		{
			name: "forward rejects wrong input count",
			fn: func() error {
				a, _ := tensor.New[float32]([]int{1, 12}, make([]float32, 12))
				b, _ := tensor.New[float32]([]int{1, 12}, make([]float32, 12))
				_, err := adapter.Forward(context.Background(), a, b)
				return err
			},
			wantErr: true,
		},
		{
			name: "parameters are non-empty",
			fn: func() error {
				params := adapter.Parameters()
				if len(params) == 0 {
					t.Error("expected non-empty parameters")
				}
				// Each block has 4 FC layers (2 params each) + theta_b (2) + theta_f (2) = 12 params.
				// 2 stacks * 1 block each = 24 params.
				if len(params) != 24 {
					t.Errorf("expected 24 parameters, got %d", len(params))
				}
				return nil
			},
		},
		{
			name: "parameter names are unique",
			fn: func() error {
				params := adapter.Parameters()
				seen := make(map[string]bool)
				for _, p := range params {
					if seen[p.Name] {
						t.Errorf("duplicate parameter name: %s", p.Name)
					}
					seen[p.Name] = true
				}
				return nil
			},
		},
		{
			name: "backward returns error",
			fn: func() error {
				_, err := adapter.Backward(context.Background(), nil)
				return err
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if (err != nil) != tt.wantErr {
				t.Errorf("got err=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}

func TestPatchTSTAdapter(t *testing.T) {
	engine, ops := newTestEngine()

	m, err := NewPatchTST(PatchTSTConfig{
		InputLength: 16,
		PatchLength: 4,
		Stride:      4,
		DModel:      8,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   4,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	adapter, err := NewPatchTSTAdapter(m)
	if err != nil {
		t.Fatalf("NewPatchTSTAdapter: %v", err)
	}

	tests := []struct {
		name    string
		fn      func() error
		wantErr bool
	}{
		{
			name: "forward returns tensor",
			fn: func() error {
				input, err := tensor.New[float32]([]int{2, 16}, make([]float32, 32))
				if err != nil {
					return err
				}
				out, err := adapter.Forward(context.Background(), input)
				if err != nil {
					return err
				}
				shape := out.Shape()
				if len(shape) != 2 || shape[0] != 2 || shape[1] != 4 {
					t.Errorf("expected shape [2, 4], got %v", shape)
				}
				return nil
			},
		},
		{
			name: "parameters are non-empty",
			fn: func() error {
				params := adapter.Parameters()
				if len(params) == 0 {
					t.Error("expected non-empty parameters")
				}
				// patch_emb (2) + pos_emb (1) + 1 layer * (q,k,v,o,ffn1,ffn2 = 12 + norm1s,norm1b,norm2s,norm2b = 4) + head (2) = 21
				if len(params) != 21 {
					t.Errorf("expected 21 parameters, got %d", len(params))
				}
				return nil
			},
		},
		{
			name: "parameter names are unique",
			fn: func() error {
				params := adapter.Parameters()
				seen := make(map[string]bool)
				for _, p := range params {
					if seen[p.Name] {
						t.Errorf("duplicate parameter name: %s", p.Name)
					}
					seen[p.Name] = true
				}
				return nil
			},
		},
		{
			name: "backward returns error",
			fn: func() error {
				_, err := adapter.Backward(context.Background(), nil)
				return err
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if (err != nil) != tt.wantErr {
				t.Errorf("got err=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}

func TestTFTAdapter(t *testing.T) {
	engine, ops := newTestEngine()

	m, err := NewTFT(TFTConfig{
		NumStaticFeatures: 2,
		NumTimeFeatures:   3,
		DModel:            8,
		NHeads:            2,
		NHorizons:         4,
		Quantiles:         []float64{0.1, 0.5, 0.9},
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewTFT: %v", err)
	}

	adapter, err := NewTFTAdapter(m)
	if err != nil {
		t.Fatalf("NewTFTAdapter: %v", err)
	}

	tests := []struct {
		name    string
		fn      func() error
		wantErr bool
	}{
		{
			name: "forward returns correct shape",
			fn: func() error {
				staticInput, err := tensor.New[float32]([]int{1, 2}, []float32{0.5, 1.0})
				if err != nil {
					return err
				}
				timeInput, err := tensor.New[float32]([]int{1, 3, 3}, make([]float32, 9))
				if err != nil {
					return err
				}
				out, err := adapter.Forward(context.Background(), staticInput, timeInput)
				if err != nil {
					return err
				}
				// Output: [batch=1, nHorizons*nQuantiles = 4*3 = 12]
				shape := out.Shape()
				if len(shape) != 2 || shape[0] != 1 || shape[1] != 12 {
					t.Errorf("expected shape [1, 12], got %v", shape)
				}
				return nil
			},
		},
		{
			name: "forward rejects wrong input count",
			fn: func() error {
				a, _ := tensor.New[float32]([]int{1, 2}, make([]float32, 2))
				_, err := adapter.Forward(context.Background(), a)
				return err
			},
			wantErr: true,
		},
		{
			name: "parameters are non-empty",
			fn: func() error {
				params := adapter.Parameters()
				if len(params) == 0 {
					t.Error("expected non-empty parameters")
				}
				return nil
			},
		},
		{
			name: "parameter names are unique",
			fn: func() error {
				params := adapter.Parameters()
				seen := make(map[string]bool)
				for _, p := range params {
					if seen[p.Name] {
						t.Errorf("duplicate parameter name: %s", p.Name)
					}
					seen[p.Name] = true
				}
				return nil
			},
		},
		{
			name: "backward returns error",
			fn: func() error {
				_, err := adapter.Backward(context.Background(), nil)
				return err
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if (err != nil) != tt.wantErr {
				t.Errorf("got err=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}

func TestTimeMixerAdapter(t *testing.T) {
	m := NewTimeMixer(TimeMixerConfig{
		InputLen:    12,
		OutputLen:   6,
		NumFeatures: 2,
		NumScales:   3,
	})

	adapter, err := NewTimeMixerAdapter(m)
	if err != nil {
		t.Fatalf("NewTimeMixerAdapter: %v", err)
	}

	tests := []struct {
		name    string
		fn      func() error
		wantErr bool
	}{
		{
			name: "forward returns correct shape",
			fn: func() error {
				// [batch=2, channels*inputLen = 2*12 = 24]
				input, err := tensor.New[float32]([]int{2, 24}, make([]float32, 48))
				if err != nil {
					return err
				}
				out, err := adapter.Forward(context.Background(), input)
				if err != nil {
					return err
				}
				shape := out.Shape()
				// [batch=2, channels*outputLen = 2*6 = 12]
				if len(shape) != 2 || shape[0] != 2 || shape[1] != 12 {
					t.Errorf("expected shape [2, 12], got %v", shape)
				}
				return nil
			},
		},
		{
			name: "forward rejects wrong input count",
			fn: func() error {
				a, _ := tensor.New[float32]([]int{1, 24}, make([]float32, 24))
				b, _ := tensor.New[float32]([]int{1, 24}, make([]float32, 24))
				_, err := adapter.Forward(context.Background(), a, b)
				return err
			},
			wantErr: true,
		},
		{
			name: "parameters match scale count",
			fn: func() error {
				params := adapter.Parameters()
				if len(params) == 0 {
					t.Error("expected non-empty parameters")
				}
				// 3 scales = 3 ma_weights parameters.
				if len(params) != 3 {
					t.Errorf("expected 3 parameters, got %d", len(params))
				}
				return nil
			},
		},
		{
			name: "parameter names are unique",
			fn: func() error {
				params := adapter.Parameters()
				seen := make(map[string]bool)
				for _, p := range params {
					if seen[p.Name] {
						t.Errorf("duplicate parameter name: %s", p.Name)
					}
					seen[p.Name] = true
				}
				return nil
			},
		},
		{
			name: "backward returns error",
			fn: func() error {
				_, err := adapter.Backward(context.Background(), nil)
				return err
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if (err != nil) != tt.wantErr {
				t.Errorf("got err=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}
