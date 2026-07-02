package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestNBEATS_NewValidation(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  NBEATSConfig
		wantErr bool
	}{
		{
			name: "valid trend+seasonality",
			config: NBEATSConfig{
				InputLength:     24,
				OutputLength:    12,
				StackTypes:      []StackType{StackTrend, StackSeasonality},
				NBlocksPerStack: 2,
				HiddenDim:       32,
				NHarmonics:      4,
			},
		},
		{
			name: "valid generic only",
			config: NBEATSConfig{
				InputLength:     10,
				OutputLength:    5,
				StackTypes:      []StackType{StackGeneric},
				NBlocksPerStack: 1,
				HiddenDim:       16,
				NHarmonics:      1,
			},
		},
		{
			name: "zero input length",
			config: NBEATSConfig{
				InputLength:     0,
				OutputLength:    5,
				StackTypes:      []StackType{StackTrend},
				NBlocksPerStack: 1,
				HiddenDim:       16,
				NHarmonics:      2,
			},
			wantErr: true,
		},
		{
			name: "zero output length",
			config: NBEATSConfig{
				InputLength:     10,
				OutputLength:    0,
				StackTypes:      []StackType{StackTrend},
				NBlocksPerStack: 1,
				HiddenDim:       16,
				NHarmonics:      2,
			},
			wantErr: true,
		},
		{
			name: "empty stack types",
			config: NBEATSConfig{
				InputLength:     10,
				OutputLength:    5,
				StackTypes:      []StackType{},
				NBlocksPerStack: 1,
				HiddenDim:       16,
				NHarmonics:      2,
			},
			wantErr: true,
		},
		{
			name: "zero blocks per stack",
			config: NBEATSConfig{
				InputLength:     10,
				OutputLength:    5,
				StackTypes:      []StackType{StackTrend},
				NBlocksPerStack: 0,
				HiddenDim:       16,
				NHarmonics:      2,
			},
			wantErr: true,
		},
		{
			name: "zero hidden dim",
			config: NBEATSConfig{
				InputLength:     10,
				OutputLength:    5,
				StackTypes:      []StackType{StackTrend},
				NBlocksPerStack: 1,
				HiddenDim:       0,
				NHarmonics:      2,
			},
			wantErr: true,
		},
		{
			name: "zero harmonics",
			config: NBEATSConfig{
				InputLength:     10,
				OutputLength:    5,
				StackTypes:      []StackType{StackSeasonality},
				NBlocksPerStack: 1,
				HiddenDim:       16,
				NHarmonics:      0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewNBEATS(tt.config, engine, ops)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if m == nil {
				t.Fatal("expected non-nil model")
			}
			if len(m.stacks) != len(tt.config.StackTypes) {
				t.Errorf("expected %d stacks, got %d", len(tt.config.StackTypes), len(m.stacks))
			}
		})
	}
}

func TestNBEATS_Forward(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name       string
		config     NBEATSConfig
		batch      int
		wantErr    bool
		errOnInput bool
	}{
		{
			name: "trend stack single batch",
			config: NBEATSConfig{
				InputLength:     12,
				OutputLength:    6,
				StackTypes:      []StackType{StackTrend},
				NBlocksPerStack: 2,
				HiddenDim:       16,
				NHarmonics:      2,
			},
			batch: 1,
		},
		{
			name: "seasonality stack batch 4",
			config: NBEATSConfig{
				InputLength:     24,
				OutputLength:    12,
				StackTypes:      []StackType{StackSeasonality},
				NBlocksPerStack: 2,
				HiddenDim:       16,
				NHarmonics:      4,
			},
			batch: 4,
		},
		{
			name: "generic stack",
			config: NBEATSConfig{
				InputLength:     10,
				OutputLength:    5,
				StackTypes:      []StackType{StackGeneric},
				NBlocksPerStack: 1,
				HiddenDim:       16,
				NHarmonics:      1,
			},
			batch: 2,
		},
		{
			name: "trend+seasonality+generic",
			config: NBEATSConfig{
				InputLength:     20,
				OutputLength:    10,
				StackTypes:      []StackType{StackTrend, StackSeasonality, StackGeneric},
				NBlocksPerStack: 2,
				HiddenDim:       32,
				NHarmonics:      3,
			},
			batch: 3,
		},
		{
			name: "wrong input shape",
			config: NBEATSConfig{
				InputLength:     10,
				OutputLength:    5,
				StackTypes:      []StackType{StackTrend},
				NBlocksPerStack: 1,
				HiddenDim:       16,
				NHarmonics:      2,
			},
			batch:      1,
			wantErr:    true,
			errOnInput: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewNBEATS(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewNBEATS: %v", err)
			}

			inputLen := tt.config.InputLength
			if tt.errOnInput {
				inputLen = tt.config.InputLength + 5 // wrong length
			}
			data := make([]float32, tt.batch*inputLen)
			for i := range data {
				data[i] = float32(i) * 0.01
			}
			x, err := tensor.New[float32]([]int{tt.batch, inputLen}, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			out, err := m.Forward(ctx, x)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Check forecast shape.
			fShape := out.Forecast.Shape()
			if len(fShape) != 2 || fShape[0] != tt.batch || fShape[1] != tt.config.OutputLength {
				t.Errorf("forecast shape = %v, want [%d, %d]", fShape, tt.batch, tt.config.OutputLength)
			}

			// Check stack forecasts count.
			if len(out.StackForecasts) != len(tt.config.StackTypes) {
				t.Errorf("stack forecasts count = %d, want %d", len(out.StackForecasts), len(tt.config.StackTypes))
			}

			// Forecast should contain finite values.
			fData := out.Forecast.Data()
			for i, v := range fData {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("forecast[%d] = %v, want finite", i, v)
					break
				}
			}
		})
	}
}

func TestNBEATS_ForwardConsistency(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := NBEATSConfig{
		InputLength:     12,
		OutputLength:    6,
		StackTypes:      []StackType{StackTrend, StackSeasonality},
		NBlocksPerStack: 2,
		HiddenDim:       16,
		NHarmonics:      3,
	}

	m, err := NewNBEATS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNBEATS: %v", err)
	}

	data := make([]float32, 2*12)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	x, err := tensor.New[float32]([]int{2, 12}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	out1, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward 1: %v", err)
	}

	// Same input must produce same output (deterministic inference).
	for i := 0; i < 5; i++ {
		out2, err := m.Forward(ctx, x)
		if err != nil {
			t.Fatalf("Forward %d: %v", i+2, err)
		}
		d1 := out1.Forecast.Data()
		d2 := out2.Forecast.Data()
		for j := range d1 {
			if d1[j] != d2[j] {
				t.Errorf("iteration %d: forecast[%d] = %v, want %v", i+2, j, d2[j], d1[j])
				break
			}
		}
	}
}

func TestNBEATS_BasisExpansion(t *testing.T) {
	t.Run("polynomial basis", func(t *testing.T) {
		degree := 3
		length := 5
		basis, err := polynomialBasis(degree, length)
		if err != nil {
			t.Fatalf("polynomialBasis: %v", err)
		}

		shape := basis.Shape()
		if shape[0] != degree || shape[1] != length {
			t.Fatalf("shape = %v, want [%d, %d]", shape, degree, length)
		}

		data := basis.Data()
		T := float64(length - 1)

		// Row 0 (degree 0): all ones (t/T)^0 = 1.
		for j := 0; j < length; j++ {
			got := data[j]
			if math.Abs(float64(got)-1.0) > 1e-6 {
				t.Errorf("basis[0][%d] = %v, want 1.0", j, got)
			}
		}

		// Row 1 (degree 1): t/T = [0, 0.25, 0.5, 0.75, 1.0].
		for j := 0; j < length; j++ {
			want := float32(float64(j) / T)
			got := data[length+j]
			if math.Abs(float64(got-want)) > 1e-6 {
				t.Errorf("basis[1][%d] = %v, want %v", j, got, want)
			}
		}

		// Row 2 (degree 2): (t/T)^2.
		for j := 0; j < length; j++ {
			v := float64(j) / T
			want := float32(v * v)
			got := data[2*length+j]
			if math.Abs(float64(got-want)) > 1e-6 {
				t.Errorf("basis[2][%d] = %v, want %v", j, got, want)
			}
		}
	})

	t.Run("fourier basis", func(t *testing.T) {
		nHarmonics := 2
		length := 8
		basis, err := fourierBasis(nHarmonics, length)
		if err != nil {
			t.Fatalf("fourierBasis: %v", err)
		}

		shape := basis.Shape()
		if shape[0] != 2*nHarmonics || shape[1] != length {
			t.Fatalf("shape = %v, want [%d, %d]", shape, 2*nHarmonics, length)
		}

		data := basis.Data()
		T := float64(length)

		// Check first cos row (k=1): cos(2*pi*1*t/T).
		for j := 0; j < length; j++ {
			want := float32(math.Cos(2 * math.Pi * float64(j) / T))
			got := data[j]
			if math.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("cos basis[0][%d] = %v, want %v", j, got, want)
			}
		}

		// Check first sin row (k=1): sin(2*pi*1*t/T).
		for j := 0; j < length; j++ {
			want := float32(math.Sin(2 * math.Pi * float64(j) / T))
			got := data[nHarmonics*length+j]
			if math.Abs(float64(got-want)) > 1e-5 {
				t.Errorf("sin basis[0][%d] = %v, want %v", j, got, want)
			}
		}

		// Cos at t=0 should be 1.0 for all harmonics.
		for k := 0; k < nHarmonics; k++ {
			got := data[k*length]
			if math.Abs(float64(got)-1.0) > 1e-5 {
				t.Errorf("cos basis[%d][0] = %v, want 1.0", k, got)
			}
		}

		// Sin at t=0 should be 0.0 for all harmonics.
		for k := 0; k < nHarmonics; k++ {
			got := data[(nHarmonics+k)*length]
			if math.Abs(float64(got)) > 1e-5 {
				t.Errorf("sin basis[%d][0] = %v, want 0.0", k, got)
			}
		}
	})
}

func TestNBEATS_Decomposition(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := NBEATSConfig{
		InputLength:     24,
		OutputLength:    12,
		StackTypes:      []StackType{StackTrend, StackSeasonality},
		NBlocksPerStack: 2,
		HiddenDim:       32,
		NHarmonics:      4,
	}

	m, err := NewNBEATS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNBEATS: %v", err)
	}

	batch := 2
	data := make([]float32, batch*config.InputLength)
	for i := range data {
		data[i] = float32(i) * 0.05
	}
	x, err := tensor.New[float32]([]int{batch, config.InputLength}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	decomp, err := m.Decompose(ctx, x)
	if err != nil {
		t.Fatalf("Decompose: %v", err)
	}

	// Should have entries for both trend and seasonality.
	if _, ok := decomp[StackTrend]; !ok {
		t.Error("missing trend decomposition")
	}
	if _, ok := decomp[StackSeasonality]; !ok {
		t.Error("missing seasonality decomposition")
	}

	// Each component should have shape [batch, outputLen].
	for st, comp := range decomp {
		shape := comp.Shape()
		if len(shape) != 2 || shape[0] != batch || shape[1] != config.OutputLength {
			t.Errorf("%v component shape = %v, want [%d, %d]", st, shape, batch, config.OutputLength)
		}
	}

	// Sum of decomposed components should equal the full forecast.
	out, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	trendData := decomp[StackTrend].Data()
	seasonData := decomp[StackSeasonality].Data()
	forecastData := out.Forecast.Data()

	for i := range forecastData {
		sum := trendData[i] + seasonData[i]
		diff := math.Abs(float64(sum - forecastData[i]))
		if diff > 1e-4 {
			t.Errorf("decomposition sum[%d] = %v, forecast = %v, diff = %v", i, sum, forecastData[i], diff)
			break
		}
	}
}

func TestNBEATS_ForwardBatched_Parity(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := NBEATSConfig{
		InputLength:     12,
		OutputLength:    6,
		StackTypes:      []StackType{StackTrend, StackSeasonality, StackGeneric},
		NBlocksPerStack: 2,
		HiddenDim:       16,
		NHarmonics:      3,
	}

	m, err := NewNBEATS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNBEATS: %v", err)
	}

	batch := 5
	channels := 3
	inputLen := config.InputLength
	outputLen := config.OutputLength

	// Build 3D input: [batch, channels, inputLen].
	inData := make([]float32, batch*channels*inputLen)
	for b := 0; b < batch; b++ {
		for c := 0; c < channels; c++ {
			for i := 0; i < inputLen; i++ {
				inData[(b*channels+c)*inputLen+i] = float32(math.Sin(float64(b*100+c*10+i)*0.3)) + float32(c)*0.5
			}
		}
	}
	x3d, err := tensor.New[float32]([]int{batch, channels, inputLen}, inData)
	if err != nil {
		t.Fatalf("tensor.New 3D: %v", err)
	}

	// Batched forward.
	batchOut, err := m.ForwardBatched(ctx, x3d)
	if err != nil {
		t.Fatalf("ForwardBatched: %v", err)
	}

	// Reference: manually average channels, then call Forward sample-by-sample.
	for b := 0; b < batch; b++ {
		avgData := make([]float32, inputLen)
		for i := 0; i < inputLen; i++ {
			var sum float32
			for c := 0; c < channels; c++ {
				sum += inData[(b*channels+c)*inputLen+i]
			}
			avgData[i] = sum / float32(channels)
		}
		singleIn, err := tensor.New[float32]([]int{1, inputLen}, avgData)
		if err != nil {
			t.Fatalf("tensor.New single: %v", err)
		}
		singleOut, err := m.Forward(ctx, singleIn)
		if err != nil {
			t.Fatalf("Forward sample %d: %v", b, err)
		}

		batchData := batchOut.Forecast.Data()
		singleData := singleOut.Forecast.Data()
		for j := 0; j < outputLen; j++ {
			diff := math.Abs(float64(batchData[b*outputLen+j] - singleData[j]))
			if diff > 1e-5 {
				t.Errorf("sample %d output[%d]: batched=%.8f single=%.8f diff=%.4e",
					b, j, batchData[b*outputLen+j], singleData[j], diff)
			}
		}
	}
}

func TestNBEATS_ForwardBatched_SingleChannel(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := NBEATSConfig{
		InputLength:     10,
		OutputLength:    5,
		StackTypes:      []StackType{StackGeneric},
		NBlocksPerStack: 1,
		HiddenDim:       16,
		NHarmonics:      1,
	}

	m, err := NewNBEATS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNBEATS: %v", err)
	}

	batch := 3
	inputLen := config.InputLength
	outputLen := config.OutputLength

	// Build 3D input with 1 channel.
	inData := make([]float32, batch*1*inputLen)
	for i := range inData {
		inData[i] = float32(i) * 0.1
	}
	x3d, err := tensor.New[float32]([]int{batch, 1, inputLen}, inData)
	if err != nil {
		t.Fatalf("tensor.New 3D: %v", err)
	}

	// 2D input (same data).
	x2d, err := tensor.New[float32]([]int{batch, inputLen}, inData)
	if err != nil {
		t.Fatalf("tensor.New 2D: %v", err)
	}

	batchOut, err := m.ForwardBatched(ctx, x3d)
	if err != nil {
		t.Fatalf("ForwardBatched: %v", err)
	}
	directOut, err := m.Forward(ctx, x2d)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	batchData := batchOut.Forecast.Data()
	directData := directOut.Forecast.Data()
	for i := 0; i < batch*outputLen; i++ {
		diff := math.Abs(float64(batchData[i] - directData[i]))
		if diff > 1e-5 {
			t.Errorf("output[%d]: batched=%.8f direct=%.8f diff=%.4e",
				i, batchData[i], directData[i], diff)
		}
	}
}

func TestNBEATS_ForwardBatched_InvalidShape(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := NBEATSConfig{
		InputLength:     10,
		OutputLength:    5,
		StackTypes:      []StackType{StackTrend},
		NBlocksPerStack: 1,
		HiddenDim:       16,
		NHarmonics:      2,
	}

	m, err := NewNBEATS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNBEATS: %v", err)
	}

	// 2D input should fail.
	x2d, err := tensor.New[float32]([]int{2, 10}, make([]float32, 20))
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	_, err = m.ForwardBatched(ctx, x2d)
	if err == nil {
		t.Fatal("expected error for 2D input to ForwardBatched")
	}

	// Wrong inputLen should fail.
	x3d, err := tensor.New[float32]([]int{2, 1, 8}, make([]float32, 16))
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	_, err = m.ForwardBatched(ctx, x3d)
	if err == nil {
		t.Fatal("expected error for wrong inputLen")
	}
}

func TestStackType_String(t *testing.T) {
	tests := []struct {
		st   StackType
		want string
	}{
		{StackTrend, "trend"},
		{StackSeasonality, "seasonality"},
		{StackGeneric, "generic"},
		{StackType(99), "StackType(99)"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.st.String(); got != tt.want {
				t.Errorf("StackType.String() = %q, want %q", got, tt.want)
			}
		})
	}
}
