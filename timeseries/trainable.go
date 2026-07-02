package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// NBEATSAdapter wraps an NBEATS model to satisfy training.Model[float32].
// The Forward method extracts the Forecast tensor from NBEATSOutput,
// discarding per-stack decomposition. Use NBEATS.Forward directly when
// interpretability (stack-level forecasts/backcasts) is needed.
type NBEATSAdapter struct {
	Model  *NBEATS
	params []*graph.Parameter[float32]
}

// NewNBEATSAdapter creates a trainable adapter for the given NBEATS model.
func NewNBEATSAdapter(m *NBEATS) (*NBEATSAdapter, error) {
	params, err := collectNBEATSParameters(m)
	if err != nil {
		return nil, fmt.Errorf("nbeats adapter: %w", err)
	}
	return &NBEATSAdapter{Model: m, params: params}, nil
}

// Forward runs the NBEATS forward pass and returns the forecast tensor.
// Expects a single input tensor of shape [batch, inputLen].
func (a *NBEATSAdapter) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("nbeats adapter: expected 1 input, got %d", len(inputs))
	}
	out, err := a.Model.Forward(ctx, inputs[0])
	if err != nil {
		return nil, err
	}
	return out.Forecast, nil
}

// Backward is not implemented for timeseries models. Timeseries models
// use standalone training loops with numerical or engine-level gradient
// computation rather than manual backward passes.
func (a *NBEATSAdapter) Backward(_ context.Context, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, fmt.Errorf("nbeats adapter: Backward not implemented; use engine-level autograd or standalone training loops")
}

// Parameters returns all trainable parameters from the NBEATS model.
func (a *NBEATSAdapter) Parameters() []*graph.Parameter[float32] {
	return a.params
}

// PatchTSTAdapter wraps a PatchTST model to satisfy training.Model[float32].
// PatchTST.Forward already returns a tensor, so the adapter is a thin pass-through.
type PatchTSTAdapter struct {
	Model  *PatchTST
	params []*graph.Parameter[float32]
}

// NewPatchTSTAdapter creates a trainable adapter for the given PatchTST model.
func NewPatchTSTAdapter(m *PatchTST) (*PatchTSTAdapter, error) {
	params, err := collectPatchTSTParameters(m)
	if err != nil {
		return nil, fmt.Errorf("patchtst adapter: %w", err)
	}
	return &PatchTSTAdapter{Model: m, params: params}, nil
}

// Forward runs the PatchTST forward pass.
// Expects a single input tensor of shape [batch, input_length] or [batch, channels, input_length].
func (a *PatchTSTAdapter) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("patchtst adapter: expected 1 input, got %d", len(inputs))
	}
	return a.Model.Forward(ctx, inputs[0])
}

// Backward is not implemented for timeseries models.
func (a *PatchTSTAdapter) Backward(_ context.Context, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, fmt.Errorf("patchtst adapter: Backward not implemented; use engine-level autograd or standalone training loops")
}

// Parameters returns all trainable parameters from the PatchTST model.
func (a *PatchTSTAdapter) Parameters() []*graph.Parameter[float32] {
	return a.params
}

// NHiTSAdapter wraps an NHiTS model to satisfy training.Model[float32].
// The Forward method runs the N-HiTS forward pass, producing a forecast tensor.
type NHiTSAdapter struct {
	Model  *NHiTS
	params []*graph.Parameter[float32]
}

// NewNHiTSAdapter creates a trainable adapter for the given NHiTS model.
func NewNHiTSAdapter(m *NHiTS) (*NHiTSAdapter, error) {
	params, err := collectNHiTSParameters(m)
	if err != nil {
		return nil, fmt.Errorf("nhits adapter: %w", err)
	}
	return &NHiTSAdapter{Model: m, params: params}, nil
}

// Forward runs the NHiTS forward pass and returns the forecast tensor.
// Expects a single input tensor of shape [batch, channels * inputLen].
func (a *NHiTSAdapter) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("nhits adapter: expected 1 input, got %d", len(inputs))
	}
	return a.Model.Forward(ctx, inputs[0])
}

// Backward is not implemented for timeseries models.
func (a *NHiTSAdapter) Backward(_ context.Context, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, fmt.Errorf("nhits adapter: Backward not implemented; use engine-level autograd or standalone training loops")
}

// Parameters returns all trainable parameters from the NHiTS model.
func (a *NHiTSAdapter) Parameters() []*graph.Parameter[float32] {
	return a.params
}

// collectNHiTSParameters collects all trainable parameters from an NHiTS model.
func collectNHiTSParameters(m *NHiTS) ([]*graph.Parameter[float32], error) {
	var params []*graph.Parameter[float32]

	for si, stack := range m.stacks {
		prefix := fmt.Sprintf("stack.%d", si)

		for li, l := range stack.mlpLayers {
			p, err := collectMLPParams(fmt.Sprintf("%s.mlp.%d", prefix, li), l)
			if err != nil {
				return nil, err
			}
			params = append(params, p...)
		}

		p, err := collectMLPParams(prefix+".output_proj", stack.outputProj)
		if err != nil {
			return nil, err
		}
		params = append(params, p...)
	}

	return params, nil
}

// TFTAdapter wraps a TFT model to satisfy training.Model[float32].
// The Forward method expects two inputs: static features [batch, numStaticFeatures]
// and time features [batch, seqLen, numTimeFeatures]. It processes each batch
// element individually through TFT.Predict and returns the stacked output.
type TFTAdapter struct {
	Model  *TFT
	params []*graph.Parameter[float32]
}

// NewTFTAdapter creates a trainable adapter for the given TFT model.
func NewTFTAdapter(m *TFT) (*TFTAdapter, error) {
	params, err := collectTFTParameters(m)
	if err != nil {
		return nil, fmt.Errorf("tft adapter: %w", err)
	}
	return &TFTAdapter{Model: m, params: params}, nil
}

// Forward runs the TFT forward pass.
// Expects two inputs: static features [batch, numStaticFeatures] and
// time features [batch, seqLen, numTimeFeatures].
// Returns [batch, nHorizons * nQuantiles].
func (a *TFTAdapter) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("tft adapter: expected 2 inputs (static, time), got %d", len(inputs))
	}

	staticT := inputs[0]
	timeT := inputs[1]

	staticShape := staticT.Shape()
	timeShape := timeT.Shape()

	if len(staticShape) != 2 {
		return nil, fmt.Errorf("tft adapter: static input must be 2D [batch, features], got shape %v", staticShape)
	}
	if len(timeShape) != 3 {
		return nil, fmt.Errorf("tft adapter: time input must be 3D [batch, seqLen, features], got shape %v", timeShape)
	}

	batch := staticShape[0]
	numStatic := staticShape[1]
	seqLen := timeShape[1]
	numTime := timeShape[2]

	if timeShape[0] != batch {
		return nil, fmt.Errorf("tft adapter: batch size mismatch: static=%d, time=%d", batch, timeShape[0])
	}

	nq := len(a.Model.config.Quantiles)
	outDim := a.Model.config.NHorizons * nq
	outData := make([]float32, batch*outDim)

	staticData := staticT.Data()
	timeData := timeT.Data()

	for b := 0; b < batch; b++ {
		// Extract static features for this batch element.
		sf := make([]float64, numStatic)
		for i := 0; i < numStatic; i++ {
			sf[i] = float64(staticData[b*numStatic+i])
		}

		// Extract time features for this batch element.
		tf := make([][]float64, seqLen)
		for t := 0; t < seqLen; t++ {
			tf[t] = make([]float64, numTime)
			for i := 0; i < numTime; i++ {
				tf[t][i] = float64(timeData[(b*seqLen+t)*numTime+i])
			}
		}

		pred, err := a.Model.Predict(sf, tf)
		if err != nil {
			return nil, fmt.Errorf("tft adapter: batch %d: %w", b, err)
		}

		// Flatten [nHorizons][nQuantiles] into output.
		for h := 0; h < a.Model.config.NHorizons; h++ {
			for q := 0; q < nq; q++ {
				outData[b*outDim+h*nq+q] = float32(pred[h][q])
			}
		}
	}

	return tensor.New[float32]([]int{batch, outDim}, outData)
}

// Backward is not implemented for timeseries models.
func (a *TFTAdapter) Backward(_ context.Context, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, fmt.Errorf("tft adapter: Backward not implemented; use engine-level autograd or standalone training loops")
}

// Parameters returns all trainable parameters from the TFT model.
func (a *TFTAdapter) Parameters() []*graph.Parameter[float32] {
	return a.params
}

// TimeMixerAdapter wraps a TimeMixer model to satisfy training.Model[float32].
// The Forward method takes a flat [batch, channels * inputLen] tensor,
// decomposes each sample at multiple scales, averages the trend components,
// and returns the last outputLen timesteps as [batch, channels * outputLen].
type TimeMixerAdapter struct {
	Model  *TimeMixer
	params []*graph.Parameter[float32]
}

// NewTimeMixerAdapter creates a trainable adapter for the given TimeMixer model.
func NewTimeMixerAdapter(m *TimeMixer) (*TimeMixerAdapter, error) {
	params, err := collectTimeMixerParameters(m)
	if err != nil {
		return nil, fmt.Errorf("timemixer adapter: %w", err)
	}
	return &TimeMixerAdapter{Model: m, params: params}, nil
}

// Forward runs the TimeMixer forward pass.
// Expects a single input tensor of shape [batch, channels * inputLen].
// Returns [batch, channels * outputLen].
func (a *TimeMixerAdapter) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("timemixer adapter: expected 1 input, got %d", len(inputs))
	}

	shape := inputs[0].Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("timemixer adapter: expected 2D input [batch, channels*inputLen], got shape %v", shape)
	}

	batch := shape[0]
	cfg := a.Model.config
	flatIn := cfg.NumFeatures * cfg.InputLen
	if shape[1] != flatIn {
		return nil, fmt.Errorf("timemixer adapter: input dim %d != channels(%d) * inputLen(%d)", shape[1], cfg.NumFeatures, cfg.InputLen)
	}

	outLen := cfg.OutputLen
	if outLen <= 0 {
		outLen = cfg.InputLen
	}
	flatOut := cfg.NumFeatures * outLen
	outData := make([]float32, batch*flatOut)
	data := inputs[0].Data()

	for b := 0; b < batch; b++ {
		// Reshape flat input into [numFeatures][inputLen].
		features := make([][]float64, cfg.NumFeatures)
		for f := 0; f < cfg.NumFeatures; f++ {
			features[f] = make([]float64, cfg.InputLen)
			for i := 0; i < cfg.InputLen; i++ {
				features[f][i] = float64(data[b*flatIn+f*cfg.InputLen+i])
			}
		}

		msOut, err := a.Model.Forward(features)
		if err != nil {
			return nil, fmt.Errorf("timemixer adapter: batch %d: %w", b, err)
		}

		// Average trend across scales and take last outputLen timesteps.
		nScales := float64(len(msOut.Scales))
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < outLen; i++ {
				srcIdx := cfg.InputLen - outLen + i
				if srcIdx < 0 {
					srcIdx = 0
				}
				avg := 0.0
				for _, sc := range msOut.Scales {
					avg += sc.trend[f][srcIdx]
				}
				outData[b*flatOut+f*outLen+i] = float32(avg / nScales)
			}
		}
	}

	return tensor.New[float32]([]int{batch, flatOut}, outData)
}

// Backward is not implemented for timeseries models.
func (a *TimeMixerAdapter) Backward(_ context.Context, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, fmt.Errorf("timemixer adapter: Backward not implemented; use engine-level autograd or standalone training loops")
}

// Parameters returns all trainable parameters from the TimeMixer model.
func (a *TimeMixerAdapter) Parameters() []*graph.Parameter[float32] {
	return a.params
}

// collectTimeMixerParameters collects all trainable parameters from a TimeMixer model.
func collectTimeMixerParameters(m *TimeMixer) ([]*graph.Parameter[float32], error) {
	var params []*graph.Parameter[float32]

	for s := 0; s < len(m.maWeights); s++ {
		data := make([]float32, len(m.maWeights[s]))
		for i, v := range m.maWeights[s] {
			data[i] = float32(v)
		}
		t, err := tensor.New[float32]([]int{len(data)}, data)
		if err != nil {
			return nil, err
		}
		p, err := graph.NewParameter(fmt.Sprintf("ma_weights.%d", s), t, tensor.New[float32])
		if err != nil {
			return nil, err
		}
		params = append(params, p)
	}

	return params, nil
}

// collectLinearParams collects weight and bias parameters from a linearLayer.
func collectLinearParams(prefix string, l linearLayer) ([]*graph.Parameter[float32], error) {
	w, err := graph.NewParameter(prefix+".weight", l.weights, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	b, err := graph.NewParameter(prefix+".bias", l.biases, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	return []*graph.Parameter[float32]{w, b}, nil
}

// collectMLPParams collects weight and bias parameters from an mlpLayer.
func collectMLPParams(prefix string, l mlpLayer) ([]*graph.Parameter[float32], error) {
	w, err := graph.NewParameter(prefix+".weight", l.weights, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	b, err := graph.NewParameter(prefix+".bias", l.biases, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	return []*graph.Parameter[float32]{w, b}, nil
}

// collectNBEATSParameters collects all trainable parameters from an NBEATS model.
func collectNBEATSParameters(m *NBEATS) ([]*graph.Parameter[float32], error) {
	var params []*graph.Parameter[float32]

	for si, stack := range m.stacks {
		for bi, block := range stack.blocks {
			prefix := fmt.Sprintf("stack.%d.block.%d", si, bi)

			for fi, fc := range block.fcLayers {
				p, err := collectMLPParams(fmt.Sprintf("%s.fc.%d", prefix, fi), fc)
				if err != nil {
					return nil, err
				}
				params = append(params, p...)
			}

			p, err := collectMLPParams(prefix+".theta_b", block.thetaBLayer)
			if err != nil {
				return nil, err
			}
			params = append(params, p...)

			p, err = collectMLPParams(prefix+".theta_f", block.thetaFLayer)
			if err != nil {
				return nil, err
			}
			params = append(params, p...)
		}
	}

	return params, nil
}

// collectPatchTSTParameters collects all trainable parameters from a PatchTST model.
func collectPatchTSTParameters(m *PatchTST) ([]*graph.Parameter[float32], error) {
	var params []*graph.Parameter[float32]

	// Patch embedding.
	p, err := collectLinearParams("patch_emb", m.patchEmb)
	if err != nil {
		return nil, err
	}
	params = append(params, p...)

	// Positional embedding.
	posParam, err := graph.NewParameter("pos_emb", m.posEmb, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	params = append(params, posParam)

	// Encoder layers.
	for i, layer := range m.layers {
		prefix := fmt.Sprintf("layer.%d", i)

		for _, pair := range []struct {
			name string
			l    linearLayer
		}{
			{"q_proj", layer.qProj},
			{"k_proj", layer.kProj},
			{"v_proj", layer.vProj},
			{"o_proj", layer.oProj},
			{"ffn1", layer.ffn1},
			{"ffn2", layer.ffn2},
		} {
			p, err := collectLinearParams(prefix+"."+pair.name, pair.l)
			if err != nil {
				return nil, err
			}
			params = append(params, p...)
		}

		// Layer norm parameters.
		for _, pair := range []struct {
			name string
			t    *tensor.TensorNumeric[float32]
		}{
			{prefix + ".norm1.scale", layer.norm1},
			{prefix + ".norm1.bias", layer.bias1},
			{prefix + ".norm2.scale", layer.norm2},
			{prefix + ".norm2.bias", layer.bias2},
		} {
			param, err := graph.NewParameter(pair.name, pair.t, tensor.New[float32])
			if err != nil {
				return nil, err
			}
			params = append(params, param)
		}
	}

	// Output head.
	p, err = collectLinearParams("head", m.head)
	if err != nil {
		return nil, err
	}
	params = append(params, p...)

	return params, nil
}

// collectGRNParams collects parameters from a grnWeights struct.
func collectGRNParams(prefix string, grn grnWeights) ([]*graph.Parameter[float32], error) {
	var params []*graph.Parameter[float32]

	for _, pair := range []struct {
		name string
		l    linearLayer
	}{
		{"fc1", grn.fc1},
		{"fc2", grn.fc2},
		{"gate", grn.gate},
		{"proj", grn.proj},
	} {
		p, err := collectLinearParams(prefix+"."+pair.name, pair.l)
		if err != nil {
			return nil, err
		}
		params = append(params, p...)
	}

	gain, err := graph.NewParameter(prefix+".ln.gain", grn.lnGain, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	params = append(params, gain)

	bias, err := graph.NewParameter(prefix+".ln.bias", grn.lnBias, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	params = append(params, bias)

	return params, nil
}

// collectTFTParameters collects all trainable parameters from a TFT model.
func collectTFTParameters(m *TFT) ([]*graph.Parameter[float32], error) {
	var params []*graph.Parameter[float32]

	// Static VSN weights.
	for i, l := range m.staticVSNWeights {
		p, err := collectLinearParams(fmt.Sprintf("static_vsn.%d", i), l)
		if err != nil {
			return nil, err
		}
		params = append(params, p...)
	}
	p, err := collectLinearParams("static_vsn.select", m.staticVSNSelect)
	if err != nil {
		return nil, err
	}
	params = append(params, p...)

	// Time VSN weights.
	for i, l := range m.timeVSNWeights {
		p, err := collectLinearParams(fmt.Sprintf("time_vsn.%d", i), l)
		if err != nil {
			return nil, err
		}
		params = append(params, p...)
	}
	p, err = collectLinearParams("time_vsn.select", m.timeVSNSelect)
	if err != nil {
		return nil, err
	}
	params = append(params, p...)

	// Static encoder GRN.
	p, err = collectGRNParams("static_encoder", m.staticEncoder)
	if err != nil {
		return nil, err
	}
	params = append(params, p...)

	// Temporal GRN.
	p, err = collectGRNParams("temporal_grn", m.temporalGRN)
	if err != nil {
		return nil, err
	}
	params = append(params, p...)

	// Attention projections.
	for _, pair := range []struct {
		name string
		l    linearLayer
	}{
		{"attn.q", m.attnQ},
		{"attn.k", m.attnK},
		{"attn.v", m.attnV},
		{"attn.o", m.attnO},
	} {
		p, err := collectLinearParams(pair.name, pair.l)
		if err != nil {
			return nil, err
		}
		params = append(params, p...)
	}

	// Post-attention GRN.
	p, err = collectGRNParams("post_attn_grn", m.postAttnGRN)
	if err != nil {
		return nil, err
	}
	params = append(params, p...)

	// Output projection.
	p, err = collectLinearParams("output_proj", m.outputProj)
	if err != nil {
		return nil, err
	}
	params = append(params, p...)

	return params, nil
}
