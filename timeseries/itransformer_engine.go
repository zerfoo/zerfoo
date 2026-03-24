package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// trainWindowedEngine implements GPU-accelerated ITransformer training using
// float32 tensor operations via the compute.Engine. The forward/backward
// analytical backpropagation logic mirrors the CPU path but uses float32
// working copies. AdamW updates use engine tensor ops for large parameter
// tensors.
func (m *ITransformer) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	channels := m.config.Channels
	outputLen := m.config.OutputLen

	if config.Epochs <= 0 {
		config.Epochs = 100
	}
	if config.LR <= 0 {
		config.LR = 1e-3
	}
	if config.Beta1 <= 0 {
		config.Beta1 = 0.9
	}
	if config.Beta2 <= 0 {
		config.Beta2 = 0.999
	}
	if config.Epsilon <= 0 {
		config.Epsilon = 1e-8
	}

	ctx := context.Background()

	// paramRef holds a float32 working copy and pointers back into the model's
	// float64 parameters so updates can be written back after each step.
	type paramRef struct {
		f64Ptrs []*float64
		f32     []float32
	}

	// Build flat parameter references matching flatParams order.
	var allParams []paramRef

	// Embedding weights [inputLen][dModel].
	{
		n := m.config.InputLen * m.config.DModel
		ptrs := make([]*float64, 0, n)
		f32 := make([]float32, 0, n)
		for i := range m.embedW {
			for j := range m.embedW[i] {
				ptrs = append(ptrs, &m.embedW[i][j])
				f32 = append(f32, float32(m.embedW[i][j]))
			}
		}
		allParams = append(allParams, paramRef{ptrs, f32})
	}
	// Embedding bias [dModel].
	{
		ptrs := make([]*float64, m.config.DModel)
		f32 := make([]float32, m.config.DModel)
		for i := range m.embedB {
			ptrs[i] = &m.embedB[i]
			f32[i] = float32(m.embedB[i])
		}
		allParams = append(allParams, paramRef{ptrs, f32})
	}

	// Encoder layers.
	for li := range m.layers {
		l := &m.layers[li]
		dModel := m.config.DModel
		dFF := m.config.DFF

		// Q, K, V, O weights: each [dModel][dModel].
		for _, wPtr := range []*[][]float64{&l.qW, &l.kW, &l.vW, &l.oW} {
			n := dModel * dModel
			ptrs := make([]*float64, 0, n)
			f32 := make([]float32, 0, n)
			for i := range *wPtr {
				for j := range (*wPtr)[i] {
					ptrs = append(ptrs, &(*wPtr)[i][j])
					f32 = append(f32, float32((*wPtr)[i][j]))
				}
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
		// Q, K, V, O biases: each [dModel].
		for _, bPtr := range []*[]float64{&l.qB, &l.kB, &l.vB, &l.oB} {
			ptrs := make([]*float64, dModel)
			f32 := make([]float32, dModel)
			for i := range *bPtr {
				ptrs[i] = &(*bPtr)[i]
				f32[i] = float32((*bPtr)[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}

		// LN1 scale and bias.
		for _, bPtr := range []*[]float64{&l.ln1Scale, &l.ln1Bias} {
			ptrs := make([]*float64, dModel)
			f32 := make([]float32, dModel)
			for i := range *bPtr {
				ptrs[i] = &(*bPtr)[i]
				f32[i] = float32((*bPtr)[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}

		// FFN: fc1W [dModel][dFF].
		{
			n := dModel * dFF
			ptrs := make([]*float64, 0, n)
			f32 := make([]float32, 0, n)
			for i := range l.fc1W {
				for j := range l.fc1W[i] {
					ptrs = append(ptrs, &l.fc1W[i][j])
					f32 = append(f32, float32(l.fc1W[i][j]))
				}
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
		// fc1B [dFF].
		{
			ptrs := make([]*float64, dFF)
			f32 := make([]float32, dFF)
			for i := range l.fc1B {
				ptrs[i] = &l.fc1B[i]
				f32[i] = float32(l.fc1B[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
		// fc2W [dFF][dModel].
		{
			n := dFF * dModel
			ptrs := make([]*float64, 0, n)
			f32 := make([]float32, 0, n)
			for i := range l.fc2W {
				for j := range l.fc2W[i] {
					ptrs = append(ptrs, &l.fc2W[i][j])
					f32 = append(f32, float32(l.fc2W[i][j]))
				}
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
		// fc2B [dModel].
		{
			ptrs := make([]*float64, dModel)
			f32 := make([]float32, dModel)
			for i := range l.fc2B {
				ptrs[i] = &l.fc2B[i]
				f32[i] = float32(l.fc2B[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}

		// LN2 scale and bias.
		for _, bPtr := range []*[]float64{&l.ln2Scale, &l.ln2Bias} {
			ptrs := make([]*float64, dModel)
			f32 := make([]float32, dModel)
			for i := range *bPtr {
				ptrs[i] = &(*bPtr)[i]
				f32[i] = float32((*bPtr)[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
	}

	// Output projection: projW [dModel][outputLen].
	{
		n := m.config.DModel * outputLen
		ptrs := make([]*float64, 0, n)
		f32 := make([]float32, 0, n)
		for i := range m.projW {
			for j := range m.projW[i] {
				ptrs = append(ptrs, &m.projW[i][j])
				f32 = append(f32, float32(m.projW[i][j]))
			}
		}
		allParams = append(allParams, paramRef{ptrs, f32})
	}
	// projB [outputLen].
	{
		ptrs := make([]*float64, outputLen)
		f32 := make([]float32, outputLen)
		for i := range m.projB {
			ptrs[i] = &m.projB[i]
			f32[i] = float32(m.projB[i])
		}
		allParams = append(allParams, paramRef{ptrs, f32})
	}

	// AdamW state per parameter group.
	type adamF32 struct {
		m []float32
		v []float32
	}
	adamStates := make([]adamF32, len(allParams))
	for i, p := range allParams {
		adamStates[i] = adamF32{m: make([]float32, len(p.f32)), v: make([]float32, len(p.f32))}
	}

	result := &TrainResult{LossHistory: make([]float64, config.Epochs)}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			bs := end - start
			batchWindows := windows[start:end]
			batchLabels := labels[start*channels*outputLen : end*channels*outputLen]

			// Write f32 working copies back to model weights for forward/backward.
			for _, p := range allParams {
				for i, ptr := range p.f64Ptrs {
					*ptr = float64(p.f32[i])
				}
			}

			// Use the existing analytical forward/backward path.
			accGrads := newITransformerGrads(m.config)
			batchLoss := 0.0
			scale := 1.0 / float64(bs*channels*outputLen)

			for s := 0; s < bs; s++ {
				pred, cache := m.forwardWithCache(batchWindows[s])

				dOutput := make([][]float64, channels)
				for c := 0; c < channels; c++ {
					dOutput[c] = make([]float64, outputLen)
					for o := 0; o < outputLen; o++ {
						labelIdx := s*channels*outputLen + c*outputLen + o
						diff := pred[c][o] - batchLabels[labelIdx]
						batchLoss += diff * diff
						dOutput[c][o] = 2.0 * diff * scale
					}
				}

				m.backward(dOutput, cache, &accGrads)
			}
			batchLoss *= scale

			gradsF64 := accGrads.collectGrads(m.config)
			epochLoss += batchLoss
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range gradsF64 {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					s := config.GradClip / norm
					for i := range gradsF64 {
						gradsF64[i] *= s
					}
				}
			}

			// Distribute flat gradients back to per-parameter-group slices.
			paramGrads := make([][]float32, len(allParams))
			offset := 0
			for pi, p := range allParams {
				n := len(p.f32)
				paramGrads[pi] = make([]float32, n)
				for i := 0; i < n; i++ {
					paramGrads[pi][i] = float32(gradsF64[offset+i])
				}
				offset += n
			}

			// AdamW update using engine for large parameter tensors.
			lr := float32(warmupLR(config.LR, epoch, config.WarmupEpochs))
			tStep := float32(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			beta1 := float32(config.Beta1)
			beta2 := float32(config.Beta2)
			eps := float32(config.Epsilon)
			wd := float32(config.WeightDecay)

			for pi, p := range allParams {
				st := &adamStates[pi]
				grads := paramGrads[pi]

				if len(p.f32) >= 64 {
					m.adamStepEngine(ctx, p.f32, grads, st.m, st.v, lr, tStep, beta1, beta2, eps, wd)
				} else {
					for i := range p.f32 {
						st.m[i] = beta1*st.m[i] + (1-beta1)*grads[i]
						st.v[i] = beta2*st.v[i] + (1-beta2)*grads[i]*grads[i]
						mHat := st.m[i] / (1 - float32(math.Pow(float64(beta1), float64(tStep))))
						vHat := st.v[i] / (1 - float32(math.Pow(float64(beta2), float64(tStep))))
						p.f32[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*p.f32[i])
					}
				}
			}
		}

		// Write back final f32 values to model weights.
		for _, p := range allParams {
			for i, ptr := range p.f64Ptrs {
				*ptr = float64(p.f32[i])
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("itransformer: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// adamStepEngine performs a single AdamW update step using engine tensor
// operations for vectorized math (Sqrt, Div, Add, Sub, MulScalar).
func (m *ITransformer) adamStepEngine(ctx context.Context, params, grads, mState, vState []float32, lr, tStep, beta1, beta2, eps, wd float32) {
	n := len(params)

	// Update moments.
	for i := 0; i < n; i++ {
		mState[i] = beta1*mState[i] + (1-beta1)*grads[i]
		vState[i] = beta2*vState[i] + (1-beta2)*grads[i]*grads[i]
	}

	mHatCorr := 1 - float32(math.Pow(float64(beta1), float64(tStep)))
	vHatCorr := 1 - float32(math.Pow(float64(beta2), float64(tStep)))

	shape := []int{n}
	pT, err := tensor.New[float32](shape, params)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	mT, err := tensor.New[float32](shape, mState)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vT, err := tensor.New[float32](shape, vState)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}

	mHat, err := m.engine.DivScalar(ctx, mT, mHatCorr)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vHat, err := m.engine.DivScalar(ctx, vT, vHatCorr)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vSqrt, err := m.engine.Sqrt(ctx, vHat)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vSqrtEps, err := m.engine.AddScalar(ctx, vSqrt, eps)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	adamUpdate, err := m.engine.Div(ctx, mHat, vSqrtEps)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	wdTerm, err := m.engine.MulScalar(ctx, pT, wd)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	total, err := m.engine.Add(ctx, adamUpdate, wdTerm)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	scaled, err := m.engine.MulScalar(ctx, total, lr)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	newP, err := m.engine.Sub(ctx, pT, scaled)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}

	copy(params, newP.Data())
}

// adamStepScalar is the scalar fallback for AdamW when engine operations fail.
func (m *ITransformer) adamStepScalar(params, mState, vState []float32, lr, tStep, beta1, beta2, eps, wd float32) {
	for i := range params {
		mHat := mState[i] / (1 - float32(math.Pow(float64(beta1), float64(tStep))))
		vHat := vState[i] / (1 - float32(math.Pow(float64(beta2), float64(tStep))))
		params[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*params[i])
	}
}
