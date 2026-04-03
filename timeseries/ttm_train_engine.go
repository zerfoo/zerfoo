package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/training/scheduler"
)

// mixerBlockF64WithCacheEngine runs one TSMixer block in float64 using
// the compute engine for MatMul operations. Caches activations for backward.
func (m *TTM) mixerBlockF64WithCacheEngine(ctx context.Context, x [][]float64, layer *ttmMixerLayerF64, nPatches, dModel int) ([][]float64, ttmMixerCacheF64, error) {
	var mc ttmMixerCacheF64
	mc.input = copyMatrix(x)

	// Time-mixing.
	normed, mean, invStd, centered := layerNormF64WithCache(x, layer.timeNormScale, layer.timeNormBias, dModel)
	mc.normed = normed
	mc.mean = mean
	mc.invStd = invStd
	mc.centered = centered

	transposed := transposeMatrix(normed, nPatches, dModel)
	mc.transposed = transposed

	mlp1Pre, err := linearF64Engine(m.engine, ctx, transposed, layer.timeMLP1W, layer.timeMLP1B, nPatches, nPatches)
	if err != nil {
		return nil, mc, fmt.Errorf("time MLP1: %w", err)
	}
	mc.mlp1Pre = mlp1Pre
	mlp1Out := geluMatrix(mlp1Pre)
	mc.mlp1Out = mlp1Out
	mlp2Out, err := linearF64Engine(m.engine, ctx, mlp1Out, layer.timeMLP2W, layer.timeMLP2B, nPatches, nPatches)
	if err != nil {
		return nil, mc, fmt.Errorf("time MLP2: %w", err)
	}
	mc.mlp2Out = mlp2Out

	h := transposeMatrix(mlp2Out, dModel, nPatches)
	for p := 0; p < nPatches; p++ {
		for j := 0; j < dModel; j++ {
			h[p][j] += mc.input[p][j]
		}
	}

	// Feature-mixing.
	if len(layer.featMLP1W) > 0 {
		mc.featInput = copyMatrix(h)
		normed, mean, invStd, centered := layerNormF64WithCache(h, layer.featNormScale, layer.featNormBias, dModel)
		mc.featNormed = normed
		mc.featMean = mean
		mc.featInvStd = invStd
		mc.featCentered = centered

		ffnDim := len(layer.featMLP1B)
		mlp1Pre, err := linearF64Engine(m.engine, ctx, normed, layer.featMLP1W, layer.featMLP1B, dModel, ffnDim)
		if err != nil {
			return nil, mc, fmt.Errorf("feat MLP1: %w", err)
		}
		mc.featMLP1Pre = mlp1Pre
		mlp1Out := geluMatrix(mlp1Pre)
		mc.featMLP1Out = mlp1Out
		mlp2Out, err := linearF64Engine(m.engine, ctx, mlp1Out, layer.featMLP2W, layer.featMLP2B, ffnDim, dModel)
		if err != nil {
			return nil, mc, fmt.Errorf("feat MLP2: %w", err)
		}
		mc.featMLP2Out = mlp2Out

		for p := 0; p < nPatches; p++ {
			for j := 0; j < dModel; j++ {
				mlp2Out[p][j] += mc.featInput[p][j]
			}
		}
		return mlp2Out, mc, nil
	}
	return h, mc, nil
}

// forwardF64WithCacheEngine runs the TTM forward pass in float64, using
// the engine for MatMul operations (patch embedding, TSMixer MLPs, forecast
// head). Layer norm, GELU, and residual connections remain on CPU. The cached
// activations are identical in structure to forwardF64WithCache so that
// backwardF64 can consume them unchanged.
func (m *TTM) forwardF64WithCacheEngine(ctx context.Context, input [][]float64, params *ttmParamsF64) ([]float64, *ttmCacheF64, error) {
	numPatches := m.config.NumPatches()
	forecastPatches := m.config.ForecastPatches()
	dModel := m.config.DModel
	channels := len(input)

	cache := &ttmCacheF64{
		channels: make([]ttmChannelCache, channels),
	}
	chanOutputs := make([][]float64, channels)

	for ch := 0; ch < channels; ch++ {
		cc := &cache.channels[ch]

		// Extract patches.
		cc.patches = make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			start := p * m.config.PatchLen
			cc.patches[p] = make([]float64, m.config.PatchLen)
			copy(cc.patches[p], input[ch][start:start+m.config.PatchLen])
		}

		// Patch embedding.
		var err error
		cc.embedded, err = linearF64Engine(m.engine, ctx, cc.patches, params.patchEmbW, params.patchEmbB, m.config.PatchLen, dModel)
		if err != nil {
			return nil, nil, fmt.Errorf("ttm engine: patch embedding: %w", err)
		}

		// Encoder.
		x := cc.embedded
		cc.encoderCaches = make([]ttmMixerCacheF64, len(params.encoder))
		for i, layer := range params.encoder {
			x, cc.encoderCaches[i], err = m.mixerBlockF64WithCacheEngine(ctx, x, &layer, numPatches, dModel)
			if err != nil {
				return nil, nil, fmt.Errorf("ttm engine: encoder block %d: %w", i, err)
			}
		}
		cc.encoderOutput = copyMatrix(x)

		// Slice to last forecastPatches.
		if numPatches > forecastPatches {
			x = x[numPatches-forecastPatches:]
		}

		// Decoder.
		cc.decoderCaches = make([]ttmMixerCacheF64, len(params.decoder))
		for i, layer := range params.decoder {
			x, cc.decoderCaches[i], err = m.mixerBlockF64WithCacheEngine(ctx, x, &layer, forecastPatches, dModel)
			if err != nil {
				return nil, nil, fmt.Errorf("ttm engine: decoder block %d: %w", i, err)
			}
		}

		// Flatten.
		cc.flatInput = make([]float64, forecastPatches*dModel)
		for p := 0; p < forecastPatches; p++ {
			copy(cc.flatInput[p*dModel:(p+1)*dModel], x[p])
		}

		headIn := forecastPatches * dModel
		headOut, err := linearF64Engine(m.engine, ctx, [][]float64{cc.flatInput}, params.headW, params.headB, headIn, m.config.ForecastLen)
		if err != nil {
			return nil, nil, fmt.Errorf("ttm engine: forecast head: %w", err)
		}
		chanOutputs[ch] = headOut[0]
	}

	result := make([]float64, m.config.ForecastLen)
	for ch := 0; ch < channels; ch++ {
		for j := 0; j < m.config.ForecastLen; j++ {
			result[j] += chanOutputs[ch][j]
		}
	}
	for j := range result {
		result[j] /= float64(channels)
	}
	return result, cache, nil
}

// batchForwardF64Engine runs the TTM forward pass for an entire batch using
// the compute engine for MatMul. Each sample's channels are processed
// independently through the TSMixer pipeline, then averaged.
//
// Input: windows[batch][channels][contextLen]
// Output: [batch][forecastLen]
func (m *TTM) batchForwardF64Engine(ctx context.Context, windows [][][]float64, params *ttmParamsF64) ([][]float64, error) {
	batch := len(windows)
	if batch == 0 {
		return nil, nil
	}
	numPatches := m.config.NumPatches()
	forecastPatches := m.config.ForecastPatches()
	dModel := m.config.DModel
	patchLen := m.config.PatchLen
	forecastLen := m.config.ForecastLen

	results := make([][]float64, batch)

	// Collect all (sample, channel) pairs into a flat list of patch matrices.
	type chanInfo struct {
		sampleIdx int
	}
	var infos []chanInfo
	var allPatches [][][]float64

	for s := 0; s < batch; s++ {
		channels := len(windows[s])
		for ch := 0; ch < channels; ch++ {
			patches := make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				start := p * patchLen
				patches[p] = make([]float64, patchLen)
				copy(patches[p], windows[s][ch][start:start+patchLen])
			}
			allPatches = append(allPatches, patches)
			infos = append(infos, chanInfo{sampleIdx: s})
		}
	}

	totalChans := len(allPatches)

	// Concatenate all patches into [totalChans*numPatches, patchLen] for batched embedding.
	bigPatches := make([][]float64, totalChans*numPatches)
	for i, patches := range allPatches {
		for p := 0; p < numPatches; p++ {
			bigPatches[i*numPatches+p] = patches[p]
		}
	}

	// Patch embedding: [totalChans*numPatches, patchLen] -> [totalChans*numPatches, dModel].
	embedded, err := linearF64Engine(m.engine, ctx, bigPatches, params.patchEmbW, params.patchEmbB, patchLen, dModel)
	if err != nil {
		return nil, fmt.Errorf("ttm batch: patch embedding: %w", err)
	}

	// Split back to per-channel: each [numPatches, dModel].
	chanEmbedded := make([][][]float64, totalChans)
	for i := 0; i < totalChans; i++ {
		chanEmbedded[i] = embedded[i*numPatches : (i+1)*numPatches]
	}

	// Process encoder and decoder per-channel.
	chanOutputs := make([][]float64, totalChans)
	for i := 0; i < totalChans; i++ {
		x := chanEmbedded[i]

		for li, layer := range params.encoder {
			x, err = m.mixerBlockF64Engine(ctx, x, &layer, numPatches, dModel)
			if err != nil {
				return nil, fmt.Errorf("ttm batch: encoder block %d: %w", li, err)
			}
		}

		if numPatches > forecastPatches {
			x = x[numPatches-forecastPatches:]
		}

		for li, layer := range params.decoder {
			x, err = m.mixerBlockF64Engine(ctx, x, &layer, forecastPatches, dModel)
			if err != nil {
				return nil, fmt.Errorf("ttm batch: decoder block %d: %w", li, err)
			}
		}

		flatInput := make([]float64, forecastPatches*dModel)
		for p := 0; p < forecastPatches; p++ {
			copy(flatInput[p*dModel:(p+1)*dModel], x[p])
		}
		chanOutputs[i] = flatInput
	}

	// Batched forecast head: [totalChans, forecastPatches*dModel] -> [totalChans, forecastLen].
	headIn := forecastPatches * dModel
	headOut, err := linearF64Engine(m.engine, ctx, chanOutputs, params.headW, params.headB, headIn, forecastLen)
	if err != nil {
		return nil, fmt.Errorf("ttm batch: forecast head: %w", err)
	}

	// Accumulate per-sample channel averages.
	chanCounts := make([]int, batch)
	for s := 0; s < batch; s++ {
		results[s] = make([]float64, forecastLen)
	}
	for i, info := range infos {
		s := info.sampleIdx
		chanCounts[s]++
		for j := 0; j < forecastLen; j++ {
			results[s][j] += headOut[i][j]
		}
	}
	for s := 0; s < batch; s++ {
		c := float64(chanCounts[s])
		for j := 0; j < forecastLen; j++ {
			results[s][j] /= c
		}
	}

	return results, nil
}

// mixerBlockF64Engine runs one TSMixer block in float64 using the engine
// for MatMul (without caching activations for backward pass).
func (m *TTM) mixerBlockF64Engine(ctx context.Context, x [][]float64, layer *ttmMixerLayerF64, nPatches, dModel int) ([][]float64, error) {
	residual := copyMatrix(x)

	normed := layerNormF64(x, layer.timeNormScale, layer.timeNormBias, dModel)
	transposed := transposeMatrix(normed, nPatches, dModel)

	mlp1, err := linearF64Engine(m.engine, ctx, transposed, layer.timeMLP1W, layer.timeMLP1B, nPatches, nPatches)
	if err != nil {
		return nil, fmt.Errorf("time MLP1: %w", err)
	}
	mlp1 = geluMatrix(mlp1)
	mlp2, err := linearF64Engine(m.engine, ctx, mlp1, layer.timeMLP2W, layer.timeMLP2B, nPatches, nPatches)
	if err != nil {
		return nil, fmt.Errorf("time MLP2: %w", err)
	}

	h := transposeMatrix(mlp2, dModel, nPatches)
	for p := 0; p < nPatches; p++ {
		for j := 0; j < dModel; j++ {
			h[p][j] += residual[p][j]
		}
	}

	if len(layer.featMLP1W) > 0 {
		featResidual := copyMatrix(h)
		normed = layerNormF64(h, layer.featNormScale, layer.featNormBias, dModel)

		ffnDim := len(layer.featMLP1B)
		mlp1, err = linearF64Engine(m.engine, ctx, normed, layer.featMLP1W, layer.featMLP1B, dModel, ffnDim)
		if err != nil {
			return nil, fmt.Errorf("feat MLP1: %w", err)
		}
		mlp1 = geluMatrix(mlp1)
		mlp2, err = linearF64Engine(m.engine, ctx, mlp1, layer.featMLP2W, layer.featMLP2B, ffnDim, dModel)
		if err != nil {
			return nil, fmt.Errorf("feat MLP2: %w", err)
		}

		for p := 0; p < nPatches; p++ {
			for j := 0; j < dModel; j++ {
				mlp2[p][j] += featResidual[p][j]
			}
		}
		return mlp2, nil
	}
	return h, nil
}

// trainWindowedEngine is the engine-accelerated training path. The forward
// pass dispatches MatMul operations to the compute.Engine[float32], while
// the backward pass (gradient computation) stays on CPU.
func (m *TTM) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	ctx := context.Background()
	nSamples := len(windows)
	outDim := m.config.ForecastLen

	params := m.extractParamsF64()
	trainableParams := params.flatParamsExcluding(m.config.FreezeEncoder)
	nParams := len(trainableParams)
	adamM := make([]float64, nParams)
	adamV := make([]float64, nParams)

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

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

			grads := make([]float64, nParams)
			batchLoss := 0.0

			for s := 0; s < bs; s++ {
				pred, cache, err := m.forwardF64WithCacheEngine(ctx, windows[start+s], params)
				if err != nil {
					return nil, fmt.Errorf("ttm: engine forward: %w", err)
				}
				sampleLabels := labels[(start+s)*outDim : (start+s+1)*outDim]

				dOutput := make([]float64, outDim)
				for j := 0; j < outDim; j++ {
					diff := pred[j] - sampleLabels[j]
					batchLoss += diff * diff
					dOutput[j] = 2.0 * diff / float64(bs*outDim)
				}

				sampleGrads := m.backwardF64(dOutput, params, cache, m.config.FreezeEncoder)
				for pi := range grads {
					grads[pi] += sampleGrads[pi]
				}
			}

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := config.GradClip / norm
					for i := range grads {
						grads[i] *= scale
					}
				}
			}

			lr := scheduler.WarmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			for i := range trainableParams {
				adamM[i] = config.Beta1*adamM[i] + (1-config.Beta1)*grads[i]
				adamV[i] = config.Beta2*adamV[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := adamM[i] / (1 - math.Pow(config.Beta1, t))
				vHat := adamV[i] / (1 - math.Pow(config.Beta2, t))
				*trainableParams[i] = *trainableParams[i] - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*trainableParams[i]))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("ttm: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	m.writeBackF32(params)

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}
