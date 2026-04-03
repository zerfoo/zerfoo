// Package residual provides residual connection layers for neural networks.
//
// Standard transformer architectures use additive residual connections:
// each layer's output is added to its input before being passed forward.
// While effective, this fixed scheme weights all previous layers equally,
// limiting the network's ability to route information across depth.
//
// This package implements Attention Residuals (arXiv:2603.15031, Kimi Team,
// 2026), which replace fixed additive residuals with learned, softmax-weighted
// aggregation over depth. Each layer carries a small pseudo-query vector that
// attends over RMSNorm-projected keys from all preceding layer outputs,
// producing per-layer attention weights that dynamically control how much
// each earlier representation contributes to the current layer's input.
//
// Two variants are provided:
//
//   - [AttnRes]: Full Attention Residuals. Every layer attends over every
//     previous layer output. This gives maximum expressiveness but requires
//     O(L*d) memory to store all L layer outputs of dimension d.
//
//   - [BlockAttnRes]: Block Attention Residuals. Layers are partitioned into
//     N blocks of S layers each. Within a block, outputs are accumulated via
//     standard addition. At block boundaries, softmax attention aggregates
//     block-level representations. This reduces memory from O(L*d) to O(N*d)
//     while recovering most of the benefit of full AttnRes. The paper shows
//     that N=8 blocks recovers the majority of full AttnRes gains.
//
// # Usage: AttnRes in a transformer layer loop
//
// For full attention residuals, create one AttnRes per layer and collect
// all layer outputs:
//
//	// During graph construction:
//	var layerOutputs []*tensor.TensorNumeric[T]
//	layerOutputs = append(layerOutputs, embedOutput) // layer 0 = embedding
//
//	for i := 0; i < numLayers; i++ {
//	    ar, _ := residual.NewAttnRes[float32](
//	        fmt.Sprintf("layer_%d_attnres", i), engine, ops, modelDim,
//	    )
//	    // Aggregate previous layers via attention.
//	    hidden, _ = ar.Forward(ctx, layerOutputs...)
//	    // Run attention + FFN on the aggregated hidden state.
//	    hidden = transformerLayer(ctx, hidden, i)
//	    layerOutputs = append(layerOutputs, hidden)
//	}
//
// # Usage: BlockAttnRes with block boundaries
//
// For block attention residuals, accumulate within blocks and attend across
// block representations:
//
//	blockSize := numLayers / 8 // N=8 blocks
//	bar, _ := residual.NewBlockAttnRes[float32](engine, ops, blockSize, modelDim, 1e-5)
//
//	var completedBlocks []*tensor.TensorNumeric[T]
//	var blockAccum *tensor.TensorNumeric[T] // running sum within current block
//
//	for i := 0; i < numLayers; i++ {
//	    // At block boundary (except first), finalize the block.
//	    if i > 0 && i%blockSize == 0 {
//	        completedBlocks = append(completedBlocks, blockAccum)
//	        blockAccum = nil
//	    }
//	    // Compute attention-weighted residual from blocks.
//	    partial := blockAccum
//	    if partial == nil { partial = hidden }
//	    hidden, _ = bar.Forward(ctx, hidden, completedBlocks, partial)
//	    // Run transformer layer.
//	    hidden = transformerLayer(ctx, hidden, i)
//	    // Accumulate into current block.
//	    if blockAccum == nil {
//	        blockAccum = hidden
//	    } else {
//	        blockAccum, _ = engine.Add(ctx, blockAccum, hidden)
//	    }
//	}
//
// # GGUF metadata
//
// Residual mode is configured via GGUF metadata keys (see inference package):
//
//   - general.residual_mode: "standard" (default), "attnres", or "block_attnres"
//   - general.attnres_blocks: number of blocks for block_attnres (default 8)
//
// # Stability
//
// This package is experimental. The API may change as the approach matures.
package residual
