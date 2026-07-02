package vision

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// LTXVAEDecoderSkeleton is a SKELETON of the LTX-2 video VAE decoder (E127
// T127.4.4). It wires the E127 convolutional primitives -- Conv3d,
// ConvTranspose3d (zerfoo#896) and GroupNormalization -- together with GELU
// into a representative residual + 2x-upsample decode forward, to prove the
// primitive set composes end-to-end and produces correctly shaped output.
//
// It is intentionally NOT the weight-accurate LTX-2 VAE: the real decoder's
// block_out_channels = [256,512,1024,2048], 32x spatial / 8x temporal upsample,
// stacked ResNet3D blocks, mid-block 3D attention, and the actual converted
// weights are the remainder of T127.4.4. Treat this as scaffolding/integration
// surface, not a correctness reference. The weights here are small deterministic
// fixtures so the forward runs and is finite.
//
// Pipeline (latent [N, Clatent, D, H, W] -> [N, OutChannels, 2D, 2H, 2W]):
//
//	in-conv (1x1x1)
//	-> residual{ GroupNorm -> GELU -> Conv3d(3x3x3, same) } (+ skip)
//	-> ConvTranspose3d(2x2x2, stride 2)  [2x upsample]
//	-> GroupNorm -> GELU -> out-conv(3x3x3, same -> OutChannels)
type LTXVAEDecoderSkeleton[T tensor.Float] struct {
	engine compute.Engine[T]

	inConv, midConv, outConv *core.Conv3d[T]
	up                       *core.ConvTranspose3d[T]
	gn1, gn2                 *normalization.GroupNormalization[T]
	act                      *activations.Gelu[T]

	inW, midW, outW, upW *tensor.TensorNumeric[T]
	outputShape          []int
}

// LTXVAEDecoderConfig parameterizes the skeleton.
type LTXVAEDecoderConfig struct {
	LatentChannels int // input channels (e.g. 128 for LTX-2)
	MidChannels    int // working channels (must be divisible by NumGroups)
	OutChannels    int // output channels (3 for RGB)
	NumGroups      int
	Epsilon        float64
}

// fixtureWeights builds a deterministic small weight tensor of the given shape.
func fixtureWeights[T tensor.Float](shape []int, seed int) (*tensor.TensorNumeric[T], error) {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]T, size)
	s := uint64(seed*2654435761 + 1)
	for i := range data {
		s = s*6364136223846793005 + 1442695040888963407
		data[i] = T(float64(int64(s>>40))/float64(1<<24)*0.1 - 0.05) // ~[-0.05,0.05]
	}
	return tensor.New[T](shape, data)
}

// NewLTXVAEDecoderSkeleton builds the skeleton with deterministic fixture weights.
func NewLTXVAEDecoderSkeleton[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	cfg LTXVAEDecoderConfig,
) (*LTXVAEDecoderSkeleton[T], error) {
	if cfg.OutChannels == 0 {
		cfg.OutChannels = 3
	}
	if cfg.NumGroups == 0 {
		cfg.NumGroups = 1
	}
	if cfg.Epsilon == 0 {
		cfg.Epsilon = 1e-5
	}
	if cfg.MidChannels%cfg.NumGroups != 0 {
		return nil, fmt.Errorf("LTXVAEDecoderSkeleton: MidChannels %d not divisible by NumGroups %d", cfg.MidChannels, cfg.NumGroups)
	}

	mkGN := func() (*normalization.GroupNormalization[T], error) {
		scaleData := make([]T, cfg.MidChannels)
		for i := range scaleData {
			scaleData[i] = T(1) // identity scale, zero bias -> clean fixture
		}
		biasData := make([]T, cfg.MidChannels)
		st, err := tensor.New[T]([]int{cfg.MidChannels}, scaleData)
		if err != nil {
			return nil, err
		}
		bt, err := tensor.New[T]([]int{cfg.MidChannels}, biasData)
		if err != nil {
			return nil, err
		}
		sp, err := graph.NewParameter[T]("scale", st, tensor.New[T])
		if err != nil {
			return nil, err
		}
		bp, err := graph.NewParameter[T]("bias", bt, tensor.New[T])
		if err != nil {
			return nil, err
		}
		return normalization.NewGroupNormalizationWithParams[T](engine, ops, cfg.NumGroups, ops.FromFloat64(cfg.Epsilon), sp, bp), nil
	}

	d := &LTXVAEDecoderSkeleton[T]{engine: engine, act: activations.NewGelu[T](engine, ops)}
	var err error
	if d.gn1, err = mkGN(); err != nil {
		return nil, err
	}
	if d.gn2, err = mkGN(); err != nil {
		return nil, err
	}
	d.inConv = core.NewConv3d[T](engine, ops, []int{1, 1, 1}, []int{0, 0, 0, 0, 0, 0}, []int{1, 1, 1}, 1)
	d.midConv = core.NewConv3d[T](engine, ops, []int{1, 1, 1}, []int{1, 1, 1, 1, 1, 1}, []int{1, 1, 1}, 1)
	d.outConv = core.NewConv3d[T](engine, ops, []int{1, 1, 1}, []int{1, 1, 1, 1, 1, 1}, []int{1, 1, 1}, 1)
	d.up = core.NewConvTranspose3d[T](engine, ops, []int{2, 2, 2}, []int{0, 0, 0, 0, 0, 0}, []int{1, 1, 1}, []int{0, 0, 0}, 1)

	// Weights: Conv3d [Cout,Cin,kD,kH,kW]; ConvTranspose3d [Cin,Cout,kD,kH,kW].
	if d.inW, err = fixtureWeights[T]([]int{cfg.MidChannels, cfg.LatentChannels, 1, 1, 1}, 1); err != nil {
		return nil, err
	}
	if d.midW, err = fixtureWeights[T]([]int{cfg.MidChannels, cfg.MidChannels, 3, 3, 3}, 2); err != nil {
		return nil, err
	}
	if d.upW, err = fixtureWeights[T]([]int{cfg.MidChannels, cfg.MidChannels, 2, 2, 2}, 3); err != nil {
		return nil, err
	}
	if d.outW, err = fixtureWeights[T]([]int{cfg.OutChannels, cfg.MidChannels, 3, 3, 3}, 4); err != nil {
		return nil, err
	}
	return d, nil
}

// Forward decodes a latent [N, LatentChannels, D, H, W] to [N, OutChannels, 2D, 2H, 2W].
func (d *LTXVAEDecoderSkeleton[T]) Forward(ctx context.Context, latent *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	e := d.engine
	h, err := d.inConv.Forward(ctx, latent, d.inW)
	if err != nil {
		return nil, fmt.Errorf("vae skeleton: in-conv: %w", err)
	}
	// Residual block: h = h + Conv3d(GELU(GroupNorm(h))).
	r, err := d.gn1.Forward(ctx, h)
	if err != nil {
		return nil, fmt.Errorf("vae skeleton: gn1: %w", err)
	}
	if r, err = d.act.Forward(ctx, r); err != nil {
		return nil, fmt.Errorf("vae skeleton: act1: %w", err)
	}
	if r, err = d.midConv.Forward(ctx, r, d.midW); err != nil {
		return nil, fmt.Errorf("vae skeleton: mid-conv: %w", err)
	}
	if h, err = e.Add(ctx, h, r); err != nil {
		return nil, fmt.Errorf("vae skeleton: residual add: %w", err)
	}
	// Upsample 2x in each spatial dim.
	if h, err = d.up.Forward(ctx, h, d.upW); err != nil {
		return nil, fmt.Errorf("vae skeleton: upsample: %w", err)
	}
	// Out head: GroupNorm -> GELU -> Conv3d -> OutChannels.
	if h, err = d.gn2.Forward(ctx, h); err != nil {
		return nil, fmt.Errorf("vae skeleton: gn2: %w", err)
	}
	if h, err = d.act.Forward(ctx, h); err != nil {
		return nil, fmt.Errorf("vae skeleton: act2: %w", err)
	}
	out, err := d.outConv.Forward(ctx, h, d.outW)
	if err != nil {
		return nil, fmt.Errorf("vae skeleton: out-conv: %w", err)
	}
	d.outputShape = out.Shape()
	return out, nil
}

// OutputShape returns the last forward output shape.
func (d *LTXVAEDecoderSkeleton[T]) OutputShape() []int { return d.outputShape }
