package disaggregated

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"google.golang.org/grpc"

	"github.com/zerfoo/ztensor/tensor"

	disaggpb "github.com/zerfoo/zerfoo/serve/disaggregated/proto"
)

// ForwardModel abstracts the single-token forward pass for decode.
// The model receives an input token tensor and returns logits.
type ForwardModel interface {
	Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)
}

// DecodeWorkerServer implements the DecodeWorker gRPC service.
// It receives KV blocks from a prefill worker, populates the KV cache,
// runs autoregressive decode steps, and streams generated tokens back.
type DecodeWorkerServer struct {
	disaggpb.UnimplementedDecodeWorkerServer

	model ForwardModel
}

// NewDecodeWorkerServer creates a new DecodeWorkerServer.
func NewDecodeWorkerServer(model ForwardModel) *DecodeWorkerServer {
	return &DecodeWorkerServer{
		model: model,
	}
}

// Decode implements the DecodeWorker gRPC service. It receives a DecodeRequest
// containing KV blocks and token IDs, runs autoregressive decode steps, and
// streams TokenStream messages back to the caller.
func (s *DecodeWorkerServer) Decode(req *disaggpb.DecodeRequest, stream grpc.ServerStreamingServer[disaggpb.TokenStream]) error {
	if req == nil {
		return fmt.Errorf("nil decode request")
	}

	maxTokens := int(req.GetMaxNewTokens())
	if maxTokens <= 0 {
		maxTokens = 1
	}

	// Deserialize KV blocks from the request into float32 slices.
	// Each KVBlock carries FP16-encoded key and value data.
	kvData := make([]kvBlockData, len(req.GetKvBlocks()))
	for i, block := range req.GetKvBlocks() {
		k, err := fp16BytesToFloat32(block.GetKData())
		if err != nil {
			return fmt.Errorf("deserialize k_data for layer %d block %d: %w",
				block.GetLayerIdx(), block.GetBlockIdx(), err)
		}
		v, err := fp16BytesToFloat32(block.GetVData())
		if err != nil {
			return fmt.Errorf("deserialize v_data for layer %d block %d: %w",
				block.GetLayerIdx(), block.GetBlockIdx(), err)
		}
		kvData[i] = kvBlockData{
			layerIdx: int(block.GetLayerIdx()),
			blockIdx: int(block.GetBlockIdx()),
			keys:     k,
			values:   v,
		}
	}

	// Build the initial input tensor from the last token in token_ids.
	tokenIDs := req.GetTokenIds()
	if len(tokenIDs) == 0 {
		return fmt.Errorf("empty token_ids in decode request")
	}
	lastToken := tokenIDs[len(tokenIDs)-1]

	ctx := stream.Context()

	// Autoregressive decode loop.
	for step := 0; step < maxTokens; step++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Create input tensor with the current token.
		inputData := []float32{float32(lastToken)}
		inputTensor, err := tensor.New[float32]([]int{1, 1}, inputData)
		if err != nil {
			return fmt.Errorf("create input tensor at step %d: %w", step, err)
		}

		// Run forward pass.
		logits, err := s.model.Forward(ctx, inputTensor)
		if err != nil {
			return fmt.Errorf("forward pass at step %d: %w", step, err)
		}

		// Sample the next token (greedy argmax, with optional temperature).
		nextToken := sampleGreedy(logits, req.GetTemperature())

		// Determine if we should stop.
		isEOS := nextToken == 2 // standard EOS token ID
		isLast := step == maxTokens-1

		finishReason := ""
		done := false
		if isEOS {
			finishReason = "stop"
			done = true
		} else if isLast {
			finishReason = "length"
			done = true
		}

		// Stream the token back.
		if err := stream.Send(&disaggpb.TokenStream{
			RequestId:    req.GetRequestId(),
			TokenId:      nextToken,
			Done:         done,
			FinishReason: finishReason,
		}); err != nil {
			return fmt.Errorf("send token at step %d: %w", step, err)
		}

		if done {
			break
		}

		lastToken = nextToken
	}

	return nil
}

// sampleGreedy returns the argmax token ID from the logits tensor.
// If temperature > 0, logits are divided by temperature before argmax.
func sampleGreedy(logits *tensor.TensorNumeric[float32], temperature float32) int32 {
	data := logits.Data()
	if len(data) == 0 {
		return 0
	}

	// Apply temperature scaling.
	if temperature > 0 && temperature != 1.0 {
		scaled := make([]float32, len(data))
		for i, v := range data {
			scaled[i] = v / temperature
		}
		data = scaled
	}

	// Argmax over the last dimension (vocab).
	bestIdx := 0
	bestVal := data[0]
	for i := 1; i < len(data); i++ {
		if data[i] > bestVal {
			bestVal = data[i]
			bestIdx = i
		}
	}
	return int32(bestIdx)
}

// kvBlockData holds deserialized KV cache data for a single block.
type kvBlockData struct {
	layerIdx int
	blockIdx int
	keys     []float32
	values   []float32
}

// fp16BytesToFloat32 converts a byte slice of IEEE 754 half-precision floats
// to a float32 slice.
func fp16BytesToFloat32(data []byte) ([]float32, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("fp16 data length %d is not a multiple of 2", len(data))
	}
	n := len(data) / 2
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2 : i*2+2])
		result[i] = fp16ToFloat32(bits)
	}
	return result, nil
}

// fp16ToFloat32 converts a single IEEE 754 half-precision value to float32.
func fp16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff

	switch {
	case exp == 0:
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: renormalize
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
		fallthrough
	case exp < 31:
		exp += 127 - 15
		return math.Float32frombits((sign << 31) | (exp << 23) | (frac << 13))
	default:
		// Inf / NaN
		exp = 255
		return math.Float32frombits((sign << 31) | (exp << 23) | (frac << 13))
	}
}
