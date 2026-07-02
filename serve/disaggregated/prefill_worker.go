// Package disaggregated implements disaggregated prefill/decode serving.
package disaggregated

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	disaggpb "github.com/zerfoo/zerfoo/serve/disaggregated/proto"

	"google.golang.org/grpc"
)

// Prefiller runs the prefill forward pass on prompt tokens and returns
// per-layer KV cache data as float32 slices. Implementations wrap the
// real inference session; tests can supply a stub.
type Prefiller interface {
	// Prefill runs the prompt through the model graph. It returns the
	// number of layers and a function to retrieve the KV data for each
	// layer. The returned float32 slices are the raw key/value cache
	// contents after the prefill pass.
	Prefill(ctx context.Context, tokenIDs []int32) (numLayers int, getKV func(layer int) (k, v []float32, err error), err error)
}

// PrefillWorkerServer implements the disaggpb.PrefillWorkerServer gRPC
// service. It delegates the forward pass to a Prefiller and streams
// the resulting KV blocks as FP16 bytes.
type PrefillWorkerServer struct {
	disaggpb.UnimplementedPrefillWorkerServer
	prefiller Prefiller
}

// NewPrefillWorkerServer creates a new PrefillWorkerServer.
func NewPrefillWorkerServer(p Prefiller) *PrefillWorkerServer {
	return &PrefillWorkerServer{prefiller: p}
}

// Register registers the server with a gRPC service registrar.
func (s *PrefillWorkerServer) Register(reg grpc.ServiceRegistrar) {
	disaggpb.RegisterPrefillWorkerServer(reg, s)
}

// Prefill runs the prefill forward pass on the prompt tokens from req,
// serialises each layer's KV cache as FP16 bytes, and streams them back
// as KVBlockStream messages. A final message with Done=true signals
// completion.
func (s *PrefillWorkerServer) Prefill(req *disaggpb.PreFillRequest, stream grpc.ServerStreamingServer[disaggpb.KVBlockStream]) error {
	ctx := stream.Context()
	reqID := req.GetRequestId()
	tokenIDs := req.GetTokenIds()

	if len(tokenIDs) == 0 {
		return fmt.Errorf("prefill: empty token list")
	}

	numLayers, getKV, err := s.prefiller.Prefill(ctx, tokenIDs)
	if err != nil {
		return fmt.Errorf("prefill forward: %w", err)
	}

	for layer := range numLayers {
		k, v, err := getKV(layer)
		if err != nil {
			return fmt.Errorf("prefill: read KV layer %d: %w", layer, err)
		}

		kBytes := float32sToFP16Bytes(k)
		vBytes := float32sToFP16Bytes(v)

		if err := stream.Send(&disaggpb.KVBlockStream{
			Block: &disaggpb.KVBlock{
				RequestId: reqID,
				LayerIdx:  int32(layer),
				BlockIdx:  0,
				KData:     kBytes,
				VData:     vBytes,
			},
		}); err != nil {
			return fmt.Errorf("prefill: stream layer %d: %w", layer, err)
		}
	}

	// Send terminal message.
	return stream.Send(&disaggpb.KVBlockStream{Done: true})
}

// float32sToFP16Bytes converts a float32 slice to IEEE 754 half-precision
// bytes (2 bytes per value, little-endian).
func float32sToFP16Bytes(vals []float32) []byte {
	buf := make([]byte, len(vals)*2)
	for i, v := range vals {
		fp16 := float32ToFP16Bits(v)
		binary.LittleEndian.PutUint16(buf[i*2:], fp16)
	}
	return buf
}

// float32ToFP16Bits converts a single float32 to its IEEE 754 half-precision
// bit pattern. This is a minimal inline conversion that avoids importing the
// float16 package (which is a separate module not depended upon directly by
// the serve package).
func float32ToFP16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127 + 15
	mant := uint16((b >> 13) & 0x03ff)

	switch {
	case exp <= 0:
		// Underflow → zero (flush subnormals for simplicity).
		return sign
	case exp >= 0x1f:
		// Overflow → infinity.
		return sign | 0x7c00
	default:
		return sign | uint16(exp<<10) | mant
	}
}
