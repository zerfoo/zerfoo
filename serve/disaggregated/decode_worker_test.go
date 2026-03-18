package disaggregated

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"github.com/zerfoo/ztensor/tensor"

	disaggpb "github.com/zerfoo/zerfoo/serve/disaggregated/proto"
)

// stubForwardModel implements ForwardModel. It returns logits where the
// token at the target index has the highest value, cycling through a
// sequence of tokens before finally returning EOS (token 2).
type stubForwardModel struct {
	calls    int
	sequence []int32
}

func (m *stubForwardModel) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	vocabSize := 8
	logits := make([]float32, vocabSize)
	for i := range logits {
		logits[i] = -100.0
	}

	var targetToken int32
	if m.calls < len(m.sequence) {
		targetToken = m.sequence[m.calls]
	} else {
		targetToken = 2 // EOS
	}
	logits[targetToken] = 10.0
	m.calls++

	t, err := tensor.New[float32]([]int{1, vocabSize}, logits)
	if err != nil {
		panic(err)
	}
	return t, nil
}

// mockTokenStreamServer implements grpc.ServerStreamingServer[disaggpb.TokenStream]
// by collecting sent messages in memory.
type mockTokenStreamServer struct {
	grpc.ServerStream
	ctx    context.Context
	tokens []*disaggpb.TokenStream
}

func newMockTokenStreamServer() *mockTokenStreamServer {
	return &mockTokenStreamServer{ctx: context.Background()}
}

func (m *mockTokenStreamServer) Send(msg *disaggpb.TokenStream) error {
	m.tokens = append(m.tokens, msg)
	return nil
}

func (m *mockTokenStreamServer) Context() context.Context {
	return m.ctx
}

func (m *mockTokenStreamServer) SetHeader(metadata.MD) error  { return nil }
func (m *mockTokenStreamServer) SendHeader(metadata.MD) error { return nil }
func (m *mockTokenStreamServer) SetTrailer(metadata.MD)       {}
func (m *mockTokenStreamServer) SendMsg(interface{}) error    { return nil }
func (m *mockTokenStreamServer) RecvMsg(interface{}) error    { return nil }

// float32ToFP16Bytes converts float32 values to FP16 little-endian bytes.
func float32ToFP16Bytes(vals []float32) []byte {
	buf := make([]byte, len(vals)*2)
	for i, v := range vals {
		bits := float32ToFP16(v)
		binary.LittleEndian.PutUint16(buf[i*2:], bits)
	}
	return buf
}

func float32ToFP16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 1)
	exp := int((bits>>23)&0xff) - 127
	frac := bits & 0x7fffff

	switch {
	case exp > 15:
		return (sign << 15) | 0x7c00
	case exp < -14:
		return sign << 15
	default:
		return (sign << 15) | uint16(exp+15)<<10 | uint16(frac>>13)
	}
}

func TestDecodeWorker(t *testing.T) {
	model := &stubForwardModel{sequence: []int32{5, 3}}
	srv := NewDecodeWorkerServer(model)

	kvKey := float32ToFP16Bytes([]float32{1.0, 2.0, 3.0, 4.0})
	kvVal := float32ToFP16Bytes([]float32{5.0, 6.0, 7.0, 8.0})

	req := &disaggpb.DecodeRequest{
		RequestId: "test-req-1",
		KvBlocks: []*disaggpb.KVBlock{
			{RequestId: "test-req-1", LayerIdx: 0, BlockIdx: 0, KData: kvKey, VData: kvVal},
			{RequestId: "test-req-1", LayerIdx: 1, BlockIdx: 0, KData: kvKey, VData: kvVal},
		},
		TokenIds:     []int32{1, 4},
		MaxNewTokens: 10,
		Temperature:  0,
	}

	stream := newMockTokenStreamServer()
	err := srv.Decode(req, stream)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}

	expectedTokens := []int32{5, 3, 2}
	if len(stream.tokens) != len(expectedTokens) {
		t.Fatalf("expected %d tokens, got %d", len(expectedTokens), len(stream.tokens))
	}
	for i, msg := range stream.tokens {
		if msg.TokenId != expectedTokens[i] {
			t.Errorf("token[%d] = %d, want %d", i, msg.TokenId, expectedTokens[i])
		}
	}

	last := stream.tokens[len(stream.tokens)-1]
	if !last.Done {
		t.Error("expected final message to have done=true")
	}
	if last.FinishReason != "stop" {
		t.Errorf("expected finish_reason='stop', got %q", last.FinishReason)
	}
}

func TestDecodeWorker_MaxTokens(t *testing.T) {
	model := &stubForwardModel{sequence: []int32{6, 6, 6, 6, 6}}
	srv := NewDecodeWorkerServer(model)

	req := &disaggpb.DecodeRequest{
		RequestId:    "test-req-2",
		TokenIds:     []int32{1},
		MaxNewTokens: 3,
		Temperature:  1.0,
	}

	stream := newMockTokenStreamServer()
	err := srv.Decode(req, stream)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}

	if len(stream.tokens) != 3 {
		t.Fatalf("expected 3 tokens, got %d", len(stream.tokens))
	}

	last := stream.tokens[len(stream.tokens)-1]
	if last.FinishReason != "length" {
		t.Errorf("expected finish_reason='length', got %q", last.FinishReason)
	}
}

func TestFP16BytesToFloat32(t *testing.T) {
	input := []byte{0x00, 0x3C} // 1.0 in little-endian FP16
	result, err := fp16BytesToFloat32(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 element, got %d", len(result))
	}
	if result[0] != 1.0 {
		t.Errorf("expected 1.0, got %f", result[0])
	}

	_, err = fp16BytesToFloat32([]byte{0x00, 0x3C, 0x00})
	if err == nil {
		t.Error("expected error for odd-length input")
	}
}
