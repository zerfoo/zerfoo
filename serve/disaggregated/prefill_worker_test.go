package disaggregated

import (
	"context"
	"fmt"
	"math"
	"testing"

	disaggpb "github.com/zerfoo/zerfoo/serve/disaggregated/proto"

	"google.golang.org/grpc"
)

// stubPrefiller returns fixed KV data for each layer.
type stubPrefiller struct {
	numLayers int
	kvSize    int // number of float32 values per K/V per layer
}

func (s *stubPrefiller) Prefill(_ context.Context, tokenIDs []int32) (int, func(int) ([]float32, []float32, error), error) {
	if len(tokenIDs) == 0 {
		return 0, nil, fmt.Errorf("empty tokens")
	}
	getKV := func(layer int) ([]float32, []float32, error) {
		if layer < 0 || layer >= s.numLayers {
			return nil, nil, fmt.Errorf("layer %d out of range", layer)
		}
		k := make([]float32, s.kvSize)
		v := make([]float32, s.kvSize)
		for i := range s.kvSize {
			k[i] = float32(layer*1000 + i)
			v[i] = float32(layer*1000 + i + 500)
		}
		return k, v, nil
	}
	return s.numLayers, getKV, nil
}

// recorderStream collects KVBlockStream messages sent during Prefill.
type recorderStream struct {
	grpc.ServerStreamingServer[disaggpb.KVBlockStream]
	msgs []*disaggpb.KVBlockStream
	ctx  context.Context
}

func (r *recorderStream) Send(msg *disaggpb.KVBlockStream) error {
	r.msgs = append(r.msgs, msg)
	return nil
}

func (r *recorderStream) Context() context.Context { return r.ctx }

func TestPrefillWorker(t *testing.T) {
	const (
		numLayers = 4
		kvSize    = 8
	)

	srv := NewPrefillWorkerServer(&stubPrefiller{
		numLayers: numLayers,
		kvSize:    kvSize,
	})

	req := &disaggpb.PreFillRequest{
		RequestId:    "req-1",
		TokenIds:     []int32{1, 2, 3, 4, 5},
		MaxNewTokens: 32,
		Temperature:  0.7,
	}
	rec := &recorderStream{ctx: context.Background()}

	if err := srv.Prefill(req, rec); err != nil {
		t.Fatalf("Prefill returned error: %v", err)
	}

	// Expect numLayers data messages + 1 done sentinel.
	wantMsgs := numLayers + 1
	if got := len(rec.msgs); got != wantMsgs {
		t.Fatalf("expected %d messages, got %d", wantMsgs, got)
	}

	// Verify data messages.
	for i := range numLayers {
		msg := rec.msgs[i]
		if msg.Done {
			t.Errorf("message %d: unexpected Done=true", i)
		}
		blk := msg.Block
		if blk == nil {
			t.Fatalf("message %d: nil block", i)
		}
		if blk.RequestId != "req-1" {
			t.Errorf("message %d: request_id = %q, want %q", i, blk.RequestId, "req-1")
		}
		if blk.LayerIdx != int32(i) {
			t.Errorf("message %d: layer_idx = %d, want %d", i, blk.LayerIdx, i)
		}

		// Each K/V should be kvSize * 2 bytes (FP16).
		wantBytes := kvSize * 2
		if len(blk.KData) != wantBytes {
			t.Errorf("message %d: KData len = %d, want %d", i, len(blk.KData), wantBytes)
		}
		if len(blk.VData) != wantBytes {
			t.Errorf("message %d: VData len = %d, want %d", i, len(blk.VData), wantBytes)
		}
	}

	// Verify done sentinel.
	last := rec.msgs[numLayers]
	if !last.Done {
		t.Error("last message: expected Done=true")
	}
}

func TestPrefillWorker_EmptyTokens(t *testing.T) {
	srv := NewPrefillWorkerServer(&stubPrefiller{numLayers: 2, kvSize: 4})

	req := &disaggpb.PreFillRequest{
		RequestId: "req-empty",
		TokenIds:  nil,
	}
	rec := &recorderStream{ctx: context.Background()}

	if err := srv.Prefill(req, rec); err == nil {
		t.Fatal("expected error for empty tokens, got nil")
	}
}

func TestFloat32sToFP16Bytes(t *testing.T) {
	tests := []struct {
		name string
		in   float32
		want uint16 // expected FP16 bit pattern
	}{
		{"zero", 0.0, 0x0000},
		{"one", 1.0, 0x3c00},
		{"neg_one", -1.0, 0xbc00},
		{"half", 0.5, 0x3800},
		{"inf", float32(math.Inf(1)), 0x7c00},
		{"neg_inf", float32(math.Inf(-1)), 0xfc00},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := float32ToFP16Bits(tt.in)
			if got != tt.want {
				t.Errorf("float32ToFP16Bits(%v) = 0x%04x, want 0x%04x", tt.in, got, tt.want)
			}
		})
	}
}
