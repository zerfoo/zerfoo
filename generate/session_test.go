package generate

import (
	"context"
	"testing"
)

func TestNewSession(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  32000,
		MaxSeqLen:  2048,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  12,
	}
	gen := NewGenerator[float32](nil, nil, nil, cfg)

	session := gen.NewSession()
	if session == nil {
		t.Fatal("expected non-nil session")
	}
	if session.gen != gen {
		t.Error("session should reference the parent generator")
	}
	if session.pos != 0 {
		t.Errorf("initial position = %d, want 0", session.pos)
	}
	if session.cache == nil {
		t.Fatal("session cache should be initialized")
	}
	if session.Cache() == nil {
		t.Fatal("Cache() should return non-nil provider")
	}
}

func TestSessionPosition(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  32000,
		MaxSeqLen:  2048,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	gen := NewGenerator[float32](nil, nil, nil, cfg)
	session := gen.NewSession()

	if pos := session.Position(); pos != 0 {
		t.Errorf("Position() = %d, want 0", pos)
	}
}

func TestSessionReset(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  32000,
		MaxSeqLen:  2048,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	gen := NewGenerator[float32](nil, nil, nil, cfg)
	session := gen.NewSession()

	// Manually advance position to simulate usage.
	session.mu.Lock()
	session.pos = 42
	session.mu.Unlock()

	session.Reset()

	if pos := session.Position(); pos != 0 {
		t.Errorf("after Reset, Position() = %d, want 0", pos)
	}
}

func TestSessionGenerateStub(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  32000,
		MaxSeqLen:  2048,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	gen := NewGenerator[float32](nil, nil, nil, cfg)
	session := gen.NewSession()

	_, err := session.Generate(context.Background(), "hello", DefaultSamplingConfig())
	if err == nil {
		t.Fatal("expected stub error from Generate")
	}
}

func TestSessionGenerateStreamStub(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  32000,
		MaxSeqLen:  2048,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	gen := NewGenerator[float32](nil, nil, nil, cfg)
	session := gen.NewSession()

	stream := TokenStreamFunc(func(token string, done bool) error {
		return nil
	})
	err := session.GenerateStream(context.Background(), "hello", DefaultSamplingConfig(), stream)
	if err == nil {
		t.Fatal("expected stub error from GenerateStream")
	}
}

func TestSessionString(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  32000,
		MaxSeqLen:  2048,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	gen := NewGenerator[float32](nil, nil, nil, cfg)
	session := gen.NewSession()

	s := session.String()
	if s == "" {
		t.Fatal("String() should return non-empty description")
	}
}

func TestMultipleSessions(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  32000,
		MaxSeqLen:  2048,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	gen := NewGenerator[float32](nil, nil, nil, cfg)

	s1 := gen.NewSession()
	s2 := gen.NewSession()

	if s1 == s2 {
		t.Fatal("expected distinct session instances")
	}
	if s1.cache == s2.cache {
		t.Fatal("sessions should have independent caches")
	}

	// Mutating one session should not affect the other.
	s1.mu.Lock()
	s1.pos = 10
	s1.mu.Unlock()

	if s2.Position() != 0 {
		t.Error("s2 position should be unaffected by s1 mutation")
	}
}
