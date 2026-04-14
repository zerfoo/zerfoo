package inference

import "testing"

// gemma4E2BLayerTypes returns the Gemma 4 E2B attention pattern:
// 35 layers, every 5th is global (full) attention at indices 4,9,14,19,24,29,34.
// All other layers are sliding-window.
func gemma4E2BLayerTypes() []LayerType {
	types := make([]LayerType, 35)
	for i := range types {
		if (i+1)%5 == 0 {
			types[i] = LayerTypeGlobal
		} else {
			types[i] = LayerTypeSliding
		}
	}
	return types
}

func TestResolveKVDonor(t *testing.T) {
	t.Parallel()
	gemma := gemma4E2BLayerTypes()
	const firstShared = 15

	tests := []struct {
		name           string
		layerIdx       int
		firstSharedIdx int
		layerTypes     []LayerType
		want           int
	}{
		{
			// Shared sliding layer 20: walk back past 19(G), 18..15(shared),
			// 14(G), to 13(sliding, <15) -> donor=13.
			name:           "gemma4_e2b_layer20_sliding_donor13",
			layerIdx:       20,
			firstSharedIdx: firstShared,
			layerTypes:     gemma,
			want:           13,
		},
		{
			// Shared global layer 24: walk back past 23..15(shared), to
			// 14(global, <15) -> donor=14.
			name:           "gemma4_e2b_layer24_global_donor14",
			layerIdx:       24,
			firstSharedIdx: firstShared,
			layerTypes:     gemma,
			want:           14,
		},
		{
			// Shared sliding layer 25: walk back past 24(G), 23..15(shared),
			// 14(G), to 13(sliding, <15) -> donor=13.
			name:           "gemma4_e2b_layer25_sliding_donor13",
			layerIdx:       25,
			firstSharedIdx: firstShared,
			layerTypes:     gemma,
			want:           13,
		},
		{
			// Shared global layer 34: walk back past 33..15 (all shared,
			// rejected by j<firstSharedIdx), to 14(global, <15) -> donor=14.
			// Note: 29 is global but 29>=firstSharedIdx=15 so it is a shared
			// layer itself and cannot be a donor.
			name:           "gemma4_e2b_layer34_global_donor14",
			layerIdx:       34,
			firstSharedIdx: firstShared,
			layerTypes:     gemma,
			want:           14,
		},
		{
			// Boundary: firstSharedIdx == layerIdx, pick immediately
			// preceding same-type layer.
			name:           "gemma4_e2b_layer15_sliding_donor13",
			layerIdx:       15,
			firstSharedIdx: firstShared,
			layerTypes:     gemma,
			want:           13,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ResolveKVDonor(tc.layerIdx, tc.firstSharedIdx, tc.layerTypes)
			if got != tc.want {
				t.Fatalf("ResolveKVDonor(%d,%d,...) = %d, want %d",
					tc.layerIdx, tc.firstSharedIdx, got, tc.want)
			}
		})
	}
}

func TestResolveKVDonor_PanicLayerIdxLessThanFirstShared(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when layerIdx < firstSharedIdx")
		}
	}()
	types := []LayerType{LayerTypeSliding, LayerTypeSliding, LayerTypeGlobal}
	_ = ResolveKVDonor(1, 2, types)
}

func TestResolveKVDonor_PanicEmptyLayerTypes(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on empty layerTypes")
		}
	}()
	_ = ResolveKVDonor(0, 0, []LayerType{})
}

func TestResolveKVDonor_PanicNoDonorFound(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when no donor exists")
		}
	}()
	// All layers before firstSharedIdx=2 are sliding; shared layer at
	// index 2 is global => no global donor exists.
	types := []LayerType{LayerTypeSliding, LayerTypeSliding, LayerTypeGlobal}
	_ = ResolveKVDonor(2, 2, types)
}

func TestResolveKVDonor_PanicFirstSharedIdxNegative(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on negative firstSharedIdx")
		}
	}()
	types := []LayerType{LayerTypeSliding}
	_ = ResolveKVDonor(0, -1, types)
}

func TestResolveKVDonor_PanicFirstSharedIdxTooLarge(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when firstSharedIdx exceeds len(layerTypes)")
		}
	}()
	types := []LayerType{LayerTypeSliding, LayerTypeGlobal}
	_ = ResolveKVDonor(1, 3, types)
}

func TestResolveKVDonor_PanicLayerIdxOutOfRange(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when layerIdx >= len(layerTypes)")
		}
	}()
	types := []LayerType{LayerTypeSliding, LayerTypeGlobal}
	_ = ResolveKVDonor(2, 2, types)
}
