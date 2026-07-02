package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// validGGUFSeed builds a minimal valid GGUF v3 binary with one metadata
// key and one tensor info entry, suitable as a seed corpus for fuzzing.
func validGGUFSeed() []byte {
	var buf bytes.Buffer
	bw(&buf, Magic)          // magic
	bw(&buf, uint32(3))      // version
	bw(&buf, uint64(1))      // tensor count
	bw(&buf, uint64(1))      // metadata kv count

	// Metadata: "general.architecture" = "llama"
	writeTestString(&buf, "general.architecture")
	bw(&buf, TypeString)
	writeTestString(&buf, "llama")

	// Tensor: "blk.0.attn_q.weight", 2-D [128, 128], F32, offset 0
	writeTestString(&buf, "blk.0.attn_q.weight")
	bw(&buf, uint32(2))   // ndims
	bw(&buf, uint64(128)) // dim 0
	bw(&buf, uint64(128)) // dim 1
	bw(&buf, uint32(GGMLTypeF32))
	bw(&buf, uint64(0)) // offset

	return buf.Bytes()
}

// emptyGGUFSeed builds a minimal valid GGUF v3 binary with no metadata
// or tensors.
func emptyGGUFSeed() []byte {
	var buf bytes.Buffer
	bw(&buf, Magic)
	bw(&buf, uint32(3))
	bw(&buf, uint64(0))
	bw(&buf, uint64(0))
	return buf.Bytes()
}

func FuzzParse(f *testing.F) {
	// Seed 1: valid GGUF with metadata and tensor info.
	f.Add(validGGUFSeed())

	// Seed 2: empty GGUF (no metadata, no tensors).
	f.Add(emptyGGUFSeed())

	// Seed 3: invalid magic number.
	f.Add([]byte{0xDE, 0xAD, 0xBE, 0xEF})

	// Seed 4: truncated header (just magic, no version).
	magic := make([]byte, 4)
	binary.LittleEndian.PutUint32(magic, Magic)
	f.Add(magic)

	// Seed 5: unsupported version.
	var v1 bytes.Buffer
	bw(&v1, Magic)
	bw(&v1, uint32(1))
	f.Add(v1.Bytes())

	// Seed 6: empty input.
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, data []byte) {
		r := bytes.NewReader(data)
		file, err := Parse(r)
		if err != nil {
			return
		}
		// If parsing succeeds, basic invariants must hold.
		if file.Version < 2 || file.Version > 3 {
			t.Errorf("parsed version %d outside valid range [2,3]", file.Version)
		}
		if file.DataOffset < 0 {
			t.Errorf("negative DataOffset %d", file.DataOffset)
		}
		if file.DataOffset%32 != 0 {
			t.Errorf("DataOffset %d not 32-byte aligned", file.DataOffset)
		}
		if file.Metadata == nil {
			t.Error("Metadata map is nil on successful parse")
		}
		for _, ti := range file.Tensors {
			if ti.Name == "" {
				t.Error("parsed tensor with empty name")
			}
		}
	})
}
