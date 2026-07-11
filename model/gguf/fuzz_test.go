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

// f1AttackSeed builds a raw GGUF binary carrying the exact deep-review 002
// finding F1 attack shape: a 3-D tensor whose first two dimensions multiply
// to exactly 1<<34 (the element-count cap) and whose third dimension then
// overflows int64 to a negative product if multiplied without a
// check-before-multiply guard (see overflow_test.go's attackDimensions and
// computeNumElements). Seeding the raw bytes (rather than only unit-testing
// computeNumElements directly) exercises the full Parse -> LoadTensors path
// the fuzzer walks.
func f1AttackSeed() []byte {
	var buf bytes.Buffer
	bw(&buf, Magic)
	bw(&buf, uint32(3))      // version
	bw(&buf, uint64(1))      // tensor count
	bw(&buf, uint64(0))      // metadata kv count
	writeTestString(&buf, "attack.f1_overflow_dims")
	bw(&buf, uint32(3)) // ndims
	bw(&buf, uint64(131072))
	bw(&buf, uint64(131072))
	bw(&buf, uint64(2147483647))
	bw(&buf, uint32(GGMLTypeF32))
	bw(&buf, uint64(0)) // offset
	return buf.Bytes()
}

// f2AttackSeed builds a raw GGUF binary carrying the deep-review 002 finding
// F2 attack shape: a tensor offset of 0x8000000000000000, which wraps to a
// negative int64 after the uint64->int64 conversion performed at every load
// call site (see loader_mmap_test.go's TestLoadTensorsMmap_HugeOffsetSignedConversion).
func f2AttackSeed() []byte {
	var buf bytes.Buffer
	bw(&buf, Magic)
	bw(&buf, uint32(3)) // version
	bw(&buf, uint64(1)) // tensor count
	bw(&buf, uint64(0)) // metadata kv count
	writeTestString(&buf, "attack.f2_huge_offset")
	bw(&buf, uint32(1)) // ndims
	bw(&buf, uint64(4))
	bw(&buf, uint32(GGMLTypeF32))
	bw(&buf, uint64(0x8000000000000000)) // offset: wraps to negative int64
	return buf.Bytes()
}

// f3AttackSeed builds a raw GGUF binary carrying the deep-review 002 finding
// F3 attack shape: a tensor declaring numDims near uint32 max with no
// dimension data following it, which -- without the maxTensorDims cap --
// would force `make([]uint64, numDims)` to attempt a multi-gigabyte
// allocation (see TestParse_TensorDimsHugeCount).
func f3AttackSeed() []byte {
	var buf bytes.Buffer
	bw(&buf, Magic)
	bw(&buf, uint32(3)) // version
	bw(&buf, uint64(1)) // tensor count
	bw(&buf, uint64(0)) // metadata kv count
	writeTestString(&buf, "attack.f3_huge_ndims")
	bw(&buf, uint32(0xFFFFFFFF)) // numDims: ~4 billion, no dimension data follows
	return buf.Bytes()
}

// FuzzParse feeds arbitrary bytes into the GGUF Parse + LoadTensors path and
// asserts the property that must hold for every input, valid or malformed:
// the pipeline either returns an error or succeeds -- it must never panic.
// This is the general-purpose backstop for the whole class of malformed-GGUF
// bugs that deep-review 002 findings F1 (element-count overflow), F2 (offset
// signed-conversion), and F3 (unbounded dimension count) belong to; the
// targeted unit tests in overflow_test.go, loader_mmap_test.go, and
// parser_test.go cover those three shapes exactly, while this fuzzer
// explores the surrounding input space the fixes must also hold for.
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

	// Seed 7: deep-review 002 finding F1 (element-count overflow dims).
	f.Add(f1AttackSeed())

	// Seed 8: deep-review 002 finding F2 (huge uint64 offset).
	f.Add(f2AttackSeed())

	// Seed 9: deep-review 002 finding F3 (>8 dimension count).
	f.Add(f3AttackSeed())

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
		// Note: a zero-length tensor name is a well-formed GGUF string
		// (readString accepts length 0) and is not itself a parser defect,
		// so it is intentionally not asserted against here -- this fuzz
		// target's property is "malformed input yields an error, never a
		// panic," not "every successfully-parsed field is semantically
		// meaningful."

		// Continue down the load path: a successfully-parsed header can
		// still carry tensor descriptors engineered to overflow allocation
		// math or seek arithmetic (F1/F2/F3 and their variants). LoadTensors
		// must reject those with an error -- never a panic -- regardless of
		// whether the declared tensor data actually follows in `data`.
		func() {
			defer func() {
				if rec := recover(); rec != nil {
					t.Fatalf("LoadTensors panicked instead of returning an error: %v", rec)
				}
			}()
			_, _ = LoadTensors(file, bytes.NewReader(data))
		}()
	})
}
