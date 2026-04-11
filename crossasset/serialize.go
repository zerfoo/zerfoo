package crossasset

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// File format:
//   Magic:   4 bytes "ZCAM"
//   Version: uint32 (little-endian), currently 1
//   CfgLen:  uint64 (little-endian), JSON config byte count
//   CfgData: CfgLen bytes of JSON-encoded Config
//   Weights: sequence of (uint64 count, count*8 bytes of float64 LE) blocks
//
// Weight order (deterministic):
//   inputW[0], inputW[1], ..., inputW[NSources-1]
//   inputB[0], inputB[1], ..., inputB[NSources-1]
//   For each layer 0..NLayers-1:
//     qW, kW, vW, outW, lnGamma, lnBeta, ffnW1, ffnB1, ffnW2, ffnB2, ffnGamma, ffnBeta
//   headW, headB

var magic = [4]byte{'Z', 'C', 'A', 'M'}

const formatVersion uint32 = 1

// Save serializes the trained model weights to the given path.
func (m *Model) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("crossasset.Save: %w", err)
	}
	defer f.Close()

	// Magic.
	if _, err := f.Write(magic[:]); err != nil {
		return fmt.Errorf("crossasset.Save: write magic: %w", err)
	}

	// Version.
	if err := binary.Write(f, binary.LittleEndian, formatVersion); err != nil {
		return fmt.Errorf("crossasset.Save: write version: %w", err)
	}

	// Config as JSON.
	cfgBytes, err := json.Marshal(m.config)
	if err != nil {
		return fmt.Errorf("crossasset.Save: marshal config: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(cfgBytes))); err != nil {
		return fmt.Errorf("crossasset.Save: write config length: %w", err)
	}
	if _, err := f.Write(cfgBytes); err != nil {
		return fmt.Errorf("crossasset.Save: write config: %w", err)
	}

	// Weights.
	// Input projections.
	for s := 0; s < m.config.NSources; s++ {
		if err := writeFloat64Slice(f, m.inputW[s]); err != nil {
			return fmt.Errorf("crossasset.Save: inputW[%d]: %w", s, err)
		}
	}
	for s := 0; s < m.config.NSources; s++ {
		if err := writeFloat64Slice(f, m.inputB[s]); err != nil {
			return fmt.Errorf("crossasset.Save: inputB[%d]: %w", s, err)
		}
	}

	// Transformer layers.
	for i, l := range m.layers {
		for _, w := range []struct {
			name string
			data []float64
		}{
			{"qW", l.qW}, {"kW", l.kW}, {"vW", l.vW}, {"outW", l.outW},
			{"lnGamma", l.lnGamma}, {"lnBeta", l.lnBeta},
			{"ffnW1", l.ffnW1}, {"ffnB1", l.ffnB1},
			{"ffnW2", l.ffnW2}, {"ffnB2", l.ffnB2},
			{"ffnGamma", l.ffnGamma}, {"ffnBeta", l.ffnBeta},
		} {
			if err := writeFloat64Slice(f, w.data); err != nil {
				return fmt.Errorf("crossasset.Save: layer[%d].%s: %w", i, w.name, err)
			}
		}
	}

	// Classification head.
	if err := writeFloat64Slice(f, m.headW); err != nil {
		return fmt.Errorf("crossasset.Save: headW: %w", err)
	}
	if err := writeFloat64Slice(f, m.headB); err != nil {
		return fmt.Errorf("crossasset.Save: headB: %w", err)
	}

	return nil
}

// LoadModel loads a previously saved crossasset model from the given path.
func LoadModel(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("crossasset.LoadModel: %w", err)
	}
	defer f.Close()

	// Magic.
	var fileMagic [4]byte
	if _, err := io.ReadFull(f, fileMagic[:]); err != nil {
		return nil, fmt.Errorf("crossasset.LoadModel: read magic: %w", err)
	}
	if fileMagic != magic {
		return nil, fmt.Errorf("crossasset.LoadModel: invalid magic %q, expected %q", fileMagic, magic)
	}

	// Version.
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("crossasset.LoadModel: read version: %w", err)
	}
	if version != formatVersion {
		return nil, fmt.Errorf("crossasset.LoadModel: unsupported version %d, expected %d", version, formatVersion)
	}

	// Config.
	var cfgLen uint64
	if err := binary.Read(f, binary.LittleEndian, &cfgLen); err != nil {
		return nil, fmt.Errorf("crossasset.LoadModel: read config length: %w", err)
	}
	if cfgLen > 1<<20 { // 1MB sanity limit
		return nil, fmt.Errorf("crossasset.LoadModel: config length %d exceeds 1MB limit", cfgLen)
	}
	cfgBytes := make([]byte, cfgLen)
	if _, err := io.ReadFull(f, cfgBytes); err != nil {
		return nil, fmt.Errorf("crossasset.LoadModel: read config: %w", err)
	}
	var config Config
	if err := json.Unmarshal(cfgBytes, &config); err != nil {
		return nil, fmt.Errorf("crossasset.LoadModel: unmarshal config: %w", err)
	}

	// Construct model with the loaded config (allocates weight slices).
	m := NewModel(config)

	// Read weights in the same order as Save.
	for s := 0; s < config.NSources; s++ {
		if err := readFloat64Slice(f, m.inputW[s]); err != nil {
			return nil, fmt.Errorf("crossasset.LoadModel: inputW[%d]: %w", s, err)
		}
	}
	for s := 0; s < config.NSources; s++ {
		if err := readFloat64Slice(f, m.inputB[s]); err != nil {
			return nil, fmt.Errorf("crossasset.LoadModel: inputB[%d]: %w", s, err)
		}
	}

	for i := range m.layers {
		l := &m.layers[i]
		for _, w := range []struct {
			name string
			data []float64
		}{
			{"qW", l.qW}, {"kW", l.kW}, {"vW", l.vW}, {"outW", l.outW},
			{"lnGamma", l.lnGamma}, {"lnBeta", l.lnBeta},
			{"ffnW1", l.ffnW1}, {"ffnB1", l.ffnB1},
			{"ffnW2", l.ffnW2}, {"ffnB2", l.ffnB2},
			{"ffnGamma", l.ffnGamma}, {"ffnBeta", l.ffnBeta},
		} {
			if err := readFloat64Slice(f, w.data); err != nil {
				return nil, fmt.Errorf("crossasset.LoadModel: layer[%d].%s: %w", i, w.name, err)
			}
		}
	}

	if err := readFloat64Slice(f, m.headW); err != nil {
		return nil, fmt.Errorf("crossasset.LoadModel: headW: %w", err)
	}
	if err := readFloat64Slice(f, m.headB); err != nil {
		return nil, fmt.Errorf("crossasset.LoadModel: headB: %w", err)
	}

	return m, nil
}

// writeFloat64Slice writes a length-prefixed float64 slice in little-endian.
func writeFloat64Slice(w io.Writer, data []float64) error {
	if err := binary.Write(w, binary.LittleEndian, uint64(len(data))); err != nil {
		return err
	}
	buf := make([]byte, 8*len(data))
	for i, v := range data {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	_, err := w.Write(buf)
	return err
}

// readFloat64Slice reads a length-prefixed float64 slice, writing into dst.
func readFloat64Slice(r io.Reader, dst []float64) error {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return fmt.Errorf("read count: %w", err)
	}
	if int(n) != len(dst) {
		return fmt.Errorf("size mismatch: file has %d elements, expected %d", n, len(dst))
	}
	buf := make([]byte, 8*len(dst))
	if _, err := io.ReadFull(r, buf); err != nil {
		return fmt.Errorf("read data: %w", err)
	}
	for i := range dst {
		dst[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:]))
	}
	return nil
}
