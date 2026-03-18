package tabular

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

var (
	magic      = [4]byte{'Z', 'T', 'A', 'B'}
	fileVersion uint32 = 1
)

// Save writes a Model to the given path in the ZTAB binary format.
//
// Format:
//   - 4-byte magic ("ZTAB")
//   - 4-byte version (uint32 little-endian, currently 1)
//   - 4-byte config length (uint32 little-endian)
//   - JSON-encoded ModelConfig
//   - Weight data: for each hidden layer then the output head,
//     weights tensor data followed by biases tensor data as raw
//     float32 little-endian bytes.
func Save(model *Model, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("tabular: save: %w", err)
	}
	defer f.Close()

	// Magic.
	if _, err := f.Write(magic[:]); err != nil {
		return fmt.Errorf("tabular: save: write magic: %w", err)
	}

	// Version.
	if err := binary.Write(f, binary.LittleEndian, fileVersion); err != nil {
		return fmt.Errorf("tabular: save: write version: %w", err)
	}

	// Config as JSON.
	configBytes, err := json.Marshal(model.config)
	if err != nil {
		return fmt.Errorf("tabular: save: marshal config: %w", err)
	}
	configLen := uint32(len(configBytes))
	if err := binary.Write(f, binary.LittleEndian, configLen); err != nil {
		return fmt.Errorf("tabular: save: write config length: %w", err)
	}
	if _, err := f.Write(configBytes); err != nil {
		return fmt.Errorf("tabular: save: write config: %w", err)
	}

	// Write layer weights: hidden layers then output head.
	allLayers := append(model.layers, model.head)
	for i, l := range allLayers {
		if err := writeFloat32Slice(f, l.weights.Data()); err != nil {
			return fmt.Errorf("tabular: save: layer %d weights: %w", i, err)
		}
		if err := writeFloat32Slice(f, l.biases.Data()); err != nil {
			return fmt.Errorf("tabular: save: layer %d biases: %w", i, err)
		}
	}

	return nil
}

// writeFloat32Slice writes a []float32 as raw little-endian bytes.
func writeFloat32Slice(f *os.File, data []float32) error {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	_, err := f.Write(buf)
	return err
}

// Load reads a Model from the given path in the ZTAB binary format.
func Load(path string, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("tabular: load: %w", err)
	}
	defer f.Close()

	// Magic.
	var m [4]byte
	if _, err := f.Read(m[:]); err != nil {
		return nil, fmt.Errorf("tabular: load: read magic: %w", err)
	}
	if m != magic {
		return nil, fmt.Errorf("tabular: load: invalid magic %q, expected %q", m[:], magic[:])
	}

	// Version.
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("tabular: load: read version: %w", err)
	}
	if version != fileVersion {
		return nil, fmt.Errorf("tabular: load: unsupported version %d, expected %d", version, fileVersion)
	}

	// Config.
	var configLen uint32
	if err := binary.Read(f, binary.LittleEndian, &configLen); err != nil {
		return nil, fmt.Errorf("tabular: load: read config length: %w", err)
	}
	configBytes := make([]byte, configLen)
	if _, err := f.Read(configBytes); err != nil {
		return nil, fmt.Errorf("tabular: load: read config: %w", err)
	}
	var config ModelConfig
	if err := json.Unmarshal(configBytes, &config); err != nil {
		return nil, fmt.Errorf("tabular: load: unmarshal config: %w", err)
	}

	// Build layer structure from config to know dimensions.
	dims := append([]int{config.InputDim}, config.HiddenDims...)
	layers := make([]mlpLayer, len(config.HiddenDims))
	for i := 0; i < len(config.HiddenDims); i++ {
		inDim := dims[i]
		outDim := dims[i+1]
		l, err := readMLPLayer(f, inDim, outDim)
		if err != nil {
			return nil, fmt.Errorf("tabular: load: layer %d: %w", i, err)
		}
		layers[i] = l
	}

	// Output head: last hidden dim -> 3 classes.
	lastHidden := config.HiddenDims[len(config.HiddenDims)-1]
	head, err := readMLPLayer(f, lastHidden, 3)
	if err != nil {
		return nil, fmt.Errorf("tabular: load: output head: %w", err)
	}

	return &Model{
		config: config,
		engine: engine,
		ops:    ops,
		layers: layers,
		head:   head,
	}, nil
}

// readMLPLayer reads weights and biases from a file for a layer with the given dimensions.
func readMLPLayer(f *os.File, inDim, outDim int) (mlpLayer, error) {
	wData, err := readFloat32Slice(f, inDim*outDim)
	if err != nil {
		return mlpLayer{}, fmt.Errorf("read weights: %w", err)
	}
	w, err := tensor.New[float32]([]int{inDim, outDim}, wData)
	if err != nil {
		return mlpLayer{}, fmt.Errorf("create weights tensor: %w", err)
	}

	bData, err := readFloat32Slice(f, outDim)
	if err != nil {
		return mlpLayer{}, fmt.Errorf("read biases: %w", err)
	}
	b, err := tensor.New[float32]([]int{1, outDim}, bData)
	if err != nil {
		return mlpLayer{}, fmt.Errorf("create biases tensor: %w", err)
	}

	return mlpLayer{weights: w, biases: b}, nil
}

// readFloat32Slice reads n float32 values from raw little-endian bytes.
func readFloat32Slice(f *os.File, n int) ([]float32, error) {
	buf := make([]byte, n*4)
	if _, err := f.Read(buf); err != nil {
		return nil, err
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return data, nil
}
