package tensorrt

import "testing"

// TestPuregoHandleLifecycle verifies the purego wrapper can create a logger,
// builder, network, config, and runtime, then tear them down without error.
func TestPuregoHandleLifecycle(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}

	logger := CreateLogger(SeverityWarning)
	if logger == nil {
		t.Fatal("CreateLogger returned nil")
	}
	defer logger.Destroy()

	builder, err := CreateBuilder(logger)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer builder.Destroy()

	network, err := builder.CreateNetwork()
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	defer network.Destroy()

	config, err := builder.CreateBuilderConfig()
	if err != nil {
		t.Fatalf("CreateBuilderConfig: %v", err)
	}
	defer config.Destroy()

	config.SetMemoryPoolLimit(1 << 20)

	runtime, err := CreateRuntime(logger)
	if err != nil {
		t.Fatalf("CreateRuntime: %v", err)
	}
	defer runtime.Destroy()

	// Verify network starts empty
	if n := network.NumInputs(); n != 0 {
		t.Errorf("NumInputs = %d, want 0", n)
	}
	if n := network.NumOutputs(); n != 0 {
		t.Errorf("NumOutputs = %d, want 0", n)
	}
	if n := network.NumLayers(); n != 0 {
		t.Errorf("NumLayers = %d, want 0", n)
	}
}

// TestPuregoNetworkBuild verifies the purego wrapper can construct a simple
// network (input -> activation -> output) and build a serialized engine.
func TestPuregoNetworkBuild(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}

	tests := []struct {
		name string
		act  ActivationType
	}{
		{"ReLU", ActivationReLU},
		{"Sigmoid", ActivationSigmoid},
		{"Tanh", ActivationTanh},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger := CreateLogger(SeverityWarning)
			if logger == nil {
				t.Fatal("CreateLogger returned nil")
			}
			defer logger.Destroy()

			builder, err := CreateBuilder(logger)
			if err != nil {
				t.Fatalf("CreateBuilder: %v", err)
			}
			defer builder.Destroy()

			network, err := builder.CreateNetwork()
			if err != nil {
				t.Fatalf("CreateNetwork: %v", err)
			}
			defer network.Destroy()

			input := network.AddInput("input", Float32, []int32{1, 4})
			if input == nil {
				t.Fatal("AddInput returned nil")
			}

			act := network.AddActivation(input, tc.act)
			if act == nil {
				t.Fatal("AddActivation returned nil")
			}

			out := act.GetOutput(0)
			if out == nil {
				t.Fatal("GetOutput returned nil")
			}
			network.MarkOutput(out)

			if network.NumInputs() != 1 {
				t.Errorf("NumInputs = %d, want 1", network.NumInputs())
			}
			if network.NumOutputs() != 1 {
				t.Errorf("NumOutputs = %d, want 1", network.NumOutputs())
			}
			if network.NumLayers() != 1 {
				t.Errorf("NumLayers = %d, want 1", network.NumLayers())
			}

			config, err := builder.CreateBuilderConfig()
			if err != nil {
				t.Fatalf("CreateBuilderConfig: %v", err)
			}
			defer config.Destroy()
			config.SetMemoryPoolLimit(1 << 20)

			serialized, err := builder.BuildSerializedNetwork(network, config)
			if err != nil {
				t.Fatalf("BuildSerializedNetwork: %v", err)
			}
			if len(serialized) == 0 {
				t.Fatal("serialized engine is empty")
			}

			// Verify engine can be deserialized
			runtime, err := CreateRuntime(logger)
			if err != nil {
				t.Fatalf("CreateRuntime: %v", err)
			}
			defer runtime.Destroy()

			engine, err := runtime.DeserializeEngine(serialized)
			if err != nil {
				t.Fatalf("DeserializeEngine: %v", err)
			}
			defer engine.Destroy()

			if engine.NumIOTensors() != 2 {
				t.Errorf("NumIOTensors = %d, want 2", engine.NumIOTensors())
			}
		})
	}
}

// TestPuregoOptimizationProfile verifies creating and adding an optimization
// profile to a builder config.
func TestPuregoOptimizationProfile(t *testing.T) {
	if !Available() {
		t.Skip("TensorRT not available")
	}

	logger := CreateLogger(SeverityWarning)
	if logger == nil {
		t.Fatal("CreateLogger returned nil")
	}
	defer logger.Destroy()

	builder, err := CreateBuilder(logger)
	if err != nil {
		t.Fatalf("CreateBuilder: %v", err)
	}
	defer builder.Destroy()

	profile, err := builder.CreateOptimizationProfile()
	if err != nil {
		t.Fatalf("CreateOptimizationProfile: %v", err)
	}

	err = profile.SetDimensions("input",
		[]int32{1, 4},  // min
		[]int32{4, 4},  // opt
		[]int32{16, 4}, // max
	)
	if err != nil {
		t.Fatalf("SetDimensions: %v", err)
	}

	config, err := builder.CreateBuilderConfig()
	if err != nil {
		t.Fatalf("CreateBuilderConfig: %v", err)
	}
	defer config.Destroy()

	idx, err := profile.AddToConfig(config)
	if err != nil {
		t.Fatalf("AddToConfig: %v", err)
	}
	if idx < 0 {
		t.Errorf("profile index = %d, want >= 0", idx)
	}
}
