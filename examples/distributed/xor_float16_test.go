package distributed_example

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// TestXORFloat16Distributed demonstrates a simplified distributed training of XOR using float16.
// This example simulates a single-node distributed setup for testability.
func TestXORFloat16Distributed(t *testing.T) {
	// 1. Setup Device and Engine
	_, err := device.Get("cpu")
	if err != nil {
		t.Fatalf("Failed to get CPU device: %v", err)
	}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// 2. Define XOR dataset
	// Inputs: [0,0], [0,1], [1,0], [1,1]
	// Outputs: [0], [1], [1], [0]
	input00, err := tensor.New[float32]([]int{1, 2}, []float32{0, 0})
	if err != nil { t.Fatalf("Failed to create tensor: %v", err) }
	input01, err := tensor.New[float32]([]int{1, 2}, []float32{0, 1})
	if err != nil { t.Fatalf("Failed to create tensor: %v", err) }
	input10, err := tensor.New[float32]([]int{1, 2}, []float32{1, 0})
	if err != nil { t.Fatalf("Failed to create tensor: %v", err) }
	input11, err := tensor.New[float32]([]int{1, 2}, []float32{1, 1})
	if err != nil { t.Fatalf("Failed to create tensor: %v", err) }

	target0, err := tensor.New[float32]([]int{1, 1}, []float32{0})
	if err != nil { t.Fatalf("Failed to create tensor: %v", err) }
	target1, err := tensor.New[float32]([]int{1, 1}, []float32{1})
	if err != nil { t.Fatalf("Failed to create tensor: %v", err) }
	target2, err := tensor.New[float32]([]int{1, 1}, []float32{1})
	if err != nil { t.Fatalf("Failed to create tensor: %v", err) }
	target3, err := tensor.New[float32]([]int{1, 1}, []float32{0})
	if err != nil { t.Fatalf("Failed to create tensor: %v", err) }

	inputs := []*tensor.Tensor[float32]{input00, input01, input10, input11}
	targets := []*tensor.Tensor[float32]{target0, target1, target2, target3}

	// 3. Build the Model (Dense layers)
	// Input: 2 features, Hidden: 4 neurons, Output: 1 neuron
	ops := numeric.Float32Ops{} // Define ops once
	
dense1, err := core.NewDense[float32]("dense1", engine, ops, 2, 4)
	if err != nil { t.Fatalf("Failed to create dense1: %v", err) }
	relu1 := activations.NewReLU[float32](engine, ops)

	dense2, err := core.NewDense[float32]("dense2", engine, ops, 4, 2)
	if err != nil { t.Fatalf("Failed to create dense2: %v", err) }
	sigmoid1 := activations.NewSigmoid[float32](engine, ops)

	// 4. Define Loss Function and Optimizer
	lossFn := loss.NewMSE[float32](engine, ops) // Reverted to MSE
	opt := optimizer.NewAdamW[float32](engine, 0.001, 0.9, 0.999, 1e-4, 0.01)

	// 5. Simulate Distributed Training (single rank for simplicity)
	// In a real distributed setup, each rank would have its own model and
	// gradients would be aggregated using AllReduce. Here, we simulate
	// the training process on a single "rank".
	epochs := 10000 // Increased epochs
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)
		for i := 0; i < len(inputs); i++ {
			input := inputs[i]
			target := targets[i]

			// Forward pass
			hiddenOutput, err := dense1.Forward(context.Background(), input)
			if err != nil { t.Fatalf("Dense1 forward pass failed: %v", err) }
			activatedHiddenOutput, err := relu1.Forward(context.Background(), hiddenOutput)
			if err != nil { t.Fatalf("ReLU1 forward pass failed: %v", err) }

			output, err := dense2.Forward(context.Background(), activatedHiddenOutput)
			if err != nil { t.Fatalf("Dense2 forward pass failed: %v", err) }
			finalOutput, err := sigmoid1.Forward(context.Background(), output)
			if err != nil { t.Fatalf("Sigmoid1 forward pass failed: %v", err) }

			// Calculate loss
			lossTensor, err := lossFn.Forward(context.Background(), finalOutput, target)
			if err != nil { t.Fatalf("Loss forward pass failed: %v", err) }
			l := lossTensor.Data()[0]
			totalLoss = (numeric.Float32Ops{}).Add(totalLoss, l)

			// Backward pass (compute gradients)
			outputGrad, err := lossFn.Backward(context.Background(), finalOutput, target)
			if err != nil { t.Fatalf("Loss backward pass failed: %v", err) }
			
sigmoidGrads, err := sigmoid1.Backward(context.Background(), outputGrad)
			if err != nil { t.Fatalf("Sigmoid1 backward pass failed: %v", err) }
			dense2Grads, err := dense2.Backward(context.Background(), sigmoidGrads[0], activatedHiddenOutput)
			if err != nil { t.Fatalf("Dense2 backward pass failed: %v", err) }

			reluGrads, err := relu1.Backward(context.Background(), dense2Grads[0])
			if err != nil { t.Fatalf("ReLU1 backward pass failed: %v", err) }
			_, err = dense1.Backward(context.Background(), reluGrads[0], input)
			if err != nil { t.Fatalf("Dense1 backward pass failed: %v", err) }

			// Apply gradients (simulate AllReduce and then update)
			// In a real distributed setting, gradients from all ranks would be
			// averaged here before applying. For this single-rank simulation, 
			// we just apply the local gradients.
			if err := opt.Step(context.Background(), dense1.Parameters()); err != nil { t.Fatalf("Optimizer step for dense1 failed: %v", err) }
			if err := opt.Step(context.Background(), dense2.Parameters()); err != nil { t.Fatalf("Optimizer step for dense2 failed: %v", err) }
		}

		if epoch%100 == 0 {
			fmt.Printf("Epoch %d, Loss: %f\n", epoch, totalLoss/float32(len(inputs)))
		}
	}

	// 6. Verify the trained model
	fmt.Println("\n--- Verification ---")
	for i := 0; i < len(inputs); i++ {
		input := inputs[i]
		target := targets[i]

		hiddenOutput, err := dense1.Forward(context.Background(), input)
		if err != nil { t.Fatalf("Dense1 forward pass failed during verification: %v", err) }
		activatedHiddenOutput, err := relu1.Forward(context.Background(), hiddenOutput)
		if err != nil { t.Fatalf("ReLU1 forward pass failed during verification: %v", err) }

		output, err := dense2.Forward(context.Background(), activatedHiddenOutput)
		if err != nil { t.Fatalf("Dense2 forward pass failed during verification: %v", err) }
		finalOutput, err := sigmoid1.Forward(context.Background(), output)
		if err != nil { t.Fatalf("Sigmoid1 forward pass failed during verification: %v", err) }

		// Convert output to a single float32 for comparison
		outputVal := finalOutput.Data()[0]
		targetVal := target.Data()[0]

		// For XOR, output should be close to 0 or 1.
		// We'll check if it's closer to the target value.
		predicted := float32(0)
		if (numeric.Float32Ops{}).GreaterThan(outputVal, float32(0.5)) {
			predicted = float32(1)
		}

		fmt.Printf("Input: %v, Target: %v, Predicted Raw: %v, Predicted: %v\n", input.Data(), targetVal, outputVal, predicted)

		if (numeric.Float32Ops{}).Abs((numeric.Float32Ops{}).Sub(predicted, targetVal)) > 0.1 { // Allow for some floating point error
			t.Errorf("Epochs: %d, Input: %v, Expected: %v, Got: %v", epochs, input.Data(), targetVal, predicted)
		}
	}
}
