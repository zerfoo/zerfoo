// Command timeseries demonstrates time-series forecasting with the N-BEATS
// model using the zerfoo timeseries package.
//
// This example creates an N-BEATS model with trend and seasonality stacks,
// runs a forward pass on synthetic data, and displays the forecast output.
// No GPU or external data required.
//
// Usage:
//
//	go build -o ts-forecast ./examples/timeseries/
//	./ts-forecast
package main

import (
	"context"
	"fmt"
	"math"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/timeseries"
)

func main() {
	fmt.Println("=== N-BEATS Time-Series Forecasting Example ===")

	const (
		inputLen  = 24 // lookback window (e.g., 24 hours)
		outputLen = 6  // forecast horizon (e.g., 6 hours ahead)
		batchSize = 2  // number of sequences in the batch
		hiddenDim = 32
		nHarmonics = 4
	)

	// --- Step 1: Configure the N-BEATS model ---
	// N-BEATS decomposes the forecast into interpretable components:
	// - Trend: captures long-term direction using polynomial basis
	// - Seasonality: captures repeating patterns using Fourier basis
	config := timeseries.NBEATSConfig{
		InputLength:     inputLen,
		OutputLength:    outputLen,
		StackTypes:      []timeseries.StackType{timeseries.StackTrend, timeseries.StackSeasonality},
		NBlocksPerStack: 2,
		HiddenDim:       hiddenDim,
		NHarmonics:      nHarmonics,
	}

	fmt.Println("Model configuration:")
	fmt.Printf("  Input length:      %d\n", config.InputLength)
	fmt.Printf("  Output length:     %d\n", config.OutputLength)
	fmt.Printf("  Stacks:            trend + seasonality\n")
	fmt.Printf("  Blocks per stack:  %d\n", config.NBlocksPerStack)
	fmt.Printf("  Hidden dim:        %d\n", config.HiddenDim)
	fmt.Printf("  Fourier harmonics: %d\n", config.NHarmonics)

	// --- Step 2: Create the compute engine and model ---
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	model, err := timeseries.NewNBEATS(config, engine, ops)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating N-BEATS model: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("\nN-BEATS model created successfully.")

	// --- Step 3: Generate synthetic input data ---
	// Simulate a time series with trend + seasonal pattern:
	//   y(t) = 0.5*t + 10*sin(2*pi*t/12) + noise
	fmt.Printf("\nGenerating synthetic data: batch_size=%d, seq_len=%d\n", batchSize, inputLen)

	inputData := make([]float32, batchSize*inputLen)
	for b := 0; b < batchSize; b++ {
		offset := float64(b) * 5.0 // different offset per batch element
		for t := 0; t < inputLen; t++ {
			ft := float64(t)
			value := 0.5*ft + offset + 10.0*math.Sin(2.0*math.Pi*ft/12.0)
			inputData[b*inputLen+t] = float32(value)
		}
	}

	input, err := tensor.New[float32]([]int{batchSize, inputLen}, inputData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating input tensor: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Input (first batch element):")
	printSlice(inputData[:inputLen], "  ")

	// --- Step 4: Run the forward pass ---
	ctx := context.Background()
	output, err := model.Forward(ctx, input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error in forward pass: %v\n", err)
		os.Exit(1)
	}

	// --- Step 5: Display the forecast ---
	fmt.Printf("\n--- Forecast (horizon=%d) ---\n", outputLen)
	forecast := output.Forecast.Data()
	for b := 0; b < batchSize; b++ {
		start := b * outputLen
		end := start + outputLen
		fmt.Printf("Batch %d forecast:\n", b)
		printSlice(forecast[start:end], "  ")
	}

	// Display per-stack decomposition.
	fmt.Println("\n--- Stack Decomposition ---")
	stackNames := []string{"trend", "seasonality"}
	for i, sf := range output.StackForecasts {
		name := "unknown"
		if i < len(stackNames) {
			name = stackNames[i]
		}
		data := sf.Data()
		fmt.Printf("%s stack (batch 0):\n", name)
		printSlice(data[:outputLen], "  ")
	}

	fmt.Println("\n=== Done ===")
}

// printSlice prints a float32 slice with formatting.
func printSlice(data []float32, prefix string) {
	fmt.Printf("%s[", prefix)
	for i, v := range data {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.4f", v)
	}
	fmt.Println("]")
}
