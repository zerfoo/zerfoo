# Zerfoo Examples and Tutorials

This directory contains practical examples and tutorials for using Zerfoo.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Basic Tutorials](#basic-tutorials)
3. [Advanced Examples](#advanced-examples)
4. [Use Case Examples](#use-case-examples)

## Quick Start Examples

### Hello Zerfoo - Your First Model

```go
package main

import (
    "fmt"
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/activations"
    "github.com/zerfoo/zerfoo/layers/core"
    "github.com/zerfoo/zerfoo/tensor"
)

func main() {
    fmt.Println("Hello Zerfoo!")
    
    // Create a simple 2-layer neural network
    engine := compute.NewCPUEngine[float32]()
    builder := graph.NewBuilder[float32](engine)
    
    // Define the model
    input := builder.Input([]int{1, 2})
    hidden := builder.AddNode(core.NewDense[float32](2, 3, engine), input)
    activation := builder.AddNode(activations.NewReLU[float32](engine), hidden)
    output := builder.AddNode(core.NewDense[float32](3, 1, engine), activation)
    
    // Build the computational graph
    forward, _, err := builder.Build(output)
    if err != nil {
        panic(err)
    }
    
    // Create input data
    inputData, _ := tensor.New[float32]([]int{1, 2}, []float32{1.0, 2.0})
    
    // Run forward pass
    result := forward(map[graph.NodeHandle]*tensor.Tensor[float32]{
        input: inputData,
    })
    
    fmt.Printf("Input: %v\n", inputData.Data())
    fmt.Printf("Output: %v\n", result.Data())
}
```

### Basic Tensor Operations

```go
package main

import (
    "fmt"
    "github.com/zerfoo/zerfoo/tensor"
)

func main() {
    // Create tensors
    a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
    b, _ := tensor.New[float32]([]int{2, 3}, []float32{6, 5, 4, 3, 2, 1})
    
    fmt.Printf("Tensor A:\n%s\n", a.String())
    fmt.Printf("Tensor B:\n%s\n", b.String())
    
    // Element-wise operations
    sum, _ := tensor.Add(a, b)
    fmt.Printf("A + B:\n%s\n", sum.String())
    
    product, _ := tensor.Mul(a, b)
    fmt.Printf("A * B (element-wise):\n%s\n", product.String())
    
    // Matrix operations
    aT, _ := tensor.Transpose(a)
    fmt.Printf("A transposed:\n%s\n", aT.String())
    
    // Reshaping
    reshaped, _ := a.Reshape([]int{3, 2})
    fmt.Printf("A reshaped to 3x2:\n%s\n", reshaped.String())
}
```

## Basic Tutorials

### Tutorial 1: Building Your First Neural Network

Let's build a simple neural network for binary classification:

```go
package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"
    
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/activations"
    "github.com/zerfoo/zerfoo/layers/core"
    "github.com/zerfoo/zerfoo/tensor"
    "github.com/zerfoo/zerfoo/training/optimizer"
)

func main() {
    rand.Seed(time.Now().UnixNano())
    
    // Step 1: Create the compute engine
    engine := compute.NewCPUEngine[float32]()
    
    // Step 2: Build the model
    model, inputHandle, outputHandle := buildBinaryClassifier(engine)
    
    // Step 3: Generate training data
    trainX, trainY := generateBinaryData(1000)
    
    // Step 4: Train the model
    trainBinaryClassifier(model, inputHandle, outputHandle, trainX, trainY, engine)
    
    // Step 5: Evaluate the model
    testX, testY := generateBinaryData(200)
    accuracy := evaluateBinaryClassifier(model, inputHandle, outputHandle, testX, testY)
    fmt.Printf("Final accuracy: %.2f%%\n", accuracy*100)
}

func buildBinaryClassifier(engine compute.Engine[float32]) (*graph.Builder[float32], graph.NodeHandle, graph.NodeHandle) {
    builder := graph.NewBuilder[float32](engine)
    
    // Input: 2 features
    input := builder.Input([]int{1, 2})
    
    // Hidden layer: 2 -> 4
    hidden1 := builder.AddNode(core.NewDense[float32](2, 4, engine), input)
    relu1 := builder.AddNode(activations.NewReLU[float32](engine), hidden1)
    
    // Hidden layer: 4 -> 3
    hidden2 := builder.AddNode(core.NewDense[float32](4, 3, engine), relu1)
    relu2 := builder.AddNode(activations.NewReLU[float32](engine), hidden2)
    
    // Output layer: 3 -> 1
    output := builder.AddNode(core.NewDense[float32](3, 1, engine), relu2)
    sigmoid := builder.AddNode(activations.NewSigmoid[float32](engine), output)
    
    return builder, input, sigmoid
}

func generateBinaryData(n int) ([]*tensor.Tensor[float32], []*tensor.Tensor[float32]) {
    var inputs, labels []*tensor.Tensor[float32]
    
    for i := 0; i < n; i++ {
        // Generate random point
        x := rand.Float32()*4 - 2 // [-2, 2]
        y := rand.Float32()*4 - 2 // [-2, 2]
        
        // Simple decision boundary: x^2 + y^2 > 1
        label := float32(0)
        if x*x + y*y > 1 {
            label = 1
        }
        
        input, _ := tensor.New[float32]([]int{1, 2}, []float32{x, y})
        output, _ := tensor.New[float32]([]int{1, 1}, []float32{label})
        
        inputs = append(inputs, input)
        labels = append(labels, output)
    }
    
    return inputs, labels
}

func trainBinaryClassifier(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle, 
                          trainX, trainY []*tensor.Tensor[float32], engine compute.Engine[float32]) {
    
    // Build the graph
    forward, backward, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    // Create optimizer
    opt := optimizer.NewAdam[float32](0.01)
    
    // Training loop
    epochs := 100
    batchSize := 32
    
    for epoch := 0; epoch < epochs; epoch++ {
        totalLoss := float32(0)
        numBatches := 0
        
        // Mini-batch training
        for i := 0; i < len(trainX); i += batchSize {
            end := i + batchSize
            if end > len(trainX) {
                end = len(trainX)
            }
            
            batchLoss := float32(0)
            
            // Process batch
            for j := i; j < end; j++ {
                // Forward pass
                inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
                    inputHandle: trainX[j],
                }
                prediction := forward(inputMap)
                
                // Compute binary cross-entropy loss
                target := trainY[j].Data()[0]
                pred := prediction.Data()[0]
                
                // Clamp prediction to avoid log(0)
                pred = float32(math.Max(float64(pred), 1e-7))
                pred = float32(math.Min(float64(pred), 1.0-1e-7))
                
                loss := -target*float32(math.Log(float64(pred))) - (1-target)*float32(math.Log(float64(1-pred)))
                batchLoss += loss
                
                // Compute gradient
                grad := pred - target
                gradTensor, _ := tensor.New[float32]([]int{1, 1}, []float32{grad})
                
                // Backward pass
                backward(gradTensor, inputMap)
            }
            
            // Update parameters
            opt.Step(builder.Parameters())
            totalLoss += batchLoss
            numBatches++
        }
        
        if epoch%10 == 0 {
            avgLoss := totalLoss / float32(len(trainX))
            fmt.Printf("Epoch %d, Average Loss: %.4f\n", epoch, avgLoss)
        }
    }
}

func evaluateBinaryClassifier(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                             testX, testY []*tensor.Tensor[float32]) float32 {
    
    forward, _, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    correct := 0
    for i := range testX {
        inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
            inputHandle: testX[i],
        }
        prediction := forward(inputMap)
        
        predicted := float32(0)
        if prediction.Data()[0] > 0.5 {
            predicted = 1
        }
        
        if predicted == testY[i].Data()[0] {
            correct++
        }
    }
    
    return float32(correct) / float32(len(testX))
}
```

### Tutorial 2: Custom Layers

Learn how to create custom layers:

```go
package main

import (
    "fmt"
    "math"
    
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/tensor"
)

// CustomSquareLayer squares its input
type CustomSquareLayer[T tensor.Numeric] struct {
    engine     compute.Engine[T]
    inputShape []int
}

func NewCustomSquareLayer[T tensor.Numeric](inputShape []int, engine compute.Engine[T]) *CustomSquareLayer[T] {
    return &CustomSquareLayer[T]{
        engine:     engine,
        inputShape: inputShape,
    }
}

func (l *CustomSquareLayer[T]) OutputShape() []int {
    return l.inputShape
}

func (l *CustomSquareLayer[T]) Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
    if len(inputs) != 1 {
        return nil, graph.ErrInvalidInputCount
    }
    
    input := inputs[0]
    output, err := tensor.New[T](input.Shape(), nil)
    if err != nil {
        return nil, err
    }
    
    // Square each element
    inputData := input.Data()
    outputData := output.Data()
    for i := range inputData {
        outputData[i] = inputData[i] * inputData[i]
    }
    
    return output, nil
}

func (l *CustomSquareLayer[T]) Backward(outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
    // For y = x^2, dy/dx = 2x
    // So gradient w.r.t input = outputGradient * 2 * input
    
    // This is a simplified implementation
    // In practice, you'd need to store the input from the forward pass
    inputGrad, err := tensor.New[T](outputGradient.Shape(), nil)
    if err != nil {
        return nil, err
    }
    
    // For demonstration, assume gradient = 2 * outputGradient
    outputData := outputGradient.Data()
    inputData := inputGrad.Data()
    for i := range outputData {
        inputData[i] = 2 * outputData[i] // Simplified
    }
    
    return []*tensor.Tensor[T]{inputGrad}, nil
}

func (l *CustomSquareLayer[T]) Parameters() []*graph.Parameter[T] {
    return nil // No trainable parameters
}

func main() {
    engine := compute.NewCPUEngine[float32]()
    builder := graph.NewBuilder[float32](engine)
    
    // Use custom layer in a model
    input := builder.Input([]int{1, 3})
    square := builder.AddNode(NewCustomSquareLayer[float32]([]int{1, 3}, engine), input)
    
    // Build the graph
    forward, _, err := builder.Build(square)
    if err != nil {
        panic(err)
    }
    
    // Test the custom layer
    inputData, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 2, 3})
    result := forward(map[graph.NodeHandle]*tensor.Tensor[float32]{
        input: inputData,
    })
    
    fmt.Printf("Input: %v\n", inputData.Data())   // [1, 2, 3]
    fmt.Printf("Output: %v\n", result.Data())     // [1, 4, 9]
}
```

### Tutorial 3: Training with Different Optimizers

Compare different optimization algorithms:

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
    
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/activations"
    "github.com/zerfoo/zerfoo/layers/core"
    "github.com/zerfoo/zerfoo/tensor"
    "github.com/zerfoo/zerfoo/training/optimizer"
)

func main() {
    rand.Seed(time.Now().UnixNano())
    
    engine := compute.NewCPUEngine[float32]()
    
    // Generate dataset
    trainX, trainY := generateRegressionData(500)
    testX, testY := generateRegressionData(100)
    
    // Test different optimizers
    optimizers := []struct {
        name string
        opt  optimizer.Optimizer[float32]
    }{
        {"SGD", optimizer.NewSGD[float32](0.01)},
        {"SGD with Momentum", optimizer.NewSGDWithMomentum[float32](0.01, 0.9)},
        {"Adam", optimizer.NewAdam[float32](0.01)},
    }
    
    for _, optConfig := range optimizers {
        fmt.Printf("\\nTesting %s:\\n", optConfig.name)
        model, inputHandle, outputHandle := buildRegressionModel(engine)
        finalLoss := trainRegressionModel(model, inputHandle, outputHandle, trainX, trainY, optConfig.opt)
        testLoss := evaluateRegressionModel(model, inputHandle, outputHandle, testX, testY)
        
        fmt.Printf("Final train loss: %.4f\\n", finalLoss)
        fmt.Printf("Test loss: %.4f\\n", testLoss)
    }
}

func buildRegressionModel(engine compute.Engine[float32]) (*graph.Builder[float32], graph.NodeHandle, graph.NodeHandle) {
    builder := graph.NewBuilder[float32](engine)
    
    input := builder.Input([]int{1, 1})
    hidden1 := builder.AddNode(core.NewDense[float32](1, 10, engine), input)
    relu1 := builder.AddNode(activations.NewReLU[float32](engine), hidden1)
    hidden2 := builder.AddNode(core.NewDense[float32](10, 5, engine), relu1)
    relu2 := builder.AddNode(activations.NewReLU[float32](engine), hidden2)
    output := builder.AddNode(core.NewDense[float32](5, 1, engine), relu2)
    
    return builder, input, output
}

func generateRegressionData(n int) ([]*tensor.Tensor[float32], []*tensor.Tensor[float32]) {
    var inputs, targets []*tensor.Tensor[float32]
    
    for i := 0; i < n; i++ {
        x := rand.Float32()*4 - 2 // [-2, 2]
        y := x*x + 0.5*x + 0.1*rand.Float32() // Quadratic with noise
        
        input, _ := tensor.New[float32]([]int{1, 1}, []float32{x})
        target, _ := tensor.New[float32]([]int{1, 1}, []float32{y})
        
        inputs = append(inputs, input)
        targets = append(targets, target)
    }
    
    return inputs, targets
}

func trainRegressionModel(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                         trainX, trainY []*tensor.Tensor[float32], opt optimizer.Optimizer[float32]) float32 {
    
    forward, backward, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    epochs := 100
    var finalLoss float32
    
    for epoch := 0; epoch < epochs; epoch++ {
        totalLoss := float32(0)
        
        for i := range trainX {
            // Forward pass
            inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
                inputHandle: trainX[i],
            }
            prediction := forward(inputMap)
            
            // MSE loss
            diff := prediction.Data()[0] - trainY[i].Data()[0]
            loss := diff * diff
            totalLoss += loss
            
            // Gradient: d(MSE)/d(pred) = 2 * (pred - target)
            grad := 2 * diff
            gradTensor, _ := tensor.New[float32]([]int{1, 1}, []float32{grad})
            
            // Backward pass
            backward(gradTensor, inputMap)
            
            // Update parameters
            opt.Step(builder.Parameters())
        }
        
        avgLoss := totalLoss / float32(len(trainX))
        finalLoss = avgLoss
        
        if epoch%20 == 0 {
            fmt.Printf("Epoch %d, Loss: %.4f\\n", epoch, avgLoss)
        }
    }
    
    return finalLoss
}

func evaluateRegressionModel(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                           testX, testY []*tensor.Tensor[float32]) float32 {
    
    forward, _, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    totalLoss := float32(0)
    for i := range testX {
        inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
            inputHandle: testX[i],
        }
        prediction := forward(inputMap)
        
        diff := prediction.Data()[0] - testY[i].Data()[0]
        loss := diff * diff
        totalLoss += loss
    }
    
    return totalLoss / float32(len(testX))
}
```

## Advanced Examples

### Multi-Layer Perceptron with Dropout

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
    
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/activations"
    "github.com/zerfoo/zerfoo/layers/core"
    "github.com/zerfoo/zerfoo/tensor"
    "github.com/zerfoo/zerfoo/training/optimizer"
)

func main() {
    rand.Seed(time.Now().UnixNano())
    
    engine := compute.NewCPUEngine[float32]()
    
    // Build model with dropout
    model, inputHandle, outputHandle := buildMLPWithDropout(engine)
    
    // Generate multi-class classification data
    trainX, trainY := generateMultiClassData(1000, 4, 3) // 4 features, 3 classes
    testX, testY := generateMultiClassData(200, 4, 3)
    
    // Train the model
    trainMultiClassModel(model, inputHandle, outputHandle, trainX, trainY)
    
    // Evaluate
    accuracy := evaluateMultiClassModel(model, inputHandle, outputHandle, testX, testY)
    fmt.Printf("Test accuracy: %.2f%%\\n", accuracy*100)
}

func buildMLPWithDropout(engine compute.Engine[float32]) (*graph.Builder[float32], graph.NodeHandle, graph.NodeHandle) {
    builder := graph.NewBuilder[float32](engine)
    
    input := builder.Input([]int{1, 4})
    
    // First hidden layer with dropout
    hidden1 := builder.AddNode(core.NewDense[float32](4, 64, engine), input)
    relu1 := builder.AddNode(activations.NewReLU[float32](engine), hidden1)
    dropout1 := builder.AddNode(core.NewDropout[float32](0.3), relu1) // 30% dropout
    
    // Second hidden layer with dropout
    hidden2 := builder.AddNode(core.NewDense[float32](64, 32, engine), dropout1)
    relu2 := builder.AddNode(activations.NewReLU[float32](engine), hidden2)
    dropout2 := builder.AddNode(core.NewDropout[float32](0.3), relu2)
    
    // Third hidden layer
    hidden3 := builder.AddNode(core.NewDense[float32](32, 16, engine), dropout2)
    relu3 := builder.AddNode(activations.NewReLU[float32](engine), hidden3)
    
    // Output layer (3 classes)
    output := builder.AddNode(core.NewDense[float32](16, 3, engine), relu3)
    softmax := builder.AddNode(activations.NewSoftmax[float32](engine), output)
    
    return builder, input, softmax
}

func generateMultiClassData(n, features, classes int) ([]*tensor.Tensor[float32], []*tensor.Tensor[float32]) {
    var inputs, labels []*tensor.Tensor[float32]
    
    for i := 0; i < n; i++ {
        // Generate random features
        data := make([]float32, features)
        for j := range data {
            data[j] = rand.Float32()*2 - 1
        }
        
        // Simple decision boundaries
        var class int
        if data[0] > 0 && data[1] > 0 {
            class = 0
        } else if data[0] < 0 && data[1] > 0 {
            class = 1
        } else {
            class = 2
        }
        
        // One-hot encoding
        labelData := make([]float32, classes)
        labelData[class] = 1
        
        input, _ := tensor.New[float32]([]int{1, features}, data)
        label, _ := tensor.New[float32]([]int{1, classes}, labelData)
        
        inputs = append(inputs, input)
        labels = append(labels, label)
    }
    
    return inputs, labels
}

func trainMultiClassModel(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                         trainX, trainY []*tensor.Tensor[float32]) {
    
    forward, backward, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    opt := optimizer.NewAdam[float32](0.001)
    epochs := 150
    
    for epoch := 0; epoch < epochs; epoch++ {
        totalLoss := float32(0)
        
        for i := range trainX {
            inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
                inputHandle: trainX[i],
            }
            prediction := forward(inputMap)
            
            // Cross-entropy loss
            predData := prediction.Data()
            targetData := trainY[i].Data()
            
            loss := float32(0)
            gradData := make([]float32, len(predData))
            
            for j := range predData {
                if predData[j] > 0 {
                    loss -= targetData[j] * float32(math.Log(float64(predData[j])))
                }
                gradData[j] = predData[j] - targetData[j]
            }
            
            totalLoss += loss
            
            gradTensor, _ := tensor.New[float32](prediction.Shape(), gradData)
            backward(gradTensor, inputMap)
            opt.Step(builder.Parameters())
        }
        
        if epoch%25 == 0 {
            avgLoss := totalLoss / float32(len(trainX))
            fmt.Printf("Epoch %d, Loss: %.4f\\n", epoch, avgLoss)
        }
    }
}

func evaluateMultiClassModel(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                           testX, testY []*tensor.Tensor[float32]) float32 {
    
    forward, _, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    correct := 0
    for i := range testX {
        inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
            inputHandle: testX[i],
        }
        prediction := forward(inputMap)
        
        // Find predicted class
        predData := prediction.Data()
        predClass := 0
        maxProb := predData[0]
        for j := 1; j < len(predData); j++ {
            if predData[j] > maxProb {
                maxProb = predData[j]
                predClass = j
            }
        }
        
        // Find true class
        targetData := testY[i].Data()
        trueClass := 0
        for j := range targetData {
            if targetData[j] == 1 {
                trueClass = j
                break
            }
        }
        
        if predClass == trueClass {
            correct++
        }
    }
    
    return float32(correct) / float32(len(testX))
}
```

## Use Case Examples

### Time Series Forecasting

```go
package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"
    
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/core"
    "github.com/zerfoo/zerfoo/tensor"
    "github.com/zerfoo/zerfoo/training/optimizer"
)

func main() {
    rand.Seed(time.Now().UnixNano())
    
    engine := compute.NewCPUEngine[float32]()
    
    // Generate time series data
    sequenceLength := 10
    trainX, trainY := generateTimeSeriesData(1000, sequenceLength)
    testX, testY := generateTimeSeriesData(200, sequenceLength)
    
    // Build and train model
    model, inputHandle, outputHandle := buildTimeSeriesModel(engine, sequenceLength)
    trainTimeSeriesModel(model, inputHandle, outputHandle, trainX, trainY)
    
    // Evaluate
    mae := evaluateTimeSeriesModel(model, inputHandle, outputHandle, testX, testY)
    fmt.Printf("Mean Absolute Error: %.4f\\n", mae)
}

func generateTimeSeriesData(n, seqLen int) ([]*tensor.Tensor[float32], []*tensor.Tensor[float32]) {
    var inputs, targets []*tensor.Tensor[float32]
    
    for i := 0; i < n; i++ {
        // Generate sine wave with noise
        sequence := make([]float32, seqLen)
        for j := range sequence {
            t := float64(i*seqLen + j) * 0.1
            sequence[j] = float32(math.Sin(t) + 0.1*rand.Float64())
        }
        
        // Target is the next value in the sequence
        t := float64(i*seqLen + seqLen) * 0.1
        target := float32(math.Sin(t) + 0.1*rand.Float64())
        
        input, _ := tensor.New[float32]([]int{1, seqLen}, sequence)
        targetTensor, _ := tensor.New[float32]([]int{1, 1}, []float32{target})
        
        inputs = append(inputs, input)
        targets = append(targets, targetTensor)
    }
    
    return inputs, targets
}

func buildTimeSeriesModel(engine compute.Engine[float32], seqLen int) (*graph.Builder[float32], graph.NodeHandle, graph.NodeHandle) {
    builder := graph.NewBuilder[float32](engine)
    
    input := builder.Input([]int{1, seqLen})
    
    // Simple feedforward approach (RNN would be better but not implemented yet)
    hidden1 := builder.AddNode(core.NewDense[float32](seqLen, 64, engine), input)
    hidden2 := builder.AddNode(core.NewDense[float32](64, 32, engine), hidden1)
    hidden3 := builder.AddNode(core.NewDense[float32](32, 16, engine), hidden2)
    output := builder.AddNode(core.NewDense[float32](16, 1, engine), hidden3)
    
    return builder, input, output
}

func trainTimeSeriesModel(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                         trainX, trainY []*tensor.Tensor[float32]) {
    
    forward, backward, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    opt := optimizer.NewAdam[float32](0.001)
    epochs := 100
    
    for epoch := 0; epoch < epochs; epoch++ {
        totalLoss := float32(0)
        
        for i := range trainX {
            inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
                inputHandle: trainX[i],
            }
            prediction := forward(inputMap)
            
            // MSE loss
            diff := prediction.Data()[0] - trainY[i].Data()[0]
            loss := diff * diff
            totalLoss += loss
            
            grad := 2 * diff
            gradTensor, _ := tensor.New[float32]([]int{1, 1}, []float32{grad})
            
            backward(gradTensor, inputMap)
            opt.Step(builder.Parameters())
        }
        
        if epoch%20 == 0 {
            avgLoss := totalLoss / float32(len(trainX))
            fmt.Printf("Epoch %d, MSE: %.6f\\n", epoch, avgLoss)
        }
    }
}

func evaluateTimeSeriesModel(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                           testX, testY []*tensor.Tensor[float32]) float32 {
    
    forward, _, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    totalError := float32(0)
    for i := range testX {
        inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
            inputHandle: testX[i],
        }
        prediction := forward(inputMap)
        
        // Mean Absolute Error
        diff := prediction.Data()[0] - testY[i].Data()[0]
        if diff < 0 {
            diff = -diff
        }
        totalError += diff
    }
    
    return totalError / float32(len(testX))
}
```

### Model Ensemble

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
    
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/activations"
    "github.com/zerfoo/zerfoo/layers/core"
    "github.com/zerfoo/zerfoo/tensor"
    "github.com/zerfoo/zerfoo/training/optimizer"
)

type ModelEnsemble[T tensor.Numeric] struct {
    models      []*graph.Builder[T]
    inputHandles []graph.NodeHandle
    outputHandles []graph.NodeHandle
    forwards    []graph.ForwardFunc[T]
}

func NewModelEnsemble[T tensor.Numeric]() *ModelEnsemble[T] {
    return &ModelEnsemble[T]{}
}

func (e *ModelEnsemble[T]) AddModel(builder *graph.Builder[T], inputHandle, outputHandle graph.NodeHandle) error {
    forward, _, err := builder.Build(outputHandle)
    if err != nil {
        return err
    }
    
    e.models = append(e.models, builder)
    e.inputHandles = append(e.inputHandles, inputHandle)
    e.outputHandles = append(e.outputHandles, outputHandle)
    e.forwards = append(e.forwards, forward)
    
    return nil
}

func (e *ModelEnsemble[T]) Predict(input *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
    if len(e.models) == 0 {
        return nil, fmt.Errorf("no models in ensemble")
    }
    
    // Get predictions from all models
    predictions := make([]*tensor.Tensor[T], len(e.models))
    for i, forward := range e.forwards {
        inputMap := map[graph.NodeHandle]*tensor.Tensor[T]{
            e.inputHandles[i]: input,
        }
        predictions[i] = forward(inputMap)
    }
    
    // Average the predictions
    outputShape := predictions[0].Shape()
    avgData := make([]T, len(predictions[0].Data()))
    
    for _, pred := range predictions {
        predData := pred.Data()
        for j, val := range predData {
            avgData[j] += val
        }
    }
    
    // Divide by number of models
    numModels := T(len(e.models))
    for j := range avgData {
        avgData[j] /= numModels
    }
    
    return tensor.New[T](outputShape, avgData)
}

func main() {
    rand.Seed(time.Now().UnixNano())
    
    engine := compute.NewCPUEngine[float32]()
    
    // Generate data
    trainX, trainY := generateBinaryData(800)
    testX, testY := generateBinaryData(200)
    
    // Create ensemble
    ensemble := NewModelEnsemble[float32]()
    
    // Train multiple models with different architectures
    architectures := []struct {
        name   string
        layers []int
    }{
        {"Small", []int{2, 8, 4, 1}},
        {"Medium", []int{2, 16, 8, 1}},
        {"Wide", []int{2, 32, 16, 1}},
    }
    
    for _, arch := range architectures {
        fmt.Printf("Training %s model...\\n", arch.name)
        
        model, inputHandle, outputHandle := buildCustomModel(engine, arch.layers)
        trainBinaryModel(model, inputHandle, outputHandle, trainX, trainY)
        
        err := ensemble.AddModel(model, inputHandle, outputHandle)
        if err != nil {
            panic(err)
        }
    }
    
    // Evaluate individual models and ensemble
    fmt.Printf("\\nEvaluation Results:\\n")
    
    for i, arch := range architectures {
        accuracy := evaluateIndividualModel(ensemble.models[i], ensemble.inputHandles[i], 
                                          ensemble.outputHandles[i], testX, testY)
        fmt.Printf("%s model accuracy: %.2f%%\\n", arch.name, accuracy*100)
    }
    
    // Evaluate ensemble
    ensembleAccuracy := evaluateEnsemble(ensemble, testX, testY)
    fmt.Printf("Ensemble accuracy: %.2f%%\\n", ensembleAccuracy*100)
}

func buildCustomModel(engine compute.Engine[float32], layers []int) (*graph.Builder[float32], graph.NodeHandle, graph.NodeHandle) {
    builder := graph.NewBuilder[float32](engine)
    
    input := builder.Input([]int{1, layers[0]})
    currentNode := input
    
    for i := 1; i < len(layers)-1; i++ {
        dense := builder.AddNode(core.NewDense[float32](layers[i-1], layers[i], engine), currentNode)
        relu := builder.AddNode(activations.NewReLU[float32](engine), dense)
        currentNode = relu
    }
    
    // Output layer
    output := builder.AddNode(core.NewDense[float32](layers[len(layers)-2], layers[len(layers)-1], engine), currentNode)
    sigmoid := builder.AddNode(activations.NewSigmoid[float32](engine), output)
    
    return builder, input, sigmoid
}

func trainBinaryModel(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                     trainX, trainY []*tensor.Tensor[float32]) {
    
    forward, backward, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    opt := optimizer.NewAdam[float32](0.01)
    epochs := 50 // Reduced for ensemble training
    
    for epoch := 0; epoch < epochs; epoch++ {
        for i := range trainX {
            inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
                inputHandle: trainX[i],
            }
            prediction := forward(inputMap)
            
            // Binary cross-entropy
            target := trainY[i].Data()[0]
            pred := prediction.Data()[0]
            
            grad := pred - target
            gradTensor, _ := tensor.New[float32]([]int{1, 1}, []float32{grad})
            
            backward(gradTensor, inputMap)
            opt.Step(builder.Parameters())
        }
    }
}

func evaluateIndividualModel(builder *graph.Builder[float32], inputHandle, outputHandle graph.NodeHandle,
                           testX, testY []*tensor.Tensor[float32]) float32 {
    
    forward, _, err := builder.Build(outputHandle)
    if err != nil {
        panic(err)
    }
    
    correct := 0
    for i := range testX {
        inputMap := map[graph.NodeHandle]*tensor.Tensor[float32]{
            inputHandle: testX[i],
        }
        prediction := forward(inputMap)
        
        predicted := float32(0)
        if prediction.Data()[0] > 0.5 {
            predicted = 1
        }
        
        if predicted == testY[i].Data()[0] {
            correct++
        }
    }
    
    return float32(correct) / float32(len(testX))
}

func evaluateEnsemble(ensemble *ModelEnsemble[float32], testX, testY []*tensor.Tensor[float32]) float32 {
    correct := 0
    
    for i := range testX {
        prediction, err := ensemble.Predict(testX[i])
        if err != nil {
            panic(err)
        }
        
        predicted := float32(0)
        if prediction.Data()[0] > 0.5 {
            predicted = 1
        }
        
        if predicted == testY[i].Data()[0] {
            correct++
        }
    }
    
    return float32(correct) / float32(len(testX))
}

// Reuse generateBinaryData from earlier examples
func generateBinaryData(n int) ([]*tensor.Tensor[float32], []*tensor.Tensor[float32]) {
    var inputs, labels []*tensor.Tensor[float32]
    
    for i := 0; i < n; i++ {
        x := rand.Float32()*4 - 2
        y := rand.Float32()*4 - 2
        
        label := float32(0)
        if x*x + y*y > 1 {
            label = 1
        }
        
        input, _ := tensor.New[float32]([]int{1, 2}, []float32{x, y})
        output, _ := tensor.New[float32]([]int{1, 1}, []float32{label})
        
        inputs = append(inputs, input)
        labels = append(labels, output)
    }
    
    return inputs, labels
}
```

## Running the Examples

To run any of these examples:

1. Create a new Go file (e.g., `example.go`)
2. Copy the example code
3. Run with: `go run example.go`

Make sure you have Zerfoo installed:
```bash
go get github.com/zerfoo/zerfoo
```

## Next Steps

- Explore the [API Reference](API_REFERENCE.md) for detailed documentation
- Check out the [Contributing Guide](CONTRIBUTING.md) to contribute your own examples
- Visit the [examples directory](../examples/) for more advanced use cases
- Read the [Architecture Design](design.md) to understand the framework internals

Happy learning with Zerfoo! 🚀