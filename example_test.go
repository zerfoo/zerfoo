package zerfoo_test

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo"
)

func ExampleModel_Chat() {
	m := zerfoo.NewModel(func(_ context.Context, _ string) (string, error) {
		return "Go interfaces define a set of method signatures. Any type that implements all methods satisfies the interface.", nil
	})
	response, err := m.Chat("Explain Go interfaces in one sentence.")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(response)
	// Output: Go interfaces define a set of method signatures. Any type that implements all methods satisfies the interface.
}

func ExampleModel_Generate() {
	m := zerfoo.NewModel(func(_ context.Context, _ string) (string, error) {
		return "The theory of relativity describes how space and time are linked for objects moving at consistent speeds in a straight line.", nil
	})
	result, err := m.Generate(context.Background(), "Explain relativity.",
		zerfoo.WithGenMaxTokens(50),
		zerfoo.WithGenTemperature(0),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(result.Text)
	// Output: The theory of relativity describes how space and time are linked for objects moving at consistent speeds in a straight line.
}

func ExampleModel_ChatStream() {
	m := zerfoo.NewModel(func(_ context.Context, _ string) (string, error) {
		return "Once upon a time there was a little cat.", nil
	})
	ch, err := m.ChatStream(context.Background(), "Tell me a story.")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	var text string
	for tok := range ch {
		if !tok.Done {
			text += tok.Text
		}
	}
	fmt.Println(text)
	// Output: Once upon a time there was a little cat.
}

func ExampleEmbedding_CosineSimilarity() {
	a := zerfoo.Embedding{Vector: []float32{1, 0, 0}}
	b := zerfoo.Embedding{Vector: []float32{0.7071, 0.7071, 0}}
	sim := a.CosineSimilarity(b)
	fmt.Printf("%.4f\n", sim)
	// Output: 0.7071
}

func ExampleNewModel() {
	m := zerfoo.NewModel(func(_ context.Context, prompt string) (string, error) {
		return fmt.Sprintf("Echo: %s", prompt), nil
	})
	result, _ := m.Generate(context.Background(), "hello")
	fmt.Println(result.Text)
	// Output: Echo: hello
}
