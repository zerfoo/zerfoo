// Package sentiment provides a high-level sentiment classification pipeline
// that wraps encoder model loading and inference. It supports pluggable
// tokenization, batch processing, and both softmax and continuous scoring modes.
//
// Basic usage with pre-tokenized input:
//
//	p, err := sentiment.New("model.gguf",
//	    sentiment.WithLabels([]string{"negative", "positive"}),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer p.Close()
//
//	results, err := p.ClassifyTokenized(ctx, [][]int{{101, 2023, 2003, 2307, 102}})
package sentiment
