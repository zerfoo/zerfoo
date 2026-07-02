# Chat Example

An interactive chatbot CLI that demonstrates the zerfoo one-line API. Loads a model, reads prompts from stdin, and prints responses in a loop.

## Build

```bash
go build -o chat ./examples/chat/
```

## Usage

```bash
# With a local GGUF file
./chat --model path/to/model.gguf

# With a HuggingFace model ID (downloads automatically)
./chat --model google/gemma-3-1b-it
```

Once running, type a message and press Enter to get a response. Type `quit` or press Ctrl-D to exit.

```
Chat started. Type your message and press Enter. Type 'quit' to exit.
> What is the capital of France?
The capital of France is Paris.
> quit
```

## Code

The core loop is minimal — load the model, then call `m.Chat(prompt)` in a readline loop:

```go
m, err := zerfoo.Load("path/to/model.gguf")
if err != nil {
    log.Fatal(err)
}
defer m.Close()

response, err := m.Chat("What is the capital of France?")
fmt.Println(response)
```
