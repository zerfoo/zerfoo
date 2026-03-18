package serve

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zerfoo/zerfoo/inference/multimodal"
)

// testPNG creates a minimal 2x2 red PNG image as raw bytes.
func testPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	img.Set(1, 0, color.RGBA{R: 255, A: 255})
	img.Set(0, 1, color.RGBA{R: 255, A: 255})
	img.Set(1, 1, color.RGBA{R: 255, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestAPIVisionInput(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	pngData := testPNG(t)
	b64 := base64.StdEncoding.EncodeToString(pngData)

	body := `{
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "What is in this image?"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,` + b64 + `"}}
			]
		}],
		"max_tokens": 5
	}`

	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Object != "chat.completion" {
		t.Errorf("Object = %q, want %q", result.Object, "chat.completion")
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
	if result.Choices[0].Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", result.Choices[0].Message.Role, "assistant")
	}
}

func TestAPIVisionInput_HTTPImage(t *testing.T) {
	pngData := testPNG(t)

	imgServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Write(pngData)
	}))
	defer imgServer.Close()

	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "Describe this image"},
				{"type": "image_url", "image_url": {"url": "` + imgServer.URL + `/test.png"}}
			]
		}],
		"max_tokens": 5
	}`

	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
}

func TestAPIVisionInput_StringContent(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
}

func TestChatMessageUnmarshal_ContentArray(t *testing.T) {
	input := `{
		"role": "user",
		"content": [
			{"type": "text", "text": "hello"},
			{"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}}
		]
	}`

	var msg ChatMessage
	if err := json.Unmarshal([]byte(input), &msg); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if msg.Role != "user" {
		t.Errorf("Role = %q, want %q", msg.Role, "user")
	}
	if msg.Content != "hello" {
		t.Errorf("Content = %q, want %q", msg.Content, "hello")
	}
	if len(msg.ImageURLs) != 1 {
		t.Fatalf("ImageURLs len = %d, want 1", len(msg.ImageURLs))
	}
	if msg.ImageURLs[0].URL != "https://example.com/img.png" {
		t.Errorf("ImageURL = %q, want %q", msg.ImageURLs[0].URL, "https://example.com/img.png")
	}
	if msg.ImageURLs[0].Detail != "high" {
		t.Errorf("Detail = %q, want %q", msg.ImageURLs[0].Detail, "high")
	}
}

func TestChatMessageUnmarshal_StringContent(t *testing.T) {
	input := `{"role": "user", "content": "plain text"}`

	var msg ChatMessage
	if err := json.Unmarshal([]byte(input), &msg); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if msg.Content != "plain text" {
		t.Errorf("Content = %q, want %q", msg.Content, "plain text")
	}
	if len(msg.ImageURLs) != 0 {
		t.Errorf("ImageURLs len = %d, want 0", len(msg.ImageURLs))
	}
}

func TestDecodeDataURI(t *testing.T) {
	original := []byte("test image data")
	encoded := base64.StdEncoding.EncodeToString(original)
	uri := "data:image/png;base64," + encoded

	got, err := decodeDataURI(uri)
	if err != nil {
		t.Fatalf("decodeDataURI error: %v", err)
	}
	if !bytes.Equal(got, original) {
		t.Errorf("decoded data mismatch")
	}
}

func TestDecodeDataURI_InvalidFormat(t *testing.T) {
	_, err := decodeDataURI("data:image/png;base64")
	if err == nil {
		t.Error("expected error for missing comma separator")
	}
}

func TestDetectImageFormat(t *testing.T) {
	jpegMagic := []byte{0xFF, 0xD8, 0xFF, 0xE0}
	if got := detectImageFormat(jpegMagic); got != multimodal.JPEG {
		t.Errorf("JPEG detection = %d, want %d", got, multimodal.JPEG)
	}

	pngMagic := []byte{0x89, 'P', 'N', 'G'}
	if got := detectImageFormat(pngMagic); got != multimodal.PNG {
		t.Errorf("PNG detection = %d, want %d", got, multimodal.PNG)
	}
}

func TestChatMessageUnmarshal_MultipleImages(t *testing.T) {
	input := `{
		"role": "user",
		"content": [
			{"type": "text", "text": "compare"},
			{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
			{"type": "image_url", "image_url": {"url": "https://example.com/b.png"}}
		]
	}`

	var msg ChatMessage
	if err := json.Unmarshal([]byte(input), &msg); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if msg.Content != "compare" {
		t.Errorf("Content = %q, want %q", msg.Content, "compare")
	}
	if len(msg.ImageURLs) != 2 {
		t.Fatalf("ImageURLs len = %d, want 2", len(msg.ImageURLs))
	}
}

func TestChatMessageUnmarshal_EmptyString(t *testing.T) {
	input := `{"role": "system", "content": ""}`

	var msg ChatMessage
	if err := json.Unmarshal([]byte(input), &msg); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if msg.Role != "system" {
		t.Errorf("Role = %q, want %q", msg.Role, "system")
	}
	if msg.Content != "" {
		t.Errorf("Content = %q, want empty", msg.Content)
	}
}
