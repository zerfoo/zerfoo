package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// mockMarketProvider implements MarketDataProvider with canned responses
// and records which methods were called.
type mockMarketProvider struct {
	mu      sync.Mutex
	called  map[string]int
	symbols map[string]MarketQuote
}

func newMockMarketProvider() *mockMarketProvider {
	return &mockMarketProvider{
		called: make(map[string]int),
		symbols: map[string]MarketQuote{
			"BTCUSD": {
				Symbol:    "BTCUSD",
				Price:     67500.50,
				Bid:       67490.00,
				Ask:       67510.00,
				Volume:    12345,
				Timestamp: "2026-03-17T10:00:00Z",
			},
		},
	}
}

func (m *mockMarketProvider) record(method string) {
	m.mu.Lock()
	m.called[method]++
	m.mu.Unlock()
}

func (m *mockMarketProvider) callCount(method string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.called[method]
}

func (m *mockMarketProvider) GetMarketData(symbol string) (MarketQuote, error) {
	m.record("GetMarketData")
	q, ok := m.symbols[symbol]
	if !ok {
		return MarketQuote{}, fmt.Errorf("symbol %q not found", symbol)
	}
	return q, nil
}

func (m *mockMarketProvider) GetOrderBook(symbol string, depth int) (OrderBook, error) {
	m.record("GetOrderBook")
	return OrderBook{
		Symbol: symbol,
		Bids: []OrderBookEntry{
			{Price: 67490.00, Quantity: 1.5},
			{Price: 67480.00, Quantity: 2.0},
		},
		Asks: []OrderBookEntry{
			{Price: 67510.00, Quantity: 1.2},
			{Price: 67520.00, Quantity: 0.8},
		},
	}, nil
}

func (m *mockMarketProvider) GetPortfolio(accountID string) (Portfolio, error) {
	m.record("GetPortfolio")
	return Portfolio{
		AccountID: accountID,
		Cash:      100000.00,
		Positions: []PortfolioPosition{
			{Symbol: "BTCUSD", Quantity: 0.5, AvgCost: 65000.00, Current: 67500.50},
		},
	}, nil
}

func (m *mockMarketProvider) GetEarningsCalendar(startDate, endDate string) ([]EarningsEvent, error) {
	m.record("GetEarningsCalendar")
	return []EarningsEvent{}, nil
}

func (m *mockMarketProvider) SearchNews(query string, limit int) ([]NewsArticle, error) {
	m.record("SearchNews")
	return []NewsArticle{
		{Title: "BTC rallies on ETF inflows", Source: "CryptoNews", Summary: "Bitcoin rises 3% on institutional demand"},
	}, nil
}

// mockRiskApprover always approves orders.
type mockRiskApprover struct{}

func (m *mockRiskApprover) Approve(symbol, side string, quantity, price float64) (bool, string, error) {
	return true, "", nil
}

// TestAgentIntegration verifies that a Supervisor can execute a multi-step
// research plan using market tools without hanging or looping.
func TestAgentIntegration(t *testing.T) {
	// Set up mock provider and registry.
	provider := newMockMarketProvider()
	risk := &mockRiskApprover{}
	toolSet := NewMarketToolSet(provider, risk)
	registry := NewToolRegistry()
	if err := toolSet.RegisterAll(registry); err != nil {
		t.Fatalf("RegisterAll: %v", err)
	}

	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{
		MaxSteps: 10,
	}, registry, parser)

	// Simulate a 5-step research plan. Each call to generateFn returns the
	// next scripted model output. The sequence is:
	//   1. GetMarketData for BTCUSD
	//   2. GetPortfolio for the account
	//   3. GetOrderBook for BTCUSD
	//   4. SearchNews for BTCUSD
	//   5. Final reasoning output (no tool call) with recommendation
	steps := []string{
		`Let me research BTCUSD. First, let me get the current market data.
{"name": "GetMarketData", "arguments": {"symbol": "BTCUSD"}}`,

		`Good, I can see the price. Let me check the current portfolio.
{"name": "GetPortfolio", "arguments": {"account_id": "main"}}`,

		`I have the portfolio. Now let me check the order book for liquidity.
{"name": "GetOrderBook", "arguments": {"symbol": "BTCUSD"}}`,

		`Let me also check recent news.
{"name": "SearchNews", "arguments": {"query": "BTCUSD", "limit": 5}}`,

		`Based on my analysis:
- BTCUSD is trading at $67,500.50 with a tight spread ($67,490-$67,510)
- Current position: 0.5 BTC at avg cost $65,000 (in profit)
- Order book shows balanced liquidity
- News sentiment is positive (ETF inflows)

Recommendation: Add 0.1 BTC at market (~$67,510). This increases the position
to 0.6 BTC while keeping total exposure under 50% of the portfolio cash value.
Risk is moderate given positive momentum and tight spreads.`,
	}

	stepIdx := 0
	var mu sync.Mutex
	generateFn := func(ctx context.Context, history []string) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		if stepIdx >= len(steps) {
			t.Fatal("generateFn called more times than expected")
		}
		out := steps[stepIdx]
		stepIdx++
		return out, nil
	}

	// Use a timeout to ensure we don't hang.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	session, err := sup.RunLoop(ctx, generateFn, "Research BTCUSD and recommend position size")
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}

	// Verify session completed normally.
	if !session.Finished {
		t.Fatal("session did not finish")
	}
	if session.StopReason != "no_tools" {
		t.Fatalf("expected stop reason %q, got %q", "no_tools", session.StopReason)
	}

	// Verify all 5 steps were executed.
	if got := len(session.Steps); got != 5 {
		t.Fatalf("expected 5 steps, got %d", got)
	}

	// Verify generateFn was called exactly 5 times.
	mu.Lock()
	if stepIdx != 5 {
		t.Fatalf("expected generateFn called 5 times, got %d", stepIdx)
	}
	mu.Unlock()

	// Verify each expected tool was called exactly once.
	expectedTools := []string{"GetMarketData", "GetPortfolio", "GetOrderBook", "SearchNews"}
	for _, tool := range expectedTools {
		count := provider.callCount(tool)
		if count != 1 {
			t.Errorf("expected tool %q called 1 time, got %d", tool, count)
		}
	}

	// Verify steps 1-4 had tool calls and step 5 did not.
	for i := 0; i < 4; i++ {
		if len(session.Steps[i].ToolCalls) == 0 {
			t.Errorf("step %d: expected tool calls, got none", i+1)
		}
		if len(session.Steps[i].ToolResults) == 0 {
			t.Errorf("step %d: expected tool results, got none", i+1)
		}
		for _, r := range session.Steps[i].ToolResults {
			if r.IsError {
				t.Errorf("step %d: tool result is error: %s", i+1, r.Output)
			}
		}
	}

	// Step 5: no tool calls, final recommendation output.
	if len(session.Steps[4].ToolCalls) != 0 {
		t.Errorf("step 5: expected no tool calls, got %d", len(session.Steps[4].ToolCalls))
	}

	// Final output should contain the recommendation.
	if !strings.Contains(session.FinalOutput, "Recommendation") {
		t.Error("final output does not contain recommendation")
	}

	// Verify total tokens is positive.
	if session.TotalTokens <= 0 {
		t.Errorf("expected positive total tokens, got %d", session.TotalTokens)
	}
}

// TestAgentIntegrationTimeout verifies that the agent loop respects context
// cancellation and does not hang indefinitely.
func TestAgentIntegrationTimeout(t *testing.T) {
	provider := newMockMarketProvider()
	risk := &mockRiskApprover{}
	toolSet := NewMarketToolSet(provider, risk)
	registry := NewToolRegistry()
	if err := toolSet.RegisterAll(registry); err != nil {
		t.Fatalf("RegisterAll: %v", err)
	}

	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{
		MaxSteps: 100,
	}, registry, parser)

	// generateFn blocks until context is cancelled, simulating a slow model.
	generateFn := func(ctx context.Context, history []string) (string, error) {
		<-ctx.Done()
		return "", ctx.Err()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err := sup.RunLoop(ctx, generateFn, "loop forever")
	if err == nil {
		t.Fatal("expected context error, got nil")
	}
	if !strings.Contains(err.Error(), "context") {
		t.Fatalf("expected context-related error, got: %v", err)
	}
}

// TestAgentIntegrationMaxSteps verifies that the agent loop respects the
// MaxSteps configuration and stops after the limit.
func TestAgentIntegrationMaxSteps(t *testing.T) {
	provider := newMockMarketProvider()
	risk := &mockRiskApprover{}
	toolSet := NewMarketToolSet(provider, risk)
	registry := NewToolRegistry()
	if err := toolSet.RegisterAll(registry); err != nil {
		t.Fatalf("RegisterAll: %v", err)
	}

	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{
		MaxSteps: 3,
	}, registry, parser)

	// Always return a tool call so the loop never naturally finishes.
	generateFn := func(ctx context.Context, history []string) (string, error) {
		return `{"name": "GetMarketData", "arguments": {"symbol": "BTCUSD"}}`, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	session, err := sup.RunLoop(ctx, generateFn, "keep calling tools")
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}

	if session.StopReason != "max_steps" {
		t.Fatalf("expected stop reason %q, got %q", "max_steps", session.StopReason)
	}
	if len(session.Steps) != 3 {
		t.Fatalf("expected 3 steps, got %d", len(session.Steps))
	}
}

// TestAgentIntegrationMultipleToolCalls verifies that the parser and supervisor
// handle multiple tool calls in a single model output.
func TestAgentIntegrationMultipleToolCalls(t *testing.T) {
	provider := newMockMarketProvider()
	risk := &mockRiskApprover{}
	toolSet := NewMarketToolSet(provider, risk)
	registry := NewToolRegistry()
	if err := toolSet.RegisterAll(registry); err != nil {
		t.Fatalf("RegisterAll: %v", err)
	}

	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{
		MaxSteps: 5,
	}, registry, parser)

	steps := []string{
		// Step 1: Two tool calls in one output.
		`Let me get market data and portfolio simultaneously.
{"name": "GetMarketData", "arguments": {"symbol": "BTCUSD"}}
{"name": "GetPortfolio", "arguments": {"account_id": "main"}}`,

		// Step 2: Final output.
		`Based on the data, position size should be 0.1 BTC. Recommendation complete.`,
	}

	stepIdx := 0
	generateFn := func(ctx context.Context, history []string) (string, error) {
		if stepIdx >= len(steps) {
			return "done", nil
		}
		out := steps[stepIdx]
		stepIdx++
		return out, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	session, err := sup.RunLoop(ctx, generateFn, "Research BTCUSD")
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}

	if session.StopReason != "no_tools" {
		t.Fatalf("expected stop reason %q, got %q", "no_tools", session.StopReason)
	}

	// Step 1 should have 2 tool calls.
	if got := len(session.Steps[0].ToolCalls); got != 2 {
		t.Fatalf("step 1: expected 2 tool calls, got %d", got)
	}

	// Both tools should have been called.
	if provider.callCount("GetMarketData") != 1 {
		t.Error("GetMarketData not called")
	}
	if provider.callCount("GetPortfolio") != 1 {
		t.Error("GetPortfolio not called")
	}
}

// TestAgentIntegrationToolResultsInHistory verifies that tool results are
// correctly passed back to the generate function via history.
func TestAgentIntegrationToolResultsInHistory(t *testing.T) {
	provider := newMockMarketProvider()
	risk := &mockRiskApprover{}
	toolSet := NewMarketToolSet(provider, risk)
	registry := NewToolRegistry()
	if err := toolSet.RegisterAll(registry); err != nil {
		t.Fatalf("RegisterAll: %v", err)
	}

	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{
		MaxSteps: 5,
	}, registry, parser)

	var capturedHistories [][]string
	var mu sync.Mutex

	steps := []string{
		`{"name": "GetMarketData", "arguments": {"symbol": "BTCUSD"}}`,
		`Done. The price is $67,500.50.`,
	}

	stepIdx := 0
	generateFn := func(ctx context.Context, history []string) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		cp := make([]string, len(history))
		copy(cp, history)
		capturedHistories = append(capturedHistories, cp)
		out := steps[stepIdx]
		stepIdx++
		return out, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := sup.RunLoop(ctx, generateFn, "get BTCUSD price")
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	// First call: history should only contain the initial prompt.
	if len(capturedHistories) < 2 {
		t.Fatalf("expected at least 2 history captures, got %d", len(capturedHistories))
	}
	if len(capturedHistories[0]) != 1 {
		t.Fatalf("first call: expected 1 history entry, got %d", len(capturedHistories[0]))
	}
	if capturedHistories[0][0] != "get BTCUSD price" {
		t.Errorf("first call: expected initial prompt, got %q", capturedHistories[0][0])
	}

	// Second call: history should contain the prompt + tool result.
	if len(capturedHistories[1]) < 2 {
		t.Fatalf("second call: expected at least 2 history entries, got %d", len(capturedHistories[1]))
	}

	// The second history entry should be a formatted tool result containing the price.
	var result struct {
		Output string `json:"output"`
	}
	if err := json.Unmarshal([]byte(capturedHistories[1][1]), &result); err != nil {
		t.Fatalf("failed to parse tool result in history: %v", err)
	}
	if !strings.Contains(result.Output, "67500.5") {
		t.Errorf("tool result does not contain expected price, got: %s", result.Output)
	}
}
