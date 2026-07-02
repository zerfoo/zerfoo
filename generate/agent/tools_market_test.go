package agent

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// mockProvider implements MarketDataProvider with canned data.
type mockProvider struct {
	quotes    map[string]MarketQuote
	books     map[string]OrderBook
	portfolio Portfolio
	earnings  []EarningsEvent
	news      []NewsArticle
}

func (m *mockProvider) GetMarketData(symbol string) (MarketQuote, error) {
	q, ok := m.quotes[symbol]
	if !ok {
		return MarketQuote{}, fmt.Errorf("symbol %q not found", symbol)
	}
	return q, nil
}

func (m *mockProvider) GetOrderBook(symbol string, depth int) (OrderBook, error) {
	b, ok := m.books[symbol]
	if !ok {
		return OrderBook{}, fmt.Errorf("symbol %q not found", symbol)
	}
	ob := OrderBook{Symbol: b.Symbol}
	if depth < len(b.Bids) {
		ob.Bids = b.Bids[:depth]
	} else {
		ob.Bids = b.Bids
	}
	if depth < len(b.Asks) {
		ob.Asks = b.Asks[:depth]
	} else {
		ob.Asks = b.Asks
	}
	return ob, nil
}

func (m *mockProvider) GetPortfolio(accountID string) (Portfolio, error) {
	if m.portfolio.AccountID != accountID {
		return Portfolio{}, fmt.Errorf("account %q not found", accountID)
	}
	return m.portfolio, nil
}

func (m *mockProvider) GetEarningsCalendar(startDate, endDate string) ([]EarningsEvent, error) {
	return m.earnings, nil
}

func (m *mockProvider) SearchNews(query string, limit int) ([]NewsArticle, error) {
	var results []NewsArticle
	for _, a := range m.news {
		if strings.Contains(strings.ToLower(a.Title), strings.ToLower(query)) {
			results = append(results, a)
			if len(results) >= limit {
				break
			}
		}
	}
	return results, nil
}

// stubRiskApprover always approves orders.
type stubRiskApprover struct{}

func (s *stubRiskApprover) Approve(symbol, side string, quantity, price float64) (bool, string, error) {
	return true, "", nil
}

// rejectingRiskApprover always rejects orders.
type rejectingRiskApprover struct {
	reason string
}

func (r *rejectingRiskApprover) Approve(symbol, side string, quantity, price float64) (bool, string, error) {
	return false, r.reason, nil
}

func newTestProvider() *mockProvider {
	return &mockProvider{
		quotes: map[string]MarketQuote{
			"AAPL": {Symbol: "AAPL", Price: 185.50, Bid: 185.45, Ask: 185.55, Volume: 1000000, Timestamp: "2026-03-17T10:00:00Z"},
			"GOOG": {Symbol: "GOOG", Price: 142.30, Bid: 142.25, Ask: 142.35, Volume: 500000, Timestamp: "2026-03-17T10:00:00Z"},
		},
		books: map[string]OrderBook{
			"AAPL": {
				Symbol: "AAPL",
				Bids:   []OrderBookEntry{{Price: 185.45, Quantity: 100}, {Price: 185.40, Quantity: 200}, {Price: 185.35, Quantity: 300}},
				Asks:   []OrderBookEntry{{Price: 185.55, Quantity: 150}, {Price: 185.60, Quantity: 250}, {Price: 185.65, Quantity: 350}},
			},
		},
		portfolio: Portfolio{
			AccountID: "acct-001",
			Cash:      50000.00,
			Positions: []PortfolioPosition{
				{Symbol: "AAPL", Quantity: 100, AvgCost: 170.00, Current: 185.50},
			},
		},
		earnings: []EarningsEvent{
			{Symbol: "AAPL", Date: "2026-04-25", EstimatedEPS: "1.62"},
			{Symbol: "GOOG", Date: "2026-04-28", EstimatedEPS: "1.85"},
		},
		news: []NewsArticle{
			{Title: "Apple reports record quarter", Source: "Reuters", URL: "https://example.com/1", Summary: "Apple Q1 earnings beat estimates", Published: "2026-03-16"},
			{Title: "Google launches new AI model", Source: "Bloomberg", URL: "https://example.com/2", Summary: "Alphabet unveils next gen AI", Published: "2026-03-15"},
		},
	}
}

func TestMarketTools_RegisterAll(t *testing.T) {
	reg := NewToolRegistry()
	ts := NewMarketToolSet(newTestProvider(), &stubRiskApprover{})
	if err := ts.RegisterAll(reg); err != nil {
		t.Fatalf("RegisterAll: %v", err)
	}

	want := []string{"GetEarningsCalendar", "GetMarketData", "GetOrderBook", "GetPortfolio", "SearchNews", "SubmitOrder"}
	defs := reg.List()
	if len(defs) != len(want) {
		t.Fatalf("got %d tools, want %d", len(defs), len(want))
	}
	for i, d := range defs {
		if d.Name != want[i] {
			t.Errorf("tool[%d] = %q, want %q", i, d.Name, want[i])
		}
	}
}

func TestMarketTools_GetMarketData(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		wantErr string
		check   func(t *testing.T, output string)
	}{
		{
			name: "valid symbol",
			args: `{"symbol":"AAPL"}`,
			check: func(t *testing.T, output string) {
				var q MarketQuote
				if err := json.Unmarshal([]byte(output), &q); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if q.Symbol != "AAPL" || q.Price != 185.50 {
					t.Errorf("got %+v", q)
				}
			},
		},
		{
			name:    "unknown symbol",
			args:    `{"symbol":"ZZZZ"}`,
			wantErr: "not found",
		},
		{
			name:    "missing symbol",
			args:    `{}`,
			wantErr: "symbol is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reg := NewToolRegistry()
			ts := NewMarketToolSet(newTestProvider(), &stubRiskApprover{})
			if err := ts.RegisterAll(reg); err != nil {
				t.Fatal(err)
			}
			result := reg.Call(ToolCall{ID: "1", Name: "GetMarketData", Arguments: json.RawMessage(tt.args)})
			if tt.wantErr != "" {
				if !result.IsError {
					t.Fatal("expected error")
				}
				if !strings.Contains(result.Output, tt.wantErr) {
					t.Errorf("error %q does not contain %q", result.Output, tt.wantErr)
				}
				return
			}
			if result.IsError {
				t.Fatalf("unexpected error: %s", result.Output)
			}
			if tt.check != nil {
				tt.check(t, result.Output)
			}
		})
	}
}

func TestMarketTools_GetOrderBook(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		wantErr string
		check   func(t *testing.T, output string)
	}{
		{
			name: "default depth",
			args: `{"symbol":"AAPL"}`,
			check: func(t *testing.T, output string) {
				var ob OrderBook
				if err := json.Unmarshal([]byte(output), &ob); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if ob.Symbol != "AAPL" {
					t.Errorf("symbol = %q", ob.Symbol)
				}
				if len(ob.Bids) != 3 {
					t.Errorf("got %d bids, want 3", len(ob.Bids))
				}
			},
		},
		{
			name: "limited depth",
			args: `{"symbol":"AAPL","depth":2}`,
			check: func(t *testing.T, output string) {
				var ob OrderBook
				if err := json.Unmarshal([]byte(output), &ob); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if len(ob.Bids) != 2 {
					t.Errorf("got %d bids, want 2", len(ob.Bids))
				}
			},
		},
		{
			name:    "unknown symbol",
			args:    `{"symbol":"ZZZZ"}`,
			wantErr: "not found",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reg := NewToolRegistry()
			ts := NewMarketToolSet(newTestProvider(), &stubRiskApprover{})
			if err := ts.RegisterAll(reg); err != nil {
				t.Fatal(err)
			}
			result := reg.Call(ToolCall{ID: "1", Name: "GetOrderBook", Arguments: json.RawMessage(tt.args)})
			if tt.wantErr != "" {
				if !result.IsError {
					t.Fatal("expected error")
				}
				if !strings.Contains(result.Output, tt.wantErr) {
					t.Errorf("error %q does not contain %q", result.Output, tt.wantErr)
				}
				return
			}
			if result.IsError {
				t.Fatalf("unexpected error: %s", result.Output)
			}
			if tt.check != nil {
				tt.check(t, result.Output)
			}
		})
	}
}

func TestMarketTools_GetPortfolio(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		wantErr string
		check   func(t *testing.T, output string)
	}{
		{
			name: "valid account",
			args: `{"account_id":"acct-001"}`,
			check: func(t *testing.T, output string) {
				var p Portfolio
				if err := json.Unmarshal([]byte(output), &p); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if p.Cash != 50000.00 {
					t.Errorf("cash = %f", p.Cash)
				}
				if len(p.Positions) != 1 {
					t.Errorf("positions = %d", len(p.Positions))
				}
			},
		},
		{
			name:    "unknown account",
			args:    `{"account_id":"acct-999"}`,
			wantErr: "not found",
		},
		{
			name:    "missing account_id",
			args:    `{}`,
			wantErr: "account_id is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reg := NewToolRegistry()
			ts := NewMarketToolSet(newTestProvider(), &stubRiskApprover{})
			if err := ts.RegisterAll(reg); err != nil {
				t.Fatal(err)
			}
			result := reg.Call(ToolCall{ID: "1", Name: "GetPortfolio", Arguments: json.RawMessage(tt.args)})
			if tt.wantErr != "" {
				if !result.IsError {
					t.Fatal("expected error")
				}
				if !strings.Contains(result.Output, tt.wantErr) {
					t.Errorf("error %q does not contain %q", result.Output, tt.wantErr)
				}
				return
			}
			if result.IsError {
				t.Fatalf("unexpected error: %s", result.Output)
			}
			if tt.check != nil {
				tt.check(t, result.Output)
			}
		})
	}
}

func TestMarketTools_GetEarningsCalendar(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		wantErr string
		check   func(t *testing.T, output string)
	}{
		{
			name: "valid range",
			args: `{"start_date":"2026-04-01","end_date":"2026-04-30"}`,
			check: func(t *testing.T, output string) {
				var events []EarningsEvent
				if err := json.Unmarshal([]byte(output), &events); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if len(events) != 2 {
					t.Errorf("got %d events, want 2", len(events))
				}
			},
		},
		{
			name:    "missing dates",
			args:    `{"start_date":"2026-04-01"}`,
			wantErr: "end_date are required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reg := NewToolRegistry()
			ts := NewMarketToolSet(newTestProvider(), &stubRiskApprover{})
			if err := ts.RegisterAll(reg); err != nil {
				t.Fatal(err)
			}
			result := reg.Call(ToolCall{ID: "1", Name: "GetEarningsCalendar", Arguments: json.RawMessage(tt.args)})
			if tt.wantErr != "" {
				if !result.IsError {
					t.Fatal("expected error")
				}
				if !strings.Contains(result.Output, tt.wantErr) {
					t.Errorf("error %q does not contain %q", result.Output, tt.wantErr)
				}
				return
			}
			if result.IsError {
				t.Fatalf("unexpected error: %s", result.Output)
			}
			if tt.check != nil {
				tt.check(t, result.Output)
			}
		})
	}
}

func TestMarketTools_SearchNews(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		wantErr string
		check   func(t *testing.T, output string)
	}{
		{
			name: "matching query",
			args: `{"query":"Apple","limit":5}`,
			check: func(t *testing.T, output string) {
				var articles []NewsArticle
				if err := json.Unmarshal([]byte(output), &articles); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if len(articles) != 1 {
					t.Errorf("got %d articles, want 1", len(articles))
				}
			},
		},
		{
			name: "no matches",
			args: `{"query":"nonexistent"}`,
			check: func(t *testing.T, output string) {
				if output != "null" && output != "[]" {
					var articles []NewsArticle
					if err := json.Unmarshal([]byte(output), &articles); err != nil {
						t.Fatalf("unmarshal: %v", err)
					}
					if len(articles) != 0 {
						t.Errorf("got %d articles, want 0", len(articles))
					}
				}
			},
		},
		{
			name:    "missing query",
			args:    `{}`,
			wantErr: "query is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reg := NewToolRegistry()
			ts := NewMarketToolSet(newTestProvider(), &stubRiskApprover{})
			if err := ts.RegisterAll(reg); err != nil {
				t.Fatal(err)
			}
			result := reg.Call(ToolCall{ID: "1", Name: "SearchNews", Arguments: json.RawMessage(tt.args)})
			if tt.wantErr != "" {
				if !result.IsError {
					t.Fatal("expected error")
				}
				if !strings.Contains(result.Output, tt.wantErr) {
					t.Errorf("error %q does not contain %q", result.Output, tt.wantErr)
				}
				return
			}
			if result.IsError {
				t.Fatalf("unexpected error: %s", result.Output)
			}
			if tt.check != nil {
				tt.check(t, result.Output)
			}
		})
	}
}

func TestMarketTools_SubmitOrder(t *testing.T) {
	tests := []struct {
		name     string
		args     string
		risk     RiskApprover
		wantErr  string
		approved bool
	}{
		{
			name:     "approved order",
			args:     `{"symbol":"AAPL","side":"buy","quantity":10,"price":185.50}`,
			risk:     &stubRiskApprover{},
			approved: true,
		},
		{
			name:     "rejected order",
			args:     `{"symbol":"AAPL","side":"buy","quantity":10000,"price":185.50}`,
			risk:     &rejectingRiskApprover{reason: "position too large"},
			approved: false,
		},
		{
			name:    "invalid side",
			args:    `{"symbol":"AAPL","side":"short","quantity":10,"price":185.50}`,
			risk:    &stubRiskApprover{},
			wantErr: "side must be",
		},
		{
			name:    "zero quantity",
			args:    `{"symbol":"AAPL","side":"buy","quantity":0,"price":185.50}`,
			risk:    &stubRiskApprover{},
			wantErr: "quantity must be positive",
		},
		{
			name:    "missing symbol",
			args:    `{"side":"buy","quantity":10,"price":185.50}`,
			risk:    &stubRiskApprover{},
			wantErr: "symbol is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reg := NewToolRegistry()
			ts := NewMarketToolSet(newTestProvider(), tt.risk)
			if err := ts.RegisterAll(reg); err != nil {
				t.Fatal(err)
			}
			result := reg.Call(ToolCall{ID: "1", Name: "SubmitOrder", Arguments: json.RawMessage(tt.args)})
			if tt.wantErr != "" {
				if !result.IsError {
					t.Fatal("expected error")
				}
				if !strings.Contains(result.Output, tt.wantErr) {
					t.Errorf("error %q does not contain %q", result.Output, tt.wantErr)
				}
				return
			}
			if result.IsError {
				t.Fatalf("unexpected error: %s", result.Output)
			}
			var resp OrderResponse
			if err := json.Unmarshal([]byte(result.Output), &resp); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if resp.Approved != tt.approved {
				t.Errorf("approved = %v, want %v", resp.Approved, tt.approved)
			}
			if resp.OrderID == "" {
				t.Error("expected non-empty order ID")
			}
			if tt.approved && resp.Status != "submitted" {
				t.Errorf("status = %q, want submitted", resp.Status)
			}
			if !tt.approved && resp.Status != "rejected" {
				t.Errorf("status = %q, want rejected", resp.Status)
			}
		})
	}
}

func TestMarketTools_SubmitOrder_SequentialIDs(t *testing.T) {
	reg := NewToolRegistry()
	ts := NewMarketToolSet(newTestProvider(), &stubRiskApprover{})
	if err := ts.RegisterAll(reg); err != nil {
		t.Fatal(err)
	}

	args := json.RawMessage(`{"symbol":"AAPL","side":"buy","quantity":1,"price":185.50}`)
	var ids []string
	for i := 0; i < 3; i++ {
		result := reg.Call(ToolCall{ID: fmt.Sprintf("call-%d", i), Name: "SubmitOrder", Arguments: args})
		if result.IsError {
			t.Fatalf("unexpected error: %s", result.Output)
		}
		var resp OrderResponse
		if err := json.Unmarshal([]byte(result.Output), &resp); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, resp.OrderID)
	}

	want := []string{"ORD-000001", "ORD-000002", "ORD-000003"}
	for i, id := range ids {
		if id != want[i] {
			t.Errorf("order[%d] = %q, want %q", i, id, want[i])
		}
	}
}
