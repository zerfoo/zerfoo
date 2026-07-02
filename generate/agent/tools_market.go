package agent

import (
	"encoding/json"
	"fmt"
	"sync"
)

// RiskApprover evaluates whether an order should be permitted.
type RiskApprover interface {
	Approve(symbol string, side string, quantity float64, price float64) (bool, string, error)
}

// MarketDataProvider supplies market data, order books, portfolios,
// earnings calendars, and news to the agentic tool set.
type MarketDataProvider interface {
	GetMarketData(symbol string) (MarketQuote, error)
	GetOrderBook(symbol string, depth int) (OrderBook, error)
	GetPortfolio(accountID string) (Portfolio, error)
	GetEarningsCalendar(startDate, endDate string) ([]EarningsEvent, error)
	SearchNews(query string, limit int) ([]NewsArticle, error)
}

// MarketQuote holds a price quote for a single symbol.
type MarketQuote struct {
	Symbol    string  `json:"symbol"`
	Price     float64 `json:"price"`
	Bid       float64 `json:"bid"`
	Ask       float64 `json:"ask"`
	Volume    int64   `json:"volume"`
	Timestamp string  `json:"timestamp"`
}

// OrderBookEntry represents a single level in an order book.
type OrderBookEntry struct {
	Price    float64 `json:"price"`
	Quantity float64 `json:"quantity"`
}

// OrderBook represents the bids and asks for a symbol.
type OrderBook struct {
	Symbol string           `json:"symbol"`
	Bids   []OrderBookEntry `json:"bids"`
	Asks   []OrderBookEntry `json:"asks"`
}

// PortfolioPosition is a single holding in a portfolio.
type PortfolioPosition struct {
	Symbol   string  `json:"symbol"`
	Quantity float64 `json:"quantity"`
	AvgCost  float64 `json:"avg_cost"`
	Current  float64 `json:"current"`
}

// Portfolio represents account holdings.
type Portfolio struct {
	AccountID string              `json:"account_id"`
	Cash      float64             `json:"cash"`
	Positions []PortfolioPosition `json:"positions"`
}

// EarningsEvent represents a scheduled earnings release.
type EarningsEvent struct {
	Symbol       string `json:"symbol"`
	Date         string `json:"date"`
	EstimatedEPS string `json:"estimated_eps"`
}

// NewsArticle is a single news search result.
type NewsArticle struct {
	Title     string `json:"title"`
	Source    string `json:"source"`
	URL       string `json:"url"`
	Summary   string `json:"summary"`
	Published string `json:"published"`
}

// OrderRequest is the input for SubmitOrder.
type OrderRequest struct {
	Symbol   string  `json:"symbol"`
	Side     string  `json:"side"`
	Quantity float64 `json:"quantity"`
	Price    float64 `json:"price"`
}

// OrderResponse is the result of a submitted order.
type OrderResponse struct {
	OrderID  string `json:"order_id"`
	Status   string `json:"status"`
	Message  string `json:"message,omitempty"`
	Approved bool   `json:"approved"`
}

// MarketToolSet holds the dependencies for market-related agent tools.
type MarketToolSet struct {
	provider MarketDataProvider
	risk     RiskApprover
	mu       sync.Mutex
	orderSeq int
}

// NewMarketToolSet creates a new MarketToolSet.
func NewMarketToolSet(provider MarketDataProvider, risk RiskApprover) *MarketToolSet {
	return &MarketToolSet{
		provider: provider,
		risk:     risk,
	}
}

// RegisterAll registers all market tools into the given registry.
func (m *MarketToolSet) RegisterAll(reg *ToolRegistry) error {
	tools := []struct {
		def ToolDef
		fn  func(json.RawMessage) (string, error)
	}{
		{
			def: ToolDef{
				Name:        "GetMarketData",
				Description: "Get current market data (price, bid, ask, volume) for a symbol",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}`),
			},
			fn: m.getMarketData,
		},
		{
			def: ToolDef{
				Name:        "GetOrderBook",
				Description: "Get the order book (bids and asks) for a symbol",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"symbol":{"type":"string"},"depth":{"type":"integer"}},"required":["symbol"]}`),
			},
			fn: m.getOrderBook,
		},
		{
			def: ToolDef{
				Name:        "GetPortfolio",
				Description: "Get portfolio holdings for an account",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"account_id":{"type":"string"}},"required":["account_id"]}`),
			},
			fn: m.getPortfolio,
		},
		{
			def: ToolDef{
				Name:        "GetEarningsCalendar",
				Description: "Get upcoming earnings events within a date range",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"start_date":{"type":"string"},"end_date":{"type":"string"}},"required":["start_date","end_date"]}`),
			},
			fn: m.getEarningsCalendar,
		},
		{
			def: ToolDef{
				Name:        "SearchNews",
				Description: "Search for news articles by query",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}`),
			},
			fn: m.searchNews,
		},
		{
			def: ToolDef{
				Name:        "SubmitOrder",
				Description: "Submit a buy or sell order (requires risk approval)",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"symbol":{"type":"string"},"side":{"type":"string","enum":["buy","sell"]},"quantity":{"type":"number"},"price":{"type":"number"}},"required":["symbol","side","quantity","price"]}`),
			},
			fn: m.submitOrder,
		},
	}

	for _, t := range tools {
		if err := reg.Register(t.def, t.fn); err != nil {
			return err
		}
	}
	return nil
}

func (m *MarketToolSet) getMarketData(args json.RawMessage) (string, error) {
	var params struct {
		Symbol string `json:"symbol"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if params.Symbol == "" {
		return "", fmt.Errorf("symbol is required")
	}
	quote, err := m.provider.GetMarketData(params.Symbol)
	if err != nil {
		return "", err
	}
	b, _ := json.Marshal(quote)
	return string(b), nil
}

func (m *MarketToolSet) getOrderBook(args json.RawMessage) (string, error) {
	var params struct {
		Symbol string `json:"symbol"`
		Depth  int    `json:"depth"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if params.Symbol == "" {
		return "", fmt.Errorf("symbol is required")
	}
	if params.Depth <= 0 {
		params.Depth = 5
	}
	book, err := m.provider.GetOrderBook(params.Symbol, params.Depth)
	if err != nil {
		return "", err
	}
	b, _ := json.Marshal(book)
	return string(b), nil
}

func (m *MarketToolSet) getPortfolio(args json.RawMessage) (string, error) {
	var params struct {
		AccountID string `json:"account_id"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if params.AccountID == "" {
		return "", fmt.Errorf("account_id is required")
	}
	portfolio, err := m.provider.GetPortfolio(params.AccountID)
	if err != nil {
		return "", err
	}
	b, _ := json.Marshal(portfolio)
	return string(b), nil
}

func (m *MarketToolSet) getEarningsCalendar(args json.RawMessage) (string, error) {
	var params struct {
		StartDate string `json:"start_date"`
		EndDate   string `json:"end_date"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if params.StartDate == "" || params.EndDate == "" {
		return "", fmt.Errorf("start_date and end_date are required")
	}
	events, err := m.provider.GetEarningsCalendar(params.StartDate, params.EndDate)
	if err != nil {
		return "", err
	}
	b, _ := json.Marshal(events)
	return string(b), nil
}

func (m *MarketToolSet) searchNews(args json.RawMessage) (string, error) {
	var params struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if params.Query == "" {
		return "", fmt.Errorf("query is required")
	}
	if params.Limit <= 0 {
		params.Limit = 10
	}
	articles, err := m.provider.SearchNews(params.Query, params.Limit)
	if err != nil {
		return "", err
	}
	b, _ := json.Marshal(articles)
	return string(b), nil
}

func (m *MarketToolSet) submitOrder(args json.RawMessage) (string, error) {
	var req OrderRequest
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if req.Symbol == "" {
		return "", fmt.Errorf("symbol is required")
	}
	if req.Side != "buy" && req.Side != "sell" {
		return "", fmt.Errorf("side must be \"buy\" or \"sell\"")
	}
	if req.Quantity <= 0 {
		return "", fmt.Errorf("quantity must be positive")
	}
	if req.Price <= 0 {
		return "", fmt.Errorf("price must be positive")
	}

	approved, reason, err := m.risk.Approve(req.Symbol, req.Side, req.Quantity, req.Price)
	if err != nil {
		return "", fmt.Errorf("risk check failed: %w", err)
	}

	m.mu.Lock()
	m.orderSeq++
	orderID := fmt.Sprintf("ORD-%06d", m.orderSeq)
	m.mu.Unlock()

	resp := OrderResponse{
		OrderID:  orderID,
		Approved: approved,
	}
	if approved {
		resp.Status = "submitted"
	} else {
		resp.Status = "rejected"
		resp.Message = reason
	}

	b, _ := json.Marshal(resp)
	return string(b), nil
}
