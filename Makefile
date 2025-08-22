.PHONY: proto test test-coverage coverage-report format lint lint-fix check

# Run tests
test:
	go test ./...

# Run tests with coverage
test-coverage:
	@echo "🧪 Running tests with coverage analysis..."
	@go test -coverprofile=coverage.out ./...
	@echo "✅ Tests completed with coverage data"

# Generate and display coverage report
coverage-report: test-coverage
	@echo "📊 Coverage Summary:"
	@go tool cover -func=coverage.out | tail -1
	@echo ""
	@echo "📋 Detailed Coverage Report:"
	@go tool cover -func=coverage.out
	@echo ""
	@echo "💡 To view HTML coverage report, run: go tool cover -html=coverage.out"

# Generate gRPC
proto:
	protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative distributed/pb/dist.proto distributed/pb/coordinator.proto

# Format code using standard Go tools
format:
	@echo "🎨 Applying code formatters..."
	@echo "  - Standard Go formatting..."
	@gofmt -w .
	@echo "  - Organizing imports..."
	@goimports -w .
	@echo "  - Strict formatting with gofumpt..."
	@gofumpt -w . 2>/dev/null || echo "    (gofumpt not available, skipping)"
	@echo "✅ Code formatting complete"

# Run linters with auto-fix
lint-fix:
	@echo "🔧 Organizing imports (goimports)..."
	@goimports -w .
	@echo "🔧 Running linters with auto-fix (golangci-lint)..."
	@golangci-lint run --fix --timeout=5m || true
	@echo "✅ Auto-fixable issues resolved"

# Run full lint check (no fixes)
lint:
	@echo "🔍 Running full lint check..."
	@golangci-lint run --timeout=5m

# Complete code quality pipeline: format -> auto-fix -> final check
check: format lint-fix lint
	@echo "🎉 Code quality check complete!"

# Complete CI pipeline: format -> lint -> test with coverage
ci: format lint-fix lint test-coverage
	@echo "🚀 CI pipeline complete - ready for deployment!"
