.PHONY: proto

# Run tests
test:
	go test ./...

# Generate gRPC
proto:
	protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative distributed/pb/dist.proto distributed/pb/coordinator.proto
