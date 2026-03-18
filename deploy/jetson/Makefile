# Makefile — Jetson Orin Nano deployment targets for zerfoo-edge
#
# Targets:
#   build       Cross-compile CPU-only binary (GOARCH=arm64 GOOS=linux)
#   build-cuda  Cross-compile with CUDA support (requires JetPack toolchain)
#   test        Run on-device validation (SSH to Jetson and execute test.sh)
#   deploy      Copy binary and test script to Jetson via SCP

REPO_ROOT := $(shell cd ../..; pwd)
BINARY    := $(REPO_ROOT)/zerfoo-edge-jetson
BINARY_CUDA := $(REPO_ROOT)/zerfoo-edge-jetson-cuda

# Jetson SSH target — override with: make deploy JETSON_HOST=user@192.168.x.x
JETSON_HOST ?= jetson
JETSON_USER ?= $(shell whoami)
JETSON_ADDR ?= $(JETSON_USER)@$(JETSON_HOST)
DEPLOY_DIR  ?= ~

.PHONY: build build-cuda test deploy clean

## build: Cross-compile CPU-only zerfoo-edge for Jetson Orin Nano
build:
	@bash $(REPO_ROOT)/deploy/jetson/build.sh

## build-cuda: Cross-compile zerfoo-edge with CUDA support for Jetson
build-cuda:
	@bash $(REPO_ROOT)/deploy/jetson/build-cuda.sh

## deploy: Copy binary and test script to Jetson (requires SSH access)
deploy: build
	@echo "Deploying to $(JETSON_ADDR):$(DEPLOY_DIR)"
	scp $(BINARY) $(JETSON_ADDR):$(DEPLOY_DIR)/zerfoo-edge-jetson
	scp $(REPO_ROOT)/deploy/jetson/test.sh $(JETSON_ADDR):$(DEPLOY_DIR)/test.sh
	ssh $(JETSON_ADDR) "chmod +x $(DEPLOY_DIR)/zerfoo-edge-jetson $(DEPLOY_DIR)/test.sh"
	@echo "Deploy complete. Run: ssh $(JETSON_ADDR) bash $(DEPLOY_DIR)/test.sh"

## test: Run on-device validation via SSH
test:
	@echo "Running on-device tests on $(JETSON_ADDR)..."
	ssh $(JETSON_ADDR) "BINARY=$(DEPLOY_DIR)/zerfoo-edge-jetson bash $(DEPLOY_DIR)/test.sh"

## clean: Remove built binaries
clean:
	rm -f $(BINARY) $(BINARY_CUDA)
