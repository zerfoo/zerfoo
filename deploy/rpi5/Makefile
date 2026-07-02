REPO_ROOT := $(shell git rev-parse --show-toplevel)
OUTPUT    := $(REPO_ROOT)/zerfoo-edge-rpi5
RPI5_HOST ?= pi@raspberrypi.local
REMOTE_DIR ?= ~/zerfoo

.PHONY: build test deploy

## build: Cross-compile zerfoo-edge for Raspberry Pi 5 (arm64 linux).
build:
	@OUTPUT="$(OUTPUT)" bash $(REPO_ROOT)/deploy/rpi5/build.sh

## test: Run go vet on cmd/zerfoo-edge and verify the cross-compiled binary.
test:
	@echo "Running go vet..."
	cd $(REPO_ROOT) && go vet ./cmd/zerfoo-edge/...
	@if [ -x "$(OUTPUT)" ]; then \
		echo "Binary $(OUTPUT) exists — run test.sh on the RPi5 to validate."; \
	else \
		echo "Binary not found at $(OUTPUT). Run 'make build' first."; \
		exit 1; \
	fi

## deploy: Copy binary and test script to RPi5 via SSH. Set RPI5_HOST to override.
deploy: build
	@echo "Deploying to $(RPI5_HOST):$(REMOTE_DIR)..."
	ssh $(RPI5_HOST) "mkdir -p $(REMOTE_DIR)"
	scp $(OUTPUT) $(RPI5_HOST):$(REMOTE_DIR)/zerfoo-edge
	scp $(REPO_ROOT)/deploy/rpi5/test.sh $(RPI5_HOST):$(REMOTE_DIR)/test.sh
	ssh $(RPI5_HOST) "chmod +x $(REMOTE_DIR)/zerfoo-edge $(REMOTE_DIR)/test.sh"
	@echo "Deployed. To validate, run:"
	@echo "  ssh $(RPI5_HOST) '$(REMOTE_DIR)/test.sh $(REMOTE_DIR)/zerfoo-edge'"
