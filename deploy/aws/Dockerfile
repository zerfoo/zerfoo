# ── Build stage ──────────────────────────────────────────────────────────────
FROM golang:1.26-bookworm AS build

WORKDIR /src

# Cache module downloads separately from source.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build the zerfoo serve binary. CGo is disabled so the binary is fully static
# and portable — GPU acceleration is loaded at runtime via purego/dlopen.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" \
    -o /out/zerfoo ./cmd/zerfoo

# ── Runtime stage ─────────────────────────────────────────────────────────────
# Distroless nonroot provides a minimal, secure runtime without a shell.
FROM gcr.io/distroless/static-debian12:nonroot

# GPU shared libraries are loaded at runtime by purego/dlopen from a host-mounted
# path or an optional CUDA layer image; they are not baked into this image.

COPY --from=build /out/zerfoo /zerfoo

# Expose the OpenAI-compatible API port.
EXPOSE 8080

# Health-check endpoint (also used by ECS).
# distroless has no curl; the ECS task definition uses its own CMD-SHELL check
# against the host network. This HEALTHCHECK is for docker run / compose usage.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["/zerfoo", "health"]

ENV ZERFOO_PORT=8080
ENV ZERFOO_LOG_FORMAT=json

ENTRYPOINT ["/zerfoo"]
CMD ["serve", "--port", "8080"]
