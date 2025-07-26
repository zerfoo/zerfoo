# pkg/context Package

This package declares the hardware abstraction layer (HAL) context interface. The `Context` interface defines operations like matrix multiplication and addition in a hardware-agnostic way:contentReference[oaicite:26]{index=26}. Concrete implementations (e.g., `CPUContext`, GPU contexts) will fulfill this interface so that models can run on different hardware by using the appropriate context.
