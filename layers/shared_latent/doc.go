// Experimental — this package is not yet wired into the main framework.
//
// Package shared provides a cross-model latent space for knowledge sharing.
//
// A LatentSpace defines a common embedding space that multiple models can
// project into and read from. Each model registers with its own input
// dimension. Learned projection matrices map between model-specific
// representation spaces and the shared latent space.
//
// The workflow is:
//
//  1. Create a latent space with NewLatentSpace(dim, engine).
//  2. Register models via Register(name, inputDim).
//  3. Train projection matrices via TrainProjections with aligned data.
//  4. Use Project to map model features into the shared space.
//  5. Use Retrieve to map shared representations back to a model's space.
//
// This enables knowledge transfer: what one model learns can benefit other
// models through the shared embedding.
//
// (Stability: alpha)
package shared_latent
