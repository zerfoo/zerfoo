See docs/design.md to understand the project vision and continuously check whether the codebase is alined with the vision. Suggest to me improvements to the design where applicable so we can update docs/design.md which is supposed to be a living document and in parity with the codebase.

See docs/tree.md for directory structure, suggest changes to me where such changes would improve clean architecture.

Always check if a file has content before writing to it to avoid accidentally overwriting code.

Always follow test driven architecture to ensure correctness as you progress.

Use standard library instead of third party libraries. For command-line tools, use the standard `flag` package instead of heavier frameworks like `cobra` unless the complexity absolutely requires it.

Use a test driven development approach.

Make many small commits.