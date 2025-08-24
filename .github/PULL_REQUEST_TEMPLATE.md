---
name: Pull Request
about: Propose changes to the codebase
title: ""
labels: ""
assignees: ""
---

## Description

<!-- Briefly describe the changes introduced by this pull request. -->

## Related Issues

<!-- Link any related issues (e.g., "Fixes #123", "Closes #456"). -->

## Checklist

- [ ] Code follows the project's [coding style guidelines](LINK_TO_CODING_GUIDELINES).
- [ ] Tests have been added or updated to cover the changes.
- [ ] All existing tests pass.
- [ ] Documentation has been updated (if applicable).

### Boundary Compliance Checklist

- [ ] **`zerfoo` module:**
  - [ ] Does NOT import `onnx` or `zonnx` packages.
  - [ ] Contains no direct references to `onnx` or `zonnx` outside of `docs`, `tests`, or `examples`.
- [ ] **`zonnx` module:**
  - [ ] Does NOT import `github.com/zerfoo/zerfoo`.
- [ ] **`zmf` module:**
  - [ ] Depends ONLY on `protobuf` (and test-only utilities).

## Reviewers

<!-- Mention any specific reviewers you'd like to involve. -->

