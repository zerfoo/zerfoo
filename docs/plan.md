# Project Plan: Remove Audacity References

## 1. Context

- **Problem Statement:** The codebase contains multiple references to a deprecated or separate project named "audacity". These references are present in user-facing messages, comments, and documentation, causing confusion and pointing to obsolete workflows. The goal is to remove all such references to clean up the codebase and streamline user experience.
- **Objectives:**
  - Eliminate all mentions of "audacity" from the codebase.
  - Remove associated dead or deprecated code that directs users to "audacity".
  - Ensure the application remains fully functional and all tests pass after the removal.
- **Non-Goals:**
  - Re-implementing the functionality that was moved to "audacity".
  - Refactoring code that is unrelated to the "audacity" references.
- **Constraints & Assumptions:**
  - The "audacity" project is considered fully deprecated or separate, and no functionality from it needs to be preserved within this project.
  - The removal of these references will not impact any core functionality of the `zerfoo` application.
- **Success Metrics:**
  - A search for "audacity" in the entire codebase yields zero results.
  - All existing unit and integration tests pass successfully.
  - The application builds and runs without errors.

## 2. Scope and Deliverables

- **In Scope:**
  - Modifying source code files to remove comments, strings, and logic related to "audacity".
  - Deleting or refactoring test files that are skipped or deprecated due to the "audacity" migration.
  - Updating documentation to remove any mention of "audacity".
- **Out of Scope:**
  - Any major refactoring not directly related to removing "audacity" references.
  - Introducing new functionality.
- **Deliverables:**
  | ID   | Description                                      | Owner | Acceptance Criteria                                                                                             |
  |------|--------------------------------------------------|-------|-----------------------------------------------------------------------------------------------------------------|
  | D1   | Codebase clean of all "audacity" references      | TBD   | A full-text search for "audacity" returns no results in the project's source files.                             |
  | D2   | Updated documentation                            | TBD   | All documentation files are free of "audacity" references.                                                      |
  | D3   | Verified build and tests                         | TBD   | The project builds successfully, and all unit and integration tests pass.                                       |

## 3. Checkable Work Breakdown

### E1: Code Cleanup

- **[ ] T1.1: Remove audacity references from `cmd/cli/framework.go`**  Owner: TBD  Est: 1h
  - **[ ] S1.1.1:** Analyze `cmd/cli/framework.go` to identify all "audacity" references.
  - **[ ] S1.1.2:** Remove the error message and related logic pointing to "audacity".
  - **[ ] S1.1.3:** Add unit tests for the changes in `cmd/cli/framework.go`.
- **[ ] T1.2: Remove audacity references from `cmd/cli/framework_test.go`**  Owner: TBD  Est: 30m
  - **[ ] S1.2.1:** Update or remove test cases that check for the "audacity" message.
- **[ ] T1.3: Remove audacity references from `cmd/zerfoo-train/main.go`**  Owner: TBD  Est: 30m
  - **[ ] S1.3.1:** Remove the deprecated `zerfoo-train` binary's main file content or the entire file if the binary is no longer needed.
- **[ ] T1.4: Remove audacity references from `integration/config_lock_integration_test.go`**  Owner: TBD  Est: 30m
  - **[ ] S1.4.1:** Remove the skipped test file `integration/config_lock_integration_test.go`.
- **[ ] T1.5: Remove audacity references from `training/interfaces_doc.go`**  Owner: TBD  Est: 1h
  - **[ ] S1.5.1:** Remove comments and documentation that refer to "audacity".

### E2: Verification and Finalization

- **[ ] T2.1: Run full test suite**  Owner: TBD  Est: 30m
  - **Dependencies:** T1.1, T1.2, T1.3, T1.4, T1.5
  - **[ ] S2.1.1:** Execute all unit and integration tests to ensure no regressions were introduced.
- **[ ] T2.2: Run linters and formatters**  Owner: TBD  Est: 30m
  - **Dependencies:** T2.1
  - **[ ] S2.2.1:** Run `gofmt` and `golangci-lint` to ensure code quality and style.
- **[ ] T2.3: Final review**  Owner: TBD  Est: 1h
  - **[ ] S2.3.1:** Perform a final search for "audacity" to ensure all references are gone.
  - **[ ] S2.3.2:** Review all changes to ensure they are correct and complete.

## 4. Timeline and Milestones

| ID   | Task/Milestone                               | Dependency IDs | Estimated Duration |
|------|----------------------------------------------|----------------|--------------------|
| M1   | Code Cleanup Complete                        | E1             | 3.5h               |
| M2   | Verification and Finalization Complete       | E2             | 2h                 |
| M3   | Project Complete                             | M1, M2         | 5.5h               |

- **Milestone 1: Code Cleanup Complete**
  - **Exit Criteria:** All tasks under Epic E1 are marked as complete. All "audacity" references in the codebase are removed.
- **Milestone 2: Verification and Finalization Complete**
  - **Exit Criteria:** All tasks under Epic E2 are marked as complete. The build is successful, and all tests and linters pass.
- **Milestone 3: Project Complete**
  - **Exit Criteria:** All work is done, and the project is ready for deployment.

## 5. Operating Procedure

- **Definition of Done:** A task is done when the code is updated, unit tests are added or updated, all tests pass, and the code is linted and formatted.
- **Review and QA:** All changes must be reviewed by a second pair of eyes.
- **Testing:** Always add tests when adding new implementation code.
- **Linting/Formatting:** Always run relevant linters and formatters after code changes.
- **Commits:** Do not commit files from different directories in the same commit. Make small, logical commits.

## 6. Progress Log

- **2025 11 17:** Plan created to remove all references to "audacity" from the codebase. No progress yet.

## 7. Hand-off Notes

- This plan is self-contained. A new person should be able to pick up this work and complete it without prior context.
- No special credentials or links are required.

## 8. Appendix

- N/A
