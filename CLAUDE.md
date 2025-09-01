# Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive context window. Use `gemini -p` to leverage Google Gemini's large context capacity for comprehensive analysis that would be impossible with smaller context windows.

## File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the gemini command:

### Examples:

**Single file analysis:**
```bash
gemini -p "@tensor/tensor.go Explain this file's purpose and structure"
```

**Multiple files:**
```bash
gemini -p "@go.mod @go.sum Analyze the dependencies used in the code"
```

**Entire directory:**
```bash
gemini -p "@compute/ Summarize the architecture of the compute package"
```

**Multiple directories:**
```bash
gemini -p "@layers/ @training/ Analyze how layers and training are coupled"
```

**Current directory and subdirectories:**
```bash
gemini -p "@./ Give me an overview of this entire project"
```

Or use `--all_files` flag:
```bash
gemini --all_files -p "Analyze the project structure and dependencies"
```

## Implementation Verification Examples

**Check if a feature is implemented:**
```bash
gemini -p "@layers/ @compute/ Has softmax been implemented? Show me the relevant files and functions"
```

**Verify model format support:**
```bash
gemini -p "@model/ Is ZMF model format supported for loading? List all ZMF-related functions"
```

**Check for specific patterns:**
```bash
gemini -p "@compute/ @internal/ Are there any custom BLAS/GEMM implementations? List them with file paths"
```

**Verify error handling:**
```bash
gemini -p "@tensor/ Is proper error handling implemented for all tensor operations? Show examples of error returns"
```

**Check for training features:**
```bash
gemini -p "@training/optimizer/ Is gradient clipping implemented? Show the implementation details"
```

**Verify memory management strategy:**
```bash
gemini -p "@device/ Is there a custom memory allocator? List all allocator-related functions and their usage"
```

**Check for data validation:**
```bash
gemini -p "@data/ How are datasets validated? Show how data integrity is maintained"
```

**Verify test coverage for features:**
```bash
gemini -p "@layers/attention/ @tests/ Is the attention module fully tested? List all test cases"
```

## Advanced Analysis Patterns

**Cross-package dependency analysis:**
```bash
gemini -p "@./ Analyze all import dependencies across packages. Identify circular dependencies and suggest refactoring opportunities"
```

**Architecture and design pattern analysis:**
```bash
gemini -p "@layers/ @graph/ @model/ Analyze the overall architecture. How do layers, graphs, and models interact? Identify design patterns used"
```

**Performance bottleneck identification:**
```bash
gemini -p "@compute/ @tensor/ @layers/ Identify potential performance bottlenecks in tensor operations and layer implementations. Suggest optimization opportunities"
```

**Code quality and technical debt assessment:**
```bash
gemini -p "@./ Analyze code quality across the entire codebase. Identify areas with high complexity, code duplication, or technical debt"
```

**Security vulnerability scanning:**
```bash
gemini -p "@./ Scan for potential security vulnerabilities including unsafe operations, input validation gaps, and data exposure risks"
```

## Quantization and ONNX-Specific Analysis

**ONNX operator coverage analysis:**
```bash
gemini -p "@layers/ @model/ ../zonnx/ Analyze current ONNX operator support. List implemented vs missing operators needed for Gemma 3"
```

**Quantization support assessment:**
```bash
gemini -p "@tensor/ @numeric/ @compute/ Assess current quantization support. What data types and quantization schemes are implemented?"
```

**Model import pipeline analysis:**
```bash
gemini -p "@model/ ../zmf/ ../zonnx/ Analyze the complete model import pipeline from ONNX to Zerfoo. Identify bottlenecks and missing components"
```

**MatMul implementation analysis:**
```bash
gemini -p "@layers/core/ @compute/ How are matrix multiplication operations currently implemented? Are there quantized variants?"
```

## Testing and Validation Workflows

**Comprehensive test coverage analysis:**
```bash
gemini -p "@./ Analyze test coverage across all packages. Which areas lack sufficient testing? Generate a prioritized list for test improvement"
```

**Parity testing framework analysis:**
```bash
gemini -p "@tests/parity/ @tests/internal/ How is the parity testing framework structured? What's needed to add new parity tests?"
```

**CI/CD pipeline analysis:**
```bash
gemini -p "@.github/ @scripts/ Analyze the CI/CD setup. What tests run in CI? Are there gaps in automated testing?"
```

**Performance regression detection:**
```bash
gemini -p "@tests/ @.github/ Is there performance regression testing? How are benchmarks tracked over time?"
```

## Development Planning and Task Breakdown

**Feature gap analysis for specific models:**
```bash
gemini -p "@./ ../gemma3/ What's missing to fully support Gemma 3? Provide detailed implementation roadmap with effort estimates"
```

**Code generation planning:**
```bash
gemini -p "@layers/ Based on existing layer patterns, generate implementation plan for missing operators: MatMulNBits, Constant, standard Gelu"
```

**Refactoring opportunity identification:**
```bash
gemini -p "@./ Identify refactoring opportunities to improve code maintainability, reduce duplication, and enhance testability"
```

**Technical debt prioritization:**
```bash
gemini -p "@./ Analyze technical debt across the codebase. Prioritize items by impact on correctness, maintainability, and performance"
```

## Complex Multi-Repository Analysis

**Cross-repository integration analysis:**
```bash
gemini -p "@./ ../zmf/ ../zonnx/ ../gemma3/ Analyze how zerfoo, zmf, zonnx, and gemma3 repositories integrate. What are the data flow patterns and dependencies?"
```

**Model format compatibility across repos:**
```bash
gemini -p "@model/ ../zmf/ ../zonnx/ How do ZMF, ONNX, and zerfoo model formats relate? Map the conversion pipeline and identify gaps"
```

**Quantization implementation status across ecosystem:**
```bash
gemini -p "@./ ../zmf/ ../zonnx/ What quantization support exists across the zerfoo ecosystem? Which components need UINT8 and 4-bit support?"
```

**API consistency analysis:**
```bash
gemini -p "@./ ../zmf/ ../zonnx/ Analyze API consistency across zerfoo ecosystem. Are there naming conflicts or design inconsistencies to address?"
```

## Large-Scale Code Generation and Planning

**Generate comprehensive implementation roadmap:**
```bash
gemini -p "@./ ../zonnx/ ../zmf/ Based on the existing codebase patterns, generate a complete implementation plan for UINT8 tensor support and MatMulNBits operator. Include file-by-file changes needed"
```

**End-to-end workflow planning:**
```bash
gemini -p "@./ ../gemma3/ ../zonnx/ ../zmf/ Plan the complete workflow for Gemma 3 quantized inference from ONNX download to final predictions. Include all required components"
```

**Test suite planning:**
```bash
gemini -p "@tests/ @./ Generate a comprehensive test plan for quantization features. Include unit tests, integration tests, and parity tests needed"
```

**Documentation generation planning:**
```bash
gemini -p "@./ Analyze the current documentation structure. Generate a plan for comprehensive documentation including tutorials, API docs, and design documents"
```

## Debugging and Problem Solving

**Root cause analysis for build failures:**
```bash
gemini -p "@./ Analyze the codebase for potential causes of build failure: 'unsupported tensor dtype: UINT8'. Trace through tensor creation and usage"
```

**Interface compatibility debugging:**
```bash
gemini -p "@layers/ @graph/ Debug interface compatibility issues. Why might layer registration fail? Analyze the type constraints and interfaces"
```

**Memory leak investigation:**
```bash
gemini -p "@device/ @tensor/ @compute/ Investigate potential memory leaks in tensor operations. Analyze allocation patterns and cleanup logic"
```

**Performance regression analysis:**
```bash
gemini -p "@./ ../gemma3/ Compare current performance characteristics with expected Gemma 3 inference speeds. Identify regression sources"
```

## Code Quality and Maintenance

**Comprehensive refactoring analysis:**
```bash
gemini -p "@./ Identify all opportunities for code refactoring to improve maintainability. Prioritize by impact and effort required"
```

**Dead code elimination:**
```bash
gemini -p "@./ Identify unused functions, variables, and imports across the entire codebase. Generate cleanup recommendations"
```

**Naming consistency audit:**
```bash
gemini -p "@./ Audit naming consistency across packages. Identify inconsistent naming patterns and suggest standardization"
```

**Error handling pattern analysis:**
```bash
gemini -p "@./ Analyze error handling patterns across the codebase. Are there inconsistencies or missing error cases?"
```

## When to Use Gemini CLI

Use `gemini -p` when:
- Analyzing entire codebases or large directories (>100KB total)
- Comparing multiple large files or repositories simultaneously
- Need to understand project-wide patterns, architecture, or dependencies
- Current context window is insufficient for comprehensive analysis
- Performing cross-repository analysis of the zerfoo ecosystem
- Generating detailed implementation plans that require understanding entire codebase
- Debugging complex issues that span multiple packages or repositories
- Conducting comprehensive code quality assessments
- Planning large-scale refactoring or feature implementation
- Analyzing quantization support across the entire tensor/compute/model stack
- Understanding ONNX import pipeline that spans multiple repositories

## Strategic Decision Making

**Technology stack evaluation:**
```bash
gemini -p "@./ Evaluate the current technology stack. Are there better alternatives for tensor operations, model formats, or testing frameworks?"
```

**Scalability assessment:**
```bash
gemini -p "@./ Assess scalability limitations in the current architecture. What changes are needed to support larger models or datasets?"
```

**Ecosystem integration opportunities:**
```bash
gemini -p "@./ ../zmf/ ../zonnx/ ../gemma3/ Identify opportunities to better integrate the zerfoo ecosystem components. What APIs or interfaces could be standardized?"
```

## Important Notes

- **Context Efficiency:** Paths in `@` syntax are relative to your current working directory when invoking `gemini`
- **Massive Context:** Gemini's context window can handle entire codebases (multiple GB) that would overflow other models
- **Multi-Repo Analysis:** Gemini excels at cross-repository analysis using relative paths like `../zmf/`
- **Specificity:** When checking implementations, be specific about what you're looking for to get accurate results
- **No Modification:** Gemini CLI is read-only - use it for analysis and planning, then implement with write tools
- **Performance:** For complex analysis spanning multiple repos, Gemini CLI is significantly faster than sequential file reading
- **Comprehensive Output:** Ask for detailed file paths, line numbers, and specific recommendations to maximize value