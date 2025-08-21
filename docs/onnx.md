# ONNX Import Plan for Gemma 3

This document outlines the plan to import the Google Gemma 3 model into the Zerfoo ecosystem. The primary goal is to successfully load a pre-trained Gemma 3 model from the ONNX format and run inference within the `zerfoo` framework.

The core strategy is to use an intermediary format, the Zerfoo Model Format (ZMF), and a separate standalone converter tool, `zonnx`, to translate the Gemma 3 ONNX file into a format that `zerfoo` can ingest.

---

### Phase 1: Implement the ZMF Importer in `zerfoo`

The first step is to complete the ZMF importer to ensure the framework can deserialize a model graph. This is a prerequisite for loading the converted Gemma 3 model.

- [ ] **Complete the `LoadModel` Logic:**
    - [ ] Finish the `LoadModel` function in `pkg/importer/importer.go`.
    - [ ] The logic must iterate through the ZMF graph nodes in the protobuf.
    - [ ] For each node, it must use the layer registry to look up and call the corresponding layer constructor.
    - [ ] The final result should be a fully constructed, executable computation graph (`graph.Graph`).

- [ ] **Register All Existing `zerfoo` Layers:**
    - [ ] Ensure that every layer currently implemented in `zerfoo` (e.g., `RMSNorm`, `Dense`, `GlobalAttention`, `TokenEmbedding`) is registered with the importer's registry. This is critical for reconstructing the graph.

- [ ] **Generalize Model Loading:**
    - [ ] Refactor the `LoadModel` function to return a generic `graph.Graph` interface, not a hardcoded `gemma.Model`, to support various architectures.

---

### Phase 2: Develop the `zonnx` Converter for Gemma 3

This phase focuses on building the `zonnx` command-line tool with a specific focus on the operators required by Gemma 3.

- [ ] **Set up `zerfoo/zonnx` Repository:**
    - [ ] Initialize the Go project with the `import`/`export` CLI structure using `cobra`.
    - [ ] Add necessary dependencies: `github.com/zerfoo/zerfoo` (for ZMF) and a suitable ONNX library.

- [ ] **Implement ONNX-to-ZMF Conversion Logic for Gemma 3:**
    - [ ] Use an ONNX library to load and parse the Gemma 3 `.onnx` file.
    - [ ] **Prioritize building a mapping for the core Gemma 3 operators.** Based on analysis, these include:
        - `MatMul`, `Add`, `Mul`, `Div` (Arithmetic)
        - `Softmax` (Activation)
        - `Reshape`, `Transpose`, `Concat`, `Slice`, `Gather`, `Unsqueeze` (Tensor Manipulation)
        - `EmbedLayerNormalization`, `Cast`, `Shape`
    - [ ] Traverse the ONNX graph and convert only the required nodes and initializers into the ZMF `format.Node` and `format.Tensor` protobufs.
    - [ ] Write the logic to serialize and save the resulting model to a `.zmf` file.

- [ ] **Develop CLI Interface:**
    - [ ] Implement the user-friendly `import` and `export` commands as designed in `zonnx/docs/zonnx.md`.
    - [ ] Example command: `zonnx import gemma-3.onnx --output gemma-3.zmf`

---

### Phase 3: Validation with Gemma 3

Thorough testing is critical to ensure the correctness of the conversion and import process. The entire focus of this phase is to successfully load and validate the Gemma 3 model.

- [ ] **Acquire the Gemma 3 ONNX Model:**
    - [ ] Download the official or a community-verified ONNX version of Gemma 3.
    - [ ] Store it in a location accessible for testing.

- [ ] **Create the End-to-End Import Test:**
    - [ ] **Step 1: Convert.** Use the `zonnx` tool to convert the `gemma-3.onnx` file into a `gemma-3.zmf` file.
    - [ ] **Step 2: Load.** In a Go test within the `zerfoo` repository, use the `importer.LoadModel` function to load `gemma-3.zmf`.
    - [ ] **Step 3: Verify.**
        - Verify that the model loads without errors.
        - Check that the reconstructed graph has the expected number of layers and parameters.
        - Perform a sanity check by running a single forward pass with a sample input and ensure it produces an output tensor of the correct shape without panicking.

- [ ] **Add Unit Tests for Gemma 3 Operators:**
    - [ ] Add specific unit tests to the `zonnx` tool for each Gemma 3 operator converter (e.g., test that an ONNX `MatMul` is correctly converted to a ZMF `Dense` or equivalent).

---

### Phase 4: Documentation and Follow-up

- [ ] **Document the Gemma 3 Import Process:**
    - [ ] Write a `README.md` for the `zerfoo/zonnx` repository with clear instructions on how to convert the Gemma 3 model.
    - [ ] Add a tutorial to the `zerfoo` documentation showing how to load and use the converted Gemma 3 model.

- [ ] **Future Work (Post-Gemma):**
    - [ ] Implement the ZMF Exporter in `zerfoo`.
    - [ ] Implement the ZMF-to-ONNX (`export`) functionality in `zonnx`.
    - [ ] Expand operator support in `zonnx` to cover models beyond Gemma.
    - [ ] Set up a full CI/CD pipeline for the `zonnx` tool.