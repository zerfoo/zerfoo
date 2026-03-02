package distributed

import "testing"

// GenerateTestCerts is an exported alias of generateTestCerts for use in
// external tests (e.g. distributed_test).
var GenerateTestCerts = generateTestCerts

// WritePEM is an exported alias of writePEM for use in external tests.
var WritePEM = writePEM

// GenerateTestCertsFunc exposes the signature for documentation.
type GenerateTestCertsFunc = func(t *testing.T, dir string) (caCertPath, serverCertPath, serverKeyPath string)
