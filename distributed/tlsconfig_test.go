package distributed

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// generateTestCerts creates a self-signed CA and a server certificate signed by that CA.
// Returns paths to caCert, serverCert, serverKey.
func generateTestCerts(t *testing.T, dir string) (caCertPath, serverCertPath, serverKeyPath string) {
	t.Helper()

	// Generate CA key and certificate
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate CA key: %v", err)
	}

	caTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{Organization: []string{"Test CA"}},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
	}

	caCertDER, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("create CA cert: %v", err)
	}

	caCertPath = filepath.Join(dir, "ca.pem")
	if err := writePEM(caCertPath, "CERTIFICATE", caCertDER); err != nil {
		t.Fatalf("write CA cert: %v", err)
	}

	caCert, err := x509.ParseCertificate(caCertDER)
	if err != nil {
		t.Fatalf("parse CA cert: %v", err)
	}

	// Generate server key and certificate signed by CA
	serverKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate server key: %v", err)
	}

	serverTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{Organization: []string{"Test Server"}},
		DNSNames:     []string{"localhost"},
		IPAddresses:  []net.IP{net.IPv4(127, 0, 0, 1)},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
	}

	serverCertDER, err := x509.CreateCertificate(rand.Reader, serverTemplate, caCert, &serverKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("create server cert: %v", err)
	}

	serverCertPath = filepath.Join(dir, "server.pem")
	if err := writePEM(serverCertPath, "CERTIFICATE", serverCertDER); err != nil {
		t.Fatalf("write server cert: %v", err)
	}

	serverKeyDER, err := x509.MarshalECPrivateKey(serverKey)
	if err != nil {
		t.Fatalf("marshal server key: %v", err)
	}

	serverKeyPath = filepath.Join(dir, "server-key.pem")
	if err := writePEM(serverKeyPath, "EC PRIVATE KEY", serverKeyDER); err != nil {
		t.Fatalf("write server key: %v", err)
	}

	return caCertPath, serverCertPath, serverKeyPath
}

func writePEM(path, blockType string, data []byte) error {
	f, err := os.Create(path) //nolint:gosec // test helper
	if err != nil {
		return err
	}
	encErr := pem.Encode(f, &pem.Block{Type: blockType, Bytes: data})
	if closeErr := f.Close(); closeErr != nil && encErr == nil {
		return closeErr
	}
	return encErr
}

func TestTLSConfig_Nil(t *testing.T) {
	var tc *TLSConfig

	serverCreds, err := tc.ServerCredentials()
	if err != nil {
		t.Fatalf("nil ServerCredentials: %v", err)
	}
	if serverCreds != nil {
		t.Error("expected nil server credentials for nil TLSConfig")
	}

	clientCreds, err := tc.ClientCredentials()
	if err != nil {
		t.Fatalf("nil ClientCredentials: %v", err)
	}
	if clientCreds != nil {
		t.Error("expected nil client credentials for nil TLSConfig")
	}
}

func TestTLSConfig_ServerCredentials(t *testing.T) {
	dir := t.TempDir()
	caCert, serverCert, serverKey := generateTestCerts(t, dir)

	t.Run("valid server credentials", func(t *testing.T) {
		tc := &TLSConfig{
			CertPath: serverCert,
			KeyPath:  serverKey,
		}
		creds, err := tc.ServerCredentials()
		if err != nil {
			t.Fatalf("ServerCredentials: %v", err)
		}
		if creds == nil {
			t.Error("expected non-nil credentials")
		}
	})

	t.Run("mTLS server credentials", func(t *testing.T) {
		tc := &TLSConfig{
			CACertPath: caCert,
			CertPath:   serverCert,
			KeyPath:    serverKey,
		}
		creds, err := tc.ServerCredentials()
		if err != nil {
			t.Fatalf("ServerCredentials with mTLS: %v", err)
		}
		if creds == nil {
			t.Error("expected non-nil credentials")
		}
	})

	t.Run("invalid cert path", func(t *testing.T) {
		tc := &TLSConfig{
			CertPath: "/nonexistent/cert.pem",
			KeyPath:  serverKey,
		}
		_, err := tc.ServerCredentials()
		if err == nil {
			t.Error("expected error for invalid cert path")
		}
	})

	t.Run("invalid CA cert path", func(t *testing.T) {
		tc := &TLSConfig{
			CACertPath: "/nonexistent/ca.pem",
			CertPath:   serverCert,
			KeyPath:    serverKey,
		}
		_, err := tc.ServerCredentials()
		if err == nil {
			t.Error("expected error for invalid CA cert path")
		}
	})

	t.Run("invalid CA cert content", func(t *testing.T) {
		badCA := filepath.Join(dir, "bad-ca.pem")
		if err := os.WriteFile(badCA, []byte("not a cert"), 0o600); err != nil {
			t.Fatal(err)
		}
		tc := &TLSConfig{
			CACertPath: badCA,
			CertPath:   serverCert,
			KeyPath:    serverKey,
		}
		_, err := tc.ServerCredentials()
		if err == nil {
			t.Error("expected error for invalid CA cert content")
		}
	})
}

func TestTLSConfig_ClientCredentials(t *testing.T) {
	dir := t.TempDir()
	caCert, clientCert, clientKey := generateTestCerts(t, dir)

	t.Run("valid client credentials without mTLS", func(t *testing.T) {
		tc := &TLSConfig{
			CACertPath: caCert,
		}
		creds, err := tc.ClientCredentials()
		if err != nil {
			t.Fatalf("ClientCredentials: %v", err)
		}
		if creds == nil {
			t.Error("expected non-nil credentials")
		}
	})

	t.Run("mTLS client credentials", func(t *testing.T) {
		tc := &TLSConfig{
			CACertPath: caCert,
			CertPath:   clientCert,
			KeyPath:    clientKey,
		}
		creds, err := tc.ClientCredentials()
		if err != nil {
			t.Fatalf("ClientCredentials with mTLS: %v", err)
		}
		if creds == nil {
			t.Error("expected non-nil credentials")
		}
	})

	t.Run("invalid CA cert path", func(t *testing.T) {
		tc := &TLSConfig{
			CACertPath: "/nonexistent/ca.pem",
		}
		_, err := tc.ClientCredentials()
		if err == nil {
			t.Error("expected error for invalid CA cert path")
		}
	})

	t.Run("invalid CA cert content", func(t *testing.T) {
		badCA := filepath.Join(dir, "bad-client-ca.pem")
		if err := os.WriteFile(badCA, []byte("not a cert"), 0o600); err != nil {
			t.Fatal(err)
		}
		tc := &TLSConfig{
			CACertPath: badCA,
		}
		_, err := tc.ClientCredentials()
		if err == nil {
			t.Error("expected error for invalid CA cert content")
		}
	})

	t.Run("invalid client cert path", func(t *testing.T) {
		tc := &TLSConfig{
			CACertPath: caCert,
			CertPath:   "/nonexistent/client.pem",
			KeyPath:    clientKey,
		}
		_, err := tc.ClientCredentials()
		if err == nil {
			t.Error("expected error for invalid client cert path")
		}
	})
}
