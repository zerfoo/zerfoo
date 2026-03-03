package distributed

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"

	"google.golang.org/grpc/credentials"
)

// TLSConfig holds TLS certificate paths for gRPC connections.
// When nil, plaintext connections are used (for local development).
type TLSConfig struct {
	// CACertPath is the path to the CA certificate for verifying peers.
	CACertPath string
	// CertPath is the path to the server or client certificate.
	CertPath string
	// KeyPath is the path to the private key for the certificate.
	KeyPath string
}

// ServerCredentials returns gRPC transport credentials for a TLS server.
// If tc is nil, returns nil (plaintext mode).
func (tc *TLSConfig) ServerCredentials() (credentials.TransportCredentials, error) {
	if tc == nil {
		return nil, nil
	}

	cert, err := tls.LoadX509KeyPair(tc.CertPath, tc.KeyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load server key pair: %w", err)
	}

	tlsCfg := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
	}

	// If CA cert is provided, enable mutual TLS by requiring and verifying client certs.
	if tc.CACertPath != "" {
		pool, poolErr := loadCAPool(tc.CACertPath)
		if poolErr != nil {
			return nil, poolErr
		}
		tlsCfg.ClientCAs = pool
		tlsCfg.ClientAuth = tls.RequireAndVerifyClientCert
	}

	return credentials.NewTLS(tlsCfg), nil
}

// ClientCredentials returns gRPC transport credentials for a TLS client.
// If tc is nil, returns nil (plaintext mode).
func (tc *TLSConfig) ClientCredentials() (credentials.TransportCredentials, error) {
	if tc == nil {
		return nil, nil
	}

	caCert, err := os.ReadFile(tc.CACertPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read CA cert: %w", err)
	}

	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caCert) {
		return nil, errors.New("failed to parse CA certificate")
	}

	tlsCfg := &tls.Config{
		RootCAs:    pool,
		MinVersion: tls.VersionTLS12,
	}

	// If client cert and key are provided, enable mutual TLS.
	if tc.CertPath != "" && tc.KeyPath != "" {
		cert, certErr := tls.LoadX509KeyPair(tc.CertPath, tc.KeyPath)
		if certErr != nil {
			return nil, fmt.Errorf("failed to load client key pair: %w", certErr)
		}
		tlsCfg.Certificates = []tls.Certificate{cert}
	}

	return credentials.NewTLS(tlsCfg), nil
}

// loadCAPool reads a PEM CA certificate and returns a cert pool.
func loadCAPool(caPath string) (*x509.CertPool, error) {
	caCert, err := os.ReadFile(caPath) //nolint:gosec // caller-provided cert path
	if err != nil {
		return nil, fmt.Errorf("failed to read CA cert: %w", err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caCert) {
		return nil, errors.New("failed to parse CA certificate")
	}
	return pool, nil
}
