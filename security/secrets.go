package security

import (
	"context"
	"errors"
	"os"
	"strings"
)

// KMS is the interface for external key management systems.
// Implementations wrap services like AWS KMS, GCP Cloud KMS, or HashiCorp Vault.
type KMS interface {
	// Encrypt encrypts plaintext using the named key.
	Encrypt(ctx context.Context, keyID string, plaintext []byte) ([]byte, error)
	// Decrypt decrypts ciphertext using the named key.
	Decrypt(ctx context.Context, keyID string, ciphertext []byte) ([]byte, error)
	// RotateKey triggers key rotation for the named key.
	RotateKey(ctx context.Context, keyID string) error
}

// SecretConfig loads configuration values securely from environment variables.
type SecretConfig struct {
	prefix string
	values map[string]string
}

// NewSecretConfig creates a SecretConfig that reads environment variables
// with the given prefix (e.g., "ZERFOO_" reads ZERFOO_DB_PASSWORD as "DB_PASSWORD").
func NewSecretConfig(prefix string) *SecretConfig {
	sc := &SecretConfig{
		prefix: prefix,
		values: make(map[string]string),
	}
	for _, env := range os.Environ() {
		k, v, ok := strings.Cut(env, "=")
		if ok && strings.HasPrefix(k, prefix) {
			key := strings.TrimPrefix(k, prefix)
			sc.values[key] = v
		}
	}
	return sc
}

// Get returns the value for the given key (without prefix).
func (sc *SecretConfig) Get(key string) (string, error) {
	v, ok := sc.values[key]
	if !ok {
		return "", errors.New("security: secret not found: " + key)
	}
	return v, nil
}

// MustGet returns the value for the given key or panics if not found.
func (sc *SecretConfig) MustGet(key string) string {
	v, err := sc.Get(key)
	if err != nil {
		panic(err)
	}
	return v
}

// Has reports whether the given key exists.
func (sc *SecretConfig) Has(key string) bool {
	_, ok := sc.values[key]
	return ok
}

// Keys returns all available secret keys (without prefix).
func (sc *SecretConfig) Keys() []string {
	keys := make([]string, 0, len(sc.values))
	for k := range sc.values {
		keys = append(keys, k)
	}
	return keys
}
