package security

import (
	"testing"
)

func TestSecretConfigFromEnv(t *testing.T) {
	t.Setenv("ZFTEST_DB_PASSWORD", "s3cret")
	t.Setenv("ZFTEST_API_KEY", "abc123")
	t.Setenv("OTHER_VAR", "ignored")

	sc := NewSecretConfig("ZFTEST_")

	v, err := sc.Get("DB_PASSWORD")
	if err != nil {
		t.Fatal(err)
	}
	if v != "s3cret" {
		t.Fatalf("expected s3cret, got %s", v)
	}

	v, err = sc.Get("API_KEY")
	if err != nil {
		t.Fatal(err)
	}
	if v != "abc123" {
		t.Fatalf("expected abc123, got %s", v)
	}

	_, err = sc.Get("OTHER_VAR")
	if err == nil {
		t.Fatal("expected error for non-prefixed var")
	}
}

func TestSecretConfigHas(t *testing.T) {
	t.Setenv("TST_KEY", "val")
	sc := NewSecretConfig("TST_")
	if !sc.Has("KEY") {
		t.Fatal("expected Has to return true")
	}
	if sc.Has("MISSING") {
		t.Fatal("expected Has to return false")
	}
}

func TestSecretConfigMustGetPanics(t *testing.T) {
	sc := NewSecretConfig("NONEXIST_")
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	sc.MustGet("MISSING")
}

func TestSecretConfigKeys(t *testing.T) {
	t.Setenv("KT_A", "1")
	t.Setenv("KT_B", "2")
	sc := NewSecretConfig("KT_")

	keys := sc.Keys()
	if len(keys) < 2 {
		t.Fatalf("expected at least 2 keys, got %d", len(keys))
	}
}
