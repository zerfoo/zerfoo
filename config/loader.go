// Package config provides file-based configuration loading with validation
// and environment variable overrides.
//
// Load reads a JSON file into a struct. LoadWithEnv additionally applies
// environment variable overrides using the "env" struct tag. Validate
// checks fields with the "validate" struct tag.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"
)

// Load reads a JSON configuration file at path into a value of type T.
func Load[T any](path string) (T, error) {
	var cfg T

	data, err := os.ReadFile(path) //nolint:gosec // path is caller-provided config file
	if err != nil {
		return cfg, fmt.Errorf("config: read %s: %w", path, err)
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("config: parse %s: %w", path, err)
	}

	return cfg, nil
}

// LoadWithEnv reads a JSON configuration file and applies environment
// variable overrides. Fields with an `env:"NAME"` struct tag are overridden
// by the environment variable PREFIX_NAME if set.
//
// Supported field types: string, int, bool.
func LoadWithEnv[T any](path, prefix string) (T, error) {
	cfg, err := Load[T](path)
	if err != nil {
		return cfg, err
	}

	if err := applyEnvOverrides(&cfg, prefix); err != nil {
		return cfg, err
	}

	return cfg, nil
}

// Validate checks struct fields tagged with `validate:"required"`.
// Returns a slice of error messages for each field that fails validation.
// Returns nil if all validations pass.
func Validate(v any) []string {
	var errs []string

	rv := reflect.ValueOf(v)
	rt := rv.Type()

	for i := range rt.NumField() {
		field := rt.Field(i)
		tag := field.Tag.Get("validate")
		if tag == "" {
			continue
		}

		if strings.Contains(tag, "required") {
			if rv.Field(i).IsZero() {
				errs = append(errs, fmt.Sprintf("field %q is required", field.Name))
			}
		}
	}

	return errs
}

// applyEnvOverrides reads environment variables and applies them to struct fields.
func applyEnvOverrides(cfgPtr any, prefix string) error {
	rv := reflect.ValueOf(cfgPtr).Elem()
	rt := rv.Type()

	for i := range rt.NumField() {
		field := rt.Field(i)
		envName := field.Tag.Get("env")
		if envName == "" {
			continue
		}

		fullName := prefix + "_" + envName
		val, ok := os.LookupEnv(fullName)
		if !ok {
			continue
		}

		fv := rv.Field(i)
		switch fv.Kind() {
		case reflect.String:
			fv.SetString(val)
		case reflect.Int, reflect.Int64:
			n, err := strconv.ParseInt(val, 10, 64)
			if err != nil {
				return fmt.Errorf("config: env %s: %w", fullName, err)
			}
			fv.SetInt(n)
		case reflect.Bool:
			b, err := strconv.ParseBool(val)
			if err != nil {
				return fmt.Errorf("config: env %s: %w", fullName, err)
			}
			fv.SetBool(b)
		}
	}

	return nil
}
