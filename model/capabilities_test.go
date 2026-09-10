package model_test

import (
	"testing"

	_ "github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
)

func TestComponentRegistryOwnsMetadata(t *testing.T) {
	first, rule, ok := model.Component("operator", "Dense")
	if !ok || rule == nil {
		t.Fatal("missing executable contract")
	}
	first.Inputs[0] = "changed"
	second, _, ok := model.Component("operator", "Dense")
	if !ok || second.Inputs[0] != "x" {
		t.Fatal("caller changed shared metadata")
	}
}
func TestComponentRegistryDoesNotQualifyUnannotatedBuilders(t *testing.T) {
	found := false
	for _, component := range model.ListComponents() {
		if component.ID == "Conv" {
			found = true
			for _, support := range component.Support {
				if support.Status == "verified" {
					t.Fatal("registration invented qualification")
				}
			}
		}
	}
	if !found {
		t.Fatal("registry hid existing unannotated operator")
	}
}

func TestAttributeSchemaValidation(t *testing.T) {
	low, high := 1.0, 16.0
	d := model.ComponentDescriptor{Attributes: map[string]model.AttributeSpec{
		"width":   {Type: "integer", Required: true, Minimum: &low, Maximum: &high},
		"enabled": {Type: "boolean"},
	}}
	valid, err := model.ValidateAttributes(d, map[string]any{"width": float64(4), "enabled": true})
	if err != nil || valid["width"] != 4 {
		t.Fatalf("normalization: %v, %v", valid, err)
	}
	for _, attrs := range []map[string]any{{}, {"width": 0}, {"width": 17}, {"width": 1.5}, {"width": "4"}, {"width": 4, "unknown": true}, {"width": 4, "enabled": "true"}} {
		if _, err := model.ValidateAttributes(d, attrs); err == nil {
			t.Fatalf("accepted invalid attributes %v", attrs)
		}
	}
}
