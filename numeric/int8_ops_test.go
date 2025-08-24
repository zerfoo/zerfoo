package numeric

import "testing"

func TestInt8Ops_Add(t *testing.T) {
	ops := Int8Ops{}
	if ops.Add(1, 2) != 3 {
		t.Errorf("1 + 2 should be 3")
	}
}

func TestInt8Ops_Sub(t *testing.T) {
	ops := Int8Ops{}
	if ops.Sub(2, 1) != 1 {
		t.Errorf("2 - 1 should be 1")
	}
}
