package numeric

import "testing"

func TestUint8Ops_Add(t *testing.T) {
	ops := Uint8Ops{}
	if ops.Add(1, 2) != 3 {
		t.Errorf("1 + 2 should be 3")
	}
	if ops.Add(250, 5) != 255 {
		t.Errorf("250 + 5 should be 255")
	}
}

func TestUint8Ops_Sub(t *testing.T) {
	ops := Uint8Ops{}
	if ops.Sub(2, 1) != 1 {
		t.Errorf("2 - 1 should be 1")
	}
	if ops.Sub(10, 5) != 5 {
		t.Errorf("10 - 5 should be 5")
	}
}

func TestUint8Ops_Mul(t *testing.T) {
	ops := Uint8Ops{}
	if ops.Mul(2, 3) != 6 {
		t.Errorf("2 * 3 should be 6")
	}
	if ops.Mul(10, 10) != 100 {
		t.Errorf("10 * 10 should be 100")
	}
}

func TestUint8Ops_Div(t *testing.T) {
	ops := Uint8Ops{}
	if ops.Div(6, 3) != 2 {
		t.Errorf("6 / 3 should be 2")
	}
	if ops.Div(100, 10) != 10 {
		t.Errorf("100 / 10 should be 10")
	}
	if ops.Div(5, 0) != 0 {
		t.Errorf("5 / 0 should be 0 to prevent panic")
	}
}

func TestUint8Ops_GreaterThan(t *testing.T) {
	ops := Uint8Ops{}
	if !ops.GreaterThan(5, 3) {
		t.Errorf("5 > 3 should be true")
	}
	if ops.GreaterThan(3, 5) {
		t.Errorf("3 > 5 should be false")
	}
	if ops.GreaterThan(5, 5) {
		t.Errorf("5 > 5 should be false")
	}
}
