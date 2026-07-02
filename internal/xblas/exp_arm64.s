#include "textflag.h"

// func VexpF32(out, x *float32, n int)
//
// Computes out[i] = exp(x[i]) using NEON with range-reduced degree-5 polynomial.
// Input is clamped to [-87, 88] to avoid ldexp overflow/underflow.
//
// Algorithm:
//   0. Clamp x to [-87, 88]
//   1. n_int = round(x / ln2)
//   2. r = x - n_int * ln2
//   3. poly = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))  (Horner)
//   4. result = ldexp(poly, n_int) = poly * 2^n_int
//
// Register allocation (vector loop):
//   V0  = input x[i:i+4]
//   V1  = n_int (integer from FCVTNS)
//   V2  = float(n_int)
//   V3  = r (reduced argument)
//   V4,V5 = Horner temporaries
//   V16 = 1/ln2, V17 = ln2
//   V18 = c0 = 1.0, V19 = c1 = 1.0
//   V20 = c2 = 0.5, V21 = c3 = 1/6
//   V22 = c4 = 1/24, V23 = c5 = 1/120
//   V24 = clamp_min = -87.0, V25 = clamp_max = 88.0
//
// Scalar tail uses F26,F27 (avoids callee-saved V8-V15).
//
// Layout: out=0(FP), x=8(FP), n=16(FP)
TEXT ·VexpF32(SB), NOSPLIT, $0-24
	MOVD	out+0(FP), R0
	MOVD	x+8(FP), R1
	MOVD	n+16(FP), R2

	CBZ	R2, exp_done

	// Load constants into NEON registers via GPR broadcast.
	MOVW	$0x3FB8AA3B, R3        // 1/ln2
	VDUP	R3, V16.S4
	MOVW	$0x3F317218, R3        // ln2
	VDUP	R3, V17.S4
	MOVW	$0x3F800000, R3        // c0 = 1.0
	VDUP	R3, V18.S4
	VMOV	V18.B16, V19.B16       // c1 = 1.0
	MOVW	$0x3F000000, R3        // c2 = 0.5
	VDUP	R3, V20.S4
	MOVW	$0x3E2AAAAB, R3        // c3 = 1/6
	VDUP	R3, V21.S4
	MOVW	$0x3D2AAAAB, R3        // c4 = 1/24
	VDUP	R3, V22.S4
	MOVW	$0x3C088889, R3        // c5 = 1/120
	VDUP	R3, V23.S4

	// Clamp constants
	MOVW	$0xC2AE0000, R3        // -87.0f
	VDUP	R3, V24.S4
	MOVW	$0x42B00000, R3        // 88.0f
	VDUP	R3, V25.S4

	CMP	$4, R2
	BLT	exp_tail

exp_loop4:
	// Load 4 input values.
	VLD1.P	16(R1), [V0.S4]

	// Clamp to [-87, 88]
	// FMAX V0.4S, V0.4S, V24.4S (clamp from below)
	WORD	$0x4E38F400
	// FMIN V0.4S, V0.4S, V25.4S (clamp from above)
	WORD	$0x4EB9F400

	// Step 1: n_int = round(x * (1/ln2))
	// FMUL V1.4S, V0.4S, V16.4S
	WORD	$0x6E30DC01
	// FCVTNS V1.4S, V1.4S  (float to int, round nearest)
	WORD	$0x4E21A821

	// Step 2: r = x - float(n_int) * ln2
	// SCVTF V2.4S, V1.4S  (int to float)
	WORD	$0x4E21D822
	// FMUL V2.4S, V2.4S, V17.4S  (float(n_int) * ln2)
	WORD	$0x6E31DC42
	// FSUB V3.4S, V0.4S, V2.4S  (r = x - n_int*ln2)
	WORD	$0x4EA2D403

	// Step 3: Horner's method
	// p = c4 + r*c5
	VMOV	V22.B16, V4.B16
	VFMLA	V23.S4, V3.S4, V4.S4

	// p = c3 + r*p
	VMOV	V21.B16, V5.B16
	VFMLA	V4.S4, V3.S4, V5.S4

	// p = c2 + r*p
	VMOV	V20.B16, V4.B16
	VFMLA	V5.S4, V3.S4, V4.S4

	// p = c1 + r*p
	VMOV	V19.B16, V5.B16
	VFMLA	V4.S4, V3.S4, V5.S4

	// p = c0 + r*p
	VMOV	V18.B16, V4.B16
	VFMLA	V5.S4, V3.S4, V4.S4

	// Step 4: ldexp(poly, n_int) via bit manipulation
	// SHL V1.4S, V1.4S, #23
	WORD	$0x4F375421
	// ADD V4.4S, V4.4S, V1.4S  (add shifted exponent to poly bits)
	WORD	$0x4EA18484

	// Store 4 results.
	VST1.P	[V4.S4], 16(R0)

	SUB	$4, R2, R2
	CMP	$4, R2
	BGE	exp_loop4

exp_tail:
	CBZ	R2, exp_done

exp_scalar:
	// Process one element at a time using scalar float ops.
	// Uses F26,F27 instead of F10,F11 to avoid callee-saved V8-V15.
	FMOVS	(R1), F0

	// Clamp to [-87, 88]
	MOVW	$0xC2AE0000, R3        // -87.0f
	FMOVS	R3, F6
	FMAXS	F6, F0, F0
	MOVW	$0x42B00000, R3        // 88.0f
	FMOVS	R3, F6
	FMINS	F6, F0, F0

	// n = round(x / ln2)
	MOVW	$0x3FB8AA3B, R3
	FMOVS	R3, F6
	FMULS	F0, F6, F1
	// FCVTNS W3, S1
	WORD	$0x1E240023
	// SCVTF S2, W3
	WORD	$0x1E220062

	// r = x - n*ln2
	MOVW	$0x3F317218, R4
	FMOVS	R4, F7
	FMULS	F2, F7, F2
	FSUBS	F2, F0, F3

	// Horner: poly = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
	MOVW	$0x3C088889, R4
	FMOVS	R4, F26                // c5
	MOVW	$0x3D2AAAAB, R4
	FMOVS	R4, F27                // c4
	FMULS	F3, F26, F26           // r*c5
	FADDS	F26, F27, F26          // c4 + r*c5

	MOVW	$0x3E2AAAAB, R4
	FMOVS	R4, F27                // c3
	FMULS	F3, F26, F26
	FADDS	F26, F27, F26

	MOVW	$0x3F000000, R4
	FMOVS	R4, F27                // c2
	FMULS	F3, F26, F26
	FADDS	F26, F27, F26

	MOVW	$0x3F800000, R4
	FMOVS	R4, F27                // c1
	FMULS	F3, F26, F26
	FADDS	F26, F27, F26

	FMOVS	R4, F27                // c0 (same as c1 = 1.0)
	FMULS	F3, F26, F26
	FADDS	F26, F27, F26          // poly

	// ldexp: add n<<23 to poly's float32 bits
	LSL	$23, R3, R5
	FMOVS	F26, R4
	ADD	R5, R4, R4
	FMOVS	R4, F26

	FMOVS	F26, (R0)

	ADD	$4, R0, R0
	ADD	$4, R1, R1
	SUB	$1, R2, R2
	CBNZ	R2, exp_scalar

exp_done:
	RET
