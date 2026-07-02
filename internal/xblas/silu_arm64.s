#include "textflag.h"

// func SiLUF32(out, x *float32, n int)
//
// Computes out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
// Uses NEON with range-reduced degree-5 polynomial for exp.
//
// Register allocation (vector loop):
//   V0  = input x[i:i+4]
//   V1  = -x (negated input)
//   V2  = n_int (from FCVTNS)
//   V3  = float(n_int)
//   V4  = r (reduced argument)
//   V5,V6 = Horner temporaries
//   V7  = exp(-x) result
//   V24 = 1 + exp(-x)         (avoids callee-saved V8-V15)
//   V25 = sigmoid = reciprocal (avoids callee-saved V8-V15)
//   V26 = Newton-Raphson temp  (avoids callee-saved V8-V15)
//   V16 = 1/ln2
//   V17 = ln2
//   V18 = c0 = 1.0
//   V19 = c1 = 1.0
//   V20 = c2 = 0.5
//   V21 = c3 = 1/6
//   V22 = c4 = 1/24
//   V23 = c5 = 1/120
//
// Layout: out=0(FP), x=8(FP), n=16(FP)
TEXT ·SiLUF32(SB), NOSPLIT, $0-24
	MOVD	out+0(FP), R0
	MOVD	x+8(FP), R1
	MOVD	n+16(FP), R2

	CBZ	R2, silu_done

	// Load constants.
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

	// Clamp constants for -x input to exp
	MOVW	$0xC2AE0000, R3        // -87.0f
	VDUP	R3, V28.S4
	MOVW	$0x42AE0000, R3        // 87.0f
	VDUP	R3, V29.S4

	CMP	$4, R2
	BLT	silu_tail

silu_loop4:
	// Load 4 input values.
	VLD1.P	16(R1), [V0.S4]

	// Negate: V1 = -x
	// FNEG V1.4S, V0.4S
	WORD	$0x6EA0F801

	// Clamp -x to [-87, 87] for safe exp
	// FMAX V1.4S, V1.4S, V28.4S (clamp from below)
	WORD	$0x4E3CF421
	// FMIN V1.4S, V1.4S, V29.4S (clamp from above)
	WORD	$0x4EBDF421

	// === exp(-x) using V1 as input ===

	// Step 1: n_int = round(-x * (1/ln2))
	// FMUL V2.4S, V1.4S, V16.4S
	WORD	$0x6E30DC22
	// FCVTNS V2.4S, V2.4S
	WORD	$0x4E21A842

	// Step 2: r = -x - float(n_int) * ln2
	// SCVTF V3.4S, V2.4S
	WORD	$0x4E21D843
	// FMUL V3.4S, V3.4S, V17.4S
	WORD	$0x6E31DC63
	// FSUB V4.4S, V1.4S, V3.4S
	WORD	$0x4EA3D424

	// Step 3: Horner's method for polynomial
	// p = c4 + r*c5
	VMOV	V22.B16, V5.B16
	VFMLA	V23.S4, V4.S4, V5.S4

	// p = c3 + r*p
	VMOV	V21.B16, V6.B16
	VFMLA	V5.S4, V4.S4, V6.S4

	// p = c2 + r*p
	VMOV	V20.B16, V5.B16
	VFMLA	V6.S4, V4.S4, V5.S4

	// p = c1 + r*p
	VMOV	V19.B16, V6.B16
	VFMLA	V5.S4, V4.S4, V6.S4

	// p = c0 + r*p
	VMOV	V18.B16, V5.B16
	VFMLA	V6.S4, V4.S4, V5.S4

	// Step 4: ldexp(poly, n_int)
	// SHL V2.4S, V2.4S, #23
	WORD	$0x4F375442
	// ADD V7.4S, V5.4S, V2.4S  (exp(-x) = poly * 2^n_int via bit add)
	WORD	$0x4EA284A7

	// === sigmoid = 1 / (1 + exp(-x)) ===

	// FADD V24.4S, V18.4S, V7.4S  (1.0 + exp(-x))
	// Q=1, sz=0, Rm=V7=00111, Rn=V18=10010, Rd=V24=11000
	WORD	$0x4E27D658

	// FRECPE V25.4S, V24.4S  (approximate 1/V24)
	// Q=1, sz=0, Rn=V24=11000, Rd=V25=11001
	WORD	$0x4EA1DB19

	// Newton-Raphson step 1: V26 = FRECPS(V24, V25), V25 = V25 * V26
	// FRECPS V26.4S, V24.4S, V25.4S
	// Q=1, sz=0, Rm=V25=11001, Rn=V24=11000, Rd=V26=11010
	WORD	$0x4E39FF1A
	// FMUL V25.4S, V25.4S, V26.4S
	// Rm=V26=11010, Rn=V25=11001, Rd=V25=11001
	WORD	$0x6E3ADF39

	// Newton-Raphson step 2: V26 = FRECPS(V24, V25), V25 = V25 * V26
	WORD	$0x4E39FF1A
	WORD	$0x6E3ADF39

	// === silu = x * sigmoid ===
	// FMUL V0.4S, V0.4S, V25.4S
	// Rm=V25=11001, Rn=V0=00000, Rd=V0=00000
	WORD	$0x6E39DC00

	// Store 4 results.
	VST1.P	[V0.S4], 16(R0)

	SUB	$4, R2, R2
	CMP	$4, R2
	BGE	silu_loop4

silu_tail:
	CBZ	R2, silu_done

silu_scalar:
	FMOVS	(R1), F0

	// Negate: F1 = -x
	FNEGS	F0, F1

	// Clamp -x to [-87, 87]
	MOVW	$0xC2AE0000, R3        // -87.0f
	FMOVS	R3, F6
	FMAXS	F6, F1, F1
	MOVW	$0x42AE0000, R3        // 87.0f
	FMOVS	R3, F6
	FMINS	F6, F1, F1

	// exp(-x) scalar
	MOVW	$0x3FB8AA3B, R3
	FMOVS	R3, F6
	FMULS	F1, F6, F2
	// FCVTNS W3, S2
	WORD	$0x1E240043
	// SCVTF S3, W3
	WORD	$0x1E220063

	MOVW	$0x3F317218, R4
	FMOVS	R4, F7
	FMULS	F3, F7, F3
	FSUBS	F3, F1, F4		// r = -x - n*ln2

	// Horner (uses F24, F25 instead of F10, F11 to avoid callee-saved)
	MOVW	$0x3C088889, R4
	FMOVS	R4, F24			// c5
	MOVW	$0x3D2AAAAB, R4
	FMOVS	R4, F25			// c4
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3E2AAAAB, R4
	FMOVS	R4, F25			// c3
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3F000000, R4
	FMOVS	R4, F25			// c2
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3F800000, R4
	FMOVS	R4, F25			// c1
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	FMOVS	R4, F25			// c0 = 1.0
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24		// poly

	// ldexp
	LSL	$23, R3, R5
	FMOVS	F24, R4
	ADD	R5, R4, R4
	FMOVS	R4, F24			// exp(-x)

	// sigmoid = 1 / (1 + exp(-x))
	MOVW	$0x3F800000, R4
	FMOVS	R4, F26			// 1.0
	FADDS	F24, F26, F25		// F25 = 1 + exp(-x)
	FDIVS	F25, F26, F25		// F25 = 1.0 / (1 + exp(-x))

	// silu = x * sigmoid
	FMULS	F0, F25, F0
	FMOVS	F0, (R0)

	ADD	$4, R0, R0
	ADD	$4, R1, R1
	SUB	$1, R2, R2
	CBNZ	R2, silu_scalar

silu_done:
	RET

// func SiLUGateF32(out, gate, up *float32, n int)
//
// Computes out[i] = silu(gate[i]) * up[i] = gate[i] * sigmoid(gate[i]) * up[i]
//
// Layout: out=0(FP), gate=8(FP), up=16(FP), n=24(FP)
TEXT ·SiLUGateF32(SB), NOSPLIT, $0-32
	MOVD	out+0(FP), R0
	MOVD	gate+8(FP), R1
	MOVD	up+16(FP), R6
	MOVD	n+24(FP), R2

	CBZ	R2, silugate_done

	// Load constants.
	MOVW	$0x3FB8AA3B, R3
	VDUP	R3, V16.S4
	MOVW	$0x3F317218, R3
	VDUP	R3, V17.S4
	MOVW	$0x3F800000, R3
	VDUP	R3, V18.S4
	VMOV	V18.B16, V19.B16
	MOVW	$0x3F000000, R3
	VDUP	R3, V20.S4
	MOVW	$0x3E2AAAAB, R3
	VDUP	R3, V21.S4
	MOVW	$0x3D2AAAAB, R3
	VDUP	R3, V22.S4
	MOVW	$0x3C088889, R3
	VDUP	R3, V23.S4

	// Clamp constants for -gate input to exp
	MOVW	$0xC2AE0000, R3
	VDUP	R3, V28.S4
	MOVW	$0x42AE0000, R3
	VDUP	R3, V29.S4

	CMP	$4, R2
	BLT	silugate_tail

silugate_loop4:
	// Load 4 gate values.
	VLD1.P	16(R1), [V0.S4]
	// Load 4 up values.
	VLD1.P	16(R6), [V27.S4]

	// Negate: V1 = -gate
	WORD	$0x6EA0F801

	// Clamp -gate to [-87, 87]
	WORD	$0x4E3CF421
	WORD	$0x4EBDF421

	// === exp(-gate) ===

	// FMUL V2.4S, V1.4S, V16.4S
	WORD	$0x6E30DC22
	// FCVTNS V2.4S, V2.4S
	WORD	$0x4E21A842

	// SCVTF V3.4S, V2.4S
	WORD	$0x4E21D843
	// FMUL V3.4S, V3.4S, V17.4S
	WORD	$0x6E31DC63
	// FSUB V4.4S, V1.4S, V3.4S
	WORD	$0x4EA3D424

	// Horner
	VMOV	V22.B16, V5.B16
	VFMLA	V23.S4, V4.S4, V5.S4

	VMOV	V21.B16, V6.B16
	VFMLA	V5.S4, V4.S4, V6.S4

	VMOV	V20.B16, V5.B16
	VFMLA	V6.S4, V4.S4, V5.S4

	VMOV	V19.B16, V6.B16
	VFMLA	V5.S4, V4.S4, V6.S4

	VMOV	V18.B16, V5.B16
	VFMLA	V6.S4, V4.S4, V5.S4

	// ldexp
	WORD	$0x4F375442
	WORD	$0x4EA284A7		// V7 = exp(-gate)

	// sigmoid = 1 / (1 + exp(-gate))
	// FADD V24.4S, V18.4S, V7.4S
	WORD	$0x4E27D658
	// FRECPE V25.4S, V24.4S
	WORD	$0x4EA1DB19
	// FRECPS V26.4S, V24.4S, V25.4S
	WORD	$0x4E39FF1A
	// FMUL V25.4S, V25.4S, V26.4S
	WORD	$0x6E3ADF39
	// FRECPS V26.4S, V24.4S, V25.4S
	WORD	$0x4E39FF1A
	// FMUL V25.4S, V25.4S, V26.4S
	WORD	$0x6E3ADF39

	// silu(gate) = gate * sigmoid
	// FMUL V0.4S, V0.4S, V25.4S
	WORD	$0x6E39DC00

	// result = silu(gate) * up
	// FMUL V0.4S, V0.4S, V27.4S
	// Rm=V27=11011, Rn=V0=00000, Rd=V0=00000
	WORD	$0x6E3BDC00

	// Store 4 results.
	VST1.P	[V0.S4], 16(R0)

	SUB	$4, R2, R2
	CMP	$4, R2
	BGE	silugate_loop4

silugate_tail:
	CBZ	R2, silugate_done

silugate_scalar:
	FMOVS	(R1), F0		// gate
	FMOVS	(R6), F26		// up (was F12, now F26 to avoid V8-V15)

	// exp(-gate) scalar
	FNEGS	F0, F1

	// Clamp -gate to [-87, 87]
	MOVW	$0xC2AE0000, R3
	FMOVS	R3, F6
	FMAXS	F6, F1, F1
	MOVW	$0x42AE0000, R3
	FMOVS	R3, F6
	FMINS	F6, F1, F1

	MOVW	$0x3FB8AA3B, R3
	FMOVS	R3, F6
	FMULS	F1, F6, F2
	// FCVTNS W3, S2
	WORD	$0x1E240043
	// SCVTF S3, W3
	WORD	$0x1E220063

	MOVW	$0x3F317218, R4
	FMOVS	R4, F7
	FMULS	F3, F7, F3
	FSUBS	F3, F1, F4

	// Horner (uses F24, F25 to avoid callee-saved)
	MOVW	$0x3C088889, R4
	FMOVS	R4, F24
	MOVW	$0x3D2AAAAB, R4
	FMOVS	R4, F25
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3E2AAAAB, R4
	FMOVS	R4, F25
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3F000000, R4
	FMOVS	R4, F25
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3F800000, R4
	FMOVS	R4, F25
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	FMOVS	R4, F25
	FMULS	F4, F24, F24
	FADDS	F24, F25, F24

	// ldexp
	LSL	$23, R3, R5
	FMOVS	F24, R4
	ADD	R5, R4, R4
	FMOVS	R4, F24			// exp(-gate)

	// sigmoid = 1 / (1 + exp(-gate))
	MOVW	$0x3F800000, R4
	FMOVS	R4, F27
	FADDS	F24, F27, F25		// 1 + exp(-gate)
	FDIVS	F25, F27, F25		// 1.0 / (1 + exp(-gate))

	// silu = gate * sigmoid
	FMULS	F0, F25, F0
	// result = silu * up
	FMULS	F0, F26, F0

	FMOVS	F0, (R0)

	ADD	$4, R0, R0
	ADD	$4, R1, R1
	ADD	$4, R6, R6
	SUB	$1, R2, R2
	CBNZ	R2, silugate_scalar

silugate_done:
	RET
