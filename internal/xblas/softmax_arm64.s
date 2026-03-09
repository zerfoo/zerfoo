#include "textflag.h"

// func SoftmaxF32(data *float32, n int)
//
// Computes softmax(x) in-place using 3-pass NEON:
//   Pass 1: max = max(x[i])
//   Pass 2: x[i] = exp(x[i] - max), sum += x[i]
//   Pass 3: x[i] *= 1/sum
//
// Uses the same degree-5 Horner exp polynomial as VexpF32.
// All registers avoid callee-saved V8-V15.
//
// Layout: data=0(FP), n=8(FP)
TEXT ·SoftmaxF32(SB), NOSPLIT, $0-16
	MOVD	data+0(FP), R0
	MOVD	n+8(FP), R1

	CBZ	R1, sm_done

	// ========== PASS 1: Find max ==========
	// Initialize max to -Inf (0xFF800000)
	MOVW	$0xFF800000, R3
	VDUP	R3, V30.S4             // V30 = max accumulator (4-wide)
	FMOVS	R3, F31                // F31 = scalar max

	MOVD	R0, R4                 // R4 = cursor
	MOVD	R1, R5                 // R5 = remaining count

	CMP	$4, R5
	BLT	max_tail

max_loop4:
	VLD1.P	16(R4), [V0.S4]
	// FMAX V30.4S, V30.4S, V0.4S
	WORD	$0x4E20F7DE
	SUB	$4, R5, R5
	CMP	$4, R5
	BGE	max_loop4

max_tail:
	CBZ	R5, max_reduce

max_scalar:
	FMOVS	(R4), F0
	FMAXS	F0, F31, F31
	ADD	$4, R4, R4
	SUB	$1, R5, R5
	CBNZ	R5, max_scalar

max_reduce:
	// Reduce V30.4S to scalar max in F31
	// FMAXV S0, V30.4S
	WORD	$0x6E30F800
	FMAXS	F0, F31, F31
	// Now F31 = global max

	// ========== PASS 2: exp(x[i] - max), accumulate sum ==========
	// Load exp constants
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

	// Broadcast max into V24.S4 for vector subtract
	WORD	$0x4E0407F8     // DUP V24.4S, V31.S[0]

	// Clamp constant: min = -87.0f for exp input
	MOVW	$0xC2AE0000, R3
	VDUP	R3, V26.S4

	// V29 = sum accumulator (zero)
	VEOR	V29.B16, V29.B16, V29.B16

	MOVD	R0, R4                 // R4 = cursor (restart)
	MOVD	R1, R5                 // R5 = remaining count
	FMOVS	ZR, F28                // F28 = scalar sum accumulator

	CMP	$4, R5
	BLT	exp_tail

exp_loop4:
	VLD1	(R4), [V0.S4]

	// V0 = x[i] - max
	// FSUB V0.4S, V0.4S, V24.4S
	WORD	$0x4EB8D400

	// Clamp (x-max) to min -87 for safe exp
	// FMAX V0.4S, V0.4S, V26.4S
	WORD	$0x4E3AF400

	// --- Inline exp polynomial (same as VexpF32) ---
	// Step 1: n_int = round((x-max) * (1/ln2))
	// FMUL V1.4S, V0.4S, V16.4S
	WORD	$0x6E30DC01
	// FCVTNS V1.4S, V1.4S
	WORD	$0x4E21A821

	// Step 2: r = (x-max) - float(n_int) * ln2
	// SCVTF V2.4S, V1.4S
	WORD	$0x4E21D822
	// FMUL V2.4S, V2.4S, V17.4S
	WORD	$0x6E31DC42
	// FSUB V3.4S, V0.4S, V2.4S
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

	// Step 4: ldexp(poly, n_int)
	// SHL V1.4S, V1.4S, #23
	WORD	$0x4F375421
	// ADD V4.4S, V4.4S, V1.4S
	WORD	$0x4EA18484

	// Store exp results back
	VST1	[V4.S4], (R4)

	// Accumulate sum: V29 += V4
	// FADD V29.4S, V29.4S, V4.4S
	WORD	$0x4E24D7BD

	ADD	$16, R4, R4
	SUB	$4, R5, R5
	CMP	$4, R5
	BGE	exp_loop4

exp_tail:
	CBZ	R5, exp_reduce

exp_scalar:
	FMOVS	(R4), F0
	FSUBS	F31, F0, F0           // x - max

	// Clamp (x-max) to min -87
	MOVW	$0xC2AE0000, R3
	FMOVS	R3, F6
	FMAXS	F6, F0, F0

	// Scalar exp using F24,F25 (avoids callee-saved V8-V15)
	MOVW	$0x3FB8AA3B, R3
	FMOVS	R3, F6
	FMULS	F0, F6, F1            // x * (1/ln2)
	// FCVTNS W3, S1
	WORD	$0x1E240023
	// SCVTF S2, W3
	WORD	$0x1E220062

	MOVW	$0x3F317218, R6
	FMOVS	R6, F7
	FMULS	F2, F7, F2            // n*ln2
	FSUBS	F2, F0, F3            // r = x - n*ln2

	// Horner
	MOVW	$0x3C088889, R6
	FMOVS	R6, F24                // c5
	MOVW	$0x3D2AAAAB, R6
	FMOVS	R6, F25                // c4
	FMULS	F3, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3E2AAAAB, R6
	FMOVS	R6, F25                // c3
	FMULS	F3, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3F000000, R6
	FMOVS	R6, F25                // c2
	FMULS	F3, F24, F24
	FADDS	F24, F25, F24

	MOVW	$0x3F800000, R6
	FMOVS	R6, F25                // c1
	FMULS	F3, F24, F24
	FADDS	F24, F25, F24

	FMOVS	R6, F25                // c0
	FMULS	F3, F24, F24
	FADDS	F24, F25, F24

	// ldexp
	LSL	$23, R3, R7
	FMOVS	F24, R6
	ADD	R7, R6, R6
	FMOVS	R6, F24

	// Store and accumulate
	FMOVS	F24, (R4)
	FADDS	F24, F28, F28

	ADD	$4, R4, R4
	SUB	$1, R5, R5
	CBNZ	R5, exp_scalar

exp_reduce:
	// Reduce V29.4S sum to scalar, add to F28
	// FADDP V29.4S, V29.4S, V29.4S
	WORD	$0x6E3DD7BD
	// FADDP S29, V29.2S
	WORD	$0x7E30DBBD
	// Add scalar tail sum
	FADDS	F29, F28, F28
	// F28 = total sum

	// ========== PASS 3: Normalize by 1/sum ==========
	MOVW	$0x3F800000, R3
	FMOVS	R3, F0
	FDIVS	F28, F0, F27           // F27 = 1/sum
	WORD	$0x4E040779     // DUP V25.4S, V27.S[0]

	MOVD	R0, R4
	MOVD	R1, R5

	CMP	$4, R5
	BLT	norm_tail

norm_loop4:
	VLD1	(R4), [V0.S4]
	// FMUL V0.4S, V0.4S, V25.4S
	WORD	$0x6E39DC00
	VST1	[V0.S4], (R4)
	ADD	$16, R4, R4
	SUB	$4, R5, R5
	CMP	$4, R5
	BGE	norm_loop4

norm_tail:
	CBZ	R5, sm_done

norm_scalar:
	FMOVS	(R4), F0
	FMULS	F0, F27, F0
	FMOVS	F0, (R4)
	ADD	$4, R4, R4
	SUB	$1, R5, R5
	CBNZ	R5, norm_scalar

sm_done:
	RET
