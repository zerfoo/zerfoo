#include "textflag.h"

// func RMSNormF32(out, x, weight *float32, D int, eps float32, scale *float32)
//
// Computes out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i].
// Writes the scale factor rsqrt(mean(x^2) + eps) to *scale.
//
// Algorithm:
//   1. Sum of squares using dual NEON accumulators
//   2. Horizontal reduce, divide by D, add eps
//   3. FRSQRTE + 2 Newton-Raphson iterations for rsqrt
//   4. Broadcast scale, multiply x * scale * weight
//
// Register allocation:
//   R0 = out ptr, R1 = x ptr, R2 = weight ptr, R3 = D, R4 = loop counter
//   V4,V5 = sum-of-squares accumulators
//   V6 = broadcast scale (normalize pass)
//   V0,V1,V2,V3 = temporaries
//   F24 = scalar scale result (avoids callee-saved V8-V15)
//   F25,F26 = Newton-Raphson temporaries
//
// Layout: out=0(FP), x=8(FP), weight=16(FP), D=24(FP), eps=32(FP), scale=40(FP)
TEXT ·RMSNormF32(SB), NOSPLIT, $0-48
	MOVD	out+0(FP), R0
	MOVD	x+8(FP), R1
	MOVD	weight+16(FP), R2
	MOVD	D+24(FP), R3

	// Zero accumulators V4, V5
	VEOR	V4.B16, V4.B16, V4.B16
	VEOR	V5.B16, V5.B16, V5.B16

	// Save x pointer for second pass
	MOVD	R1, R5

	MOVD	R3, R4  // R4 = remaining count

	// ---- Pass 1: Sum of squares ----
	CMP	$8, R4
	BLT	sumsq_tail4

sumsq_loop8:
	// Load 8 floats (2x4)
	VLD1.P	16(R1), [V0.S4]
	VLD1.P	16(R1), [V1.S4]

	// FMUL V2.4S, V0.4S, V0.4S
	WORD	$0x6E20DC02
	// FMUL V3.4S, V1.4S, V1.4S
	WORD	$0x6E21DC23

	// Accumulate
	// FADD V4.4S, V4.4S, V2.4S
	WORD	$0x4E22D484
	// FADD V5.4S, V5.4S, V3.4S
	WORD	$0x4E23D4A5

	SUB	$8, R4, R4
	CMP	$8, R4
	BGE	sumsq_loop8

sumsq_tail4:
	CMP	$4, R4
	BLT	sumsq_tail1

	VLD1.P	16(R1), [V0.S4]
	// FMUL V2.4S, V0.4S, V0.4S
	WORD	$0x6E20DC02
	// FADD V4.4S, V4.4S, V2.4S
	WORD	$0x4E22D484

	SUB	$4, R4, R4

sumsq_tail1:
	// Scalar tail uses F0 as accumulator to avoid zeroing V4 upper lanes.
	// Writing to S4 (scalar FADDS) would zero lanes 1-3 of V4, destroying
	// NEON-accumulated values. Instead accumulate separately and add later.
	FMOVS	ZR, F0
	CBZ	R4, sumsq_reduce

sumsq_scalar:
	FMOVS	(R1), F1
	FMULS	F1, F1, F2
	FADDS	F2, F0, F0  // Accumulate into F0 (separate from V4)
	ADD	$4, R1, R1
	SUB	$1, R4, R4
	CBNZ	R4, sumsq_scalar

sumsq_reduce:
	// Combine V4 and V5: V4 = V4 + V5
	// FADD V4.4S, V4.4S, V5.4S
	WORD	$0x4E25D484

	// Horizontal reduce V4 to scalar
	// FADDP V4.4S, V4.4S, V4.4S
	WORD	$0x6E24D484
	// FADDP S4, V4.2S
	WORD	$0x7E30D884

	// Add scalar tail sum
	FADDS	F0, F4, F4

	// S4 now holds sum of squares.
	// Compute mean = sumSq / D
	WORD	$0x9E220060     // SCVTF S0, X3 -- F0 = float(D)
	FDIVS	F0, F4, F4      // F4 = sumSq / D

	// Add eps
	FMOVS	eps+32(FP), F1  // F1 = eps
	FADDS	F1, F4, F4      // F4 = mean + eps

	// Compute rsqrt(F4) via FRSQRTE + 2 Newton-Raphson iterations
	// y0 = FRSQRTE(S24, S4)
	// FRSQRTE Sd, Sn: 0x7EA1D800 | (Rn<<5) | Rd
	// Rn=V4=00100, Rd=V24=11000
	WORD	$0x7EA1D898  // FRSQRTE S24, S4

	// Newton step 1: y1 = y0 * FRSQRTS(F4, y0*y0)
	FMULS	F24, F24, F25    // F25 = y0*y0
	// FRSQRTS S26, S4, S25: 0x5EA0FC00 | (Rm<<16) | (Rn<<5) | Rd
	// Rm=V25=11001, Rn=V4=00100, Rd=V26=11010
	WORD	$0x5EB9FC9A      // FRSQRTS S26, S4, S25
	FMULS	F26, F24, F24    // F24 = y1

	// Newton step 2: y2 = y1 * FRSQRTS(F4, y1*y1)
	FMULS	F24, F24, F25    // F25 = y1*y1
	WORD	$0x5EB9FC9A      // FRSQRTS S26, S4, S25
	FMULS	F26, F24, F24    // F24 = y2 = scale

	// Store scale to *scale pointer
	MOVD	scale+40(FP), R6
	FMOVS	F24, (R6)

	// ---- Pass 2: Normalize ----
	// Broadcast scale to V6.4S
	// DUP V6.4S, V24.S[0]: imm5=00100, Rn=V24=11000, Rd=V6=00110
	WORD	$0x4E040706

	// Restore x pointer
	MOVD	R5, R1
	MOVD	R3, R4  // R4 = D

	CMP	$4, R4
	BLT	norm_tail1

norm_loop4:
	// Load x[i:i+4] and weight[i:i+4]
	VLD1.P	16(R1), [V0.S4]
	VLD1.P	16(R2), [V1.S4]

	// V0 = x * scale
	// FMUL V0.4S, V0.4S, V6.4S
	WORD	$0x6E26DC00
	// V0 = (x * scale) * weight
	// FMUL V0.4S, V0.4S, V1.4S
	WORD	$0x6E21DC00

	// Store
	VST1.P	[V0.S4], 16(R0)

	SUB	$4, R4, R4
	CMP	$4, R4
	BGE	norm_loop4

norm_tail1:
	CBZ	R4, done

norm_scalar:
	FMOVS	(R1), F0
	FMOVS	(R2), F1
	FMULS	F24, F0, F0    // x * scale
	FMULS	F1, F0, F0     // * weight
	FMOVS	F0, (R0)

	ADD	$4, R0, R0
	ADD	$4, R1, R1
	ADD	$4, R2, R2
	SUB	$1, R4, R4
	CBNZ	R4, norm_scalar

done:
	RET
