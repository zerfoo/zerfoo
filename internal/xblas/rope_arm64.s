#include "textflag.h"

// func RoPEF32(out, in, cos, sin *float32, halfDim, headDim int)
//
// out    = +0(FP)
// in     = +8(FP)
// cos    = +16(FP)
// sin    = +24(FP)
// halfDim = +32(FP)
// headDim = +40(FP)
//
// Algorithm (per position):
//   For i in [0, halfDim):
//     out[i]         = in[i]*cos[i] - in[i+halfDim]*sin[i]
//     out[i+halfDim] = in[i+halfDim]*cos[i] + in[i]*sin[i]
//   For j in [2*halfDim, headDim):
//     out[j] = in[j]
//
// Register allocation (avoids R16/R17 which are ARM64 IP0/IP1):
//   R0 = out first-half cursor
//   R1 = in first-half cursor
//   R2 = cos cursor
//   R3 = sin cursor
//   R4 = halfDim (preserved)
//   R5 = headDim (preserved)
//   R6 = temp
//   R7 = in second-half cursor
//   R8 = out second-half cursor
//   R9 = NEON loop counter
//   R10 = scalar loop counter

TEXT ·RoPEF32(SB), NOSPLIT, $0-48
	MOVD out+0(FP), R0       // R0 = out first-half cursor
	MOVD in+8(FP), R1        // R1 = in first-half cursor
	MOVD cos+16(FP), R2      // R2 = cos cursor
	MOVD sin+24(FP), R3      // R3 = sin cursor
	MOVD halfDim+32(FP), R4  // R4 = halfDim
	MOVD headDim+40(FP), R5  // R5 = headDim

	// Compute byte offset for the second-half pointers
	LSL $2, R4, R6           // R6 = halfDim * 4 (byte offset)

	// R7 = &in[halfDim], R8 = &out[halfDim]
	ADD R6, R1, R7           // R7 = in second-half cursor
	ADD R6, R0, R8           // R8 = out second-half cursor

	// Set up loop counters
	LSR $2, R4, R9           // R9 = halfDim / 4 (NEON iterations)
	AND $3, R4, R10          // R10 = halfDim % 4 (scalar remainder)

	CBZ R9, tail_scalar

neon_loop:
	// Load cos[i:i+4] and sin[i:i+4]
	VLD1.P 16(R2), [V0.S4]  // V0 = cos[i..i+3]
	VLD1.P 16(R3), [V1.S4]  // V1 = sin[i..i+3]

	// Load in[i:i+4] (first half) and in[i+halfDim:i+halfDim+4] (second half)
	VLD1.P 16(R1), [V2.S4]  // V2 = in[i..i+3] (first half)
	VLD1.P 16(R7), [V3.S4]  // V3 = in[i+hd..i+hd+3] (second half)

	// out[i] = in[i]*cos[i] - in[i+halfDim]*sin[i]
	// FMUL V4.4S, V2.4S, V0.4S
	WORD $0x6E20DC44
	// FMLS V4.4S, V3.4S, V1.4S
	// 0x4EA0CC00 | (Rm=1 << 16) | (Rn=3 << 5) | Rd=4 = 0x4EA1CC64
	WORD $0x4EA1CC64

	// out[i+halfDim] = in[i+halfDim]*cos[i] + in[i]*sin[i]
	// FMUL V5.4S, V3.4S, V0.4S
	WORD $0x6E20DC65
	// FMLA V5.4S, V2.4S, V1.4S
	VFMLA V2.S4, V1.S4, V5.S4

	// Store results
	VST1.P [V4.S4], 16(R0)  // out[i..i+3]
	VST1.P [V5.S4], 16(R8)  // out[i+hd..i+hd+3]

	SUB $1, R9, R9
	CBNZ R9, neon_loop

tail_scalar:
	CBZ R10, passthrough

scalar_loop:
	FMOVS (R2), F0           // cos[i]
	FMOVS (R3), F1           // sin[i]
	FMOVS (R1), F2           // in[i] (first half)
	FMOVS (R7), F3           // in[i+halfDim] (second half)

	// out[i] = in[i]*cos[i] - in[i+halfDim]*sin[i]
	FMULS F0, F2, F4          // F4 = in_first * cos
	FMULS F1, F3, F5          // F5 = in_second * sin
	FSUBS F5, F4, F4           // F4 = F4 - F5

	// out[i+halfDim] = in[i+halfDim]*cos[i] + in[i]*sin[i]
	FMULS F0, F3, F5          // F5 = in_second * cos
	FMULS F1, F2, F6          // F6 = in_first * sin
	FADDS F6, F5, F5           // F5 = F5 + F6

	FMOVS F4, (R0)           // store out[i]
	FMOVS F5, (R8)           // store out[i+halfDim]

	ADD $4, R0, R0
	ADD $4, R1, R1
	ADD $4, R7, R7
	ADD $4, R8, R8
	ADD $4, R2, R2
	ADD $4, R3, R3

	SUB $1, R10, R10
	CBNZ R10, scalar_loop

passthrough:
	// After the RoPE loop, R7 = &in[2*halfDim], R8 = &out[2*halfDim]
	// R6 = headDim - 2*halfDim
	LSL $1, R4, R6           // R6 = 2 * halfDim
	SUB R6, R5, R6           // R6 = headDim - 2*halfDim
	CBZ R6, done

	// Try 4-wide NEON copy
	LSR $2, R6, R9           // R9 = remaining / 4
	AND $3, R6, R10          // R10 = remaining % 4
	CBZ R9, passthrough_scalar

passthrough_neon:
	VLD1.P 16(R7), [V0.S4]
	VST1.P [V0.S4], 16(R8)
	SUB $1, R9, R9
	CBNZ R9, passthrough_neon

passthrough_scalar:
	CBZ R10, done

passthrough_scalar_loop:
	FMOVS (R7), F0
	FMOVS F0, (R8)
	ADD $4, R7, R7
	ADD $4, R8, R8
	SUB $1, R10, R10
	CBNZ R10, passthrough_scalar_loop

done:
	RET
