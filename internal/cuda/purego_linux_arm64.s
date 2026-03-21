//go:build linux && arm64 && !cuda

#include "textflag.h"

// ccallTrampoline is called by asmcgocall(ccallTrampoline, &ccallArgs).
// It runs on the system stack (g0). R0 points to a ccallArgs struct:
//
//   struct ccallArgs {
//       fn   uintptr     // offset 0
//       args [20]uintptr  // offset 8 (args[0] at 8, args[1] at 16, ... args[19] at 160)
//       ret  uintptr     // offset 168
//   }
//
// AAPCS64 calling convention: R0-R7 hold the first 8 args, stack holds the rest.
// R9 is used as a scratch register (caller-saved).
// R19, R20 are callee-saved in AAPCS64.
//
// Go's arm64 assembler saves LR at RSP+0 in its prologue. The C calling
// convention also places outgoing stack arguments at RSP+0. To avoid the
// stack args overwriting the saved LR, we push an extra 96 bytes for the
// outgoing argument area below the Go-managed frame.
TEXT ·ccallTrampoline(SB),NOSPLIT,$0
	// Save args struct pointer and LR in callee-saved registers.
	MOVD R0, R19
	MOVD R30, R20

	// Load C function pointer
	MOVD 0(R19), R9

	// Load register arguments (AAPCS64: R0-R7)
	MOVD 8(R19), R0
	MOVD 16(R19), R1
	MOVD 24(R19), R2
	MOVD 32(R19), R3
	MOVD 40(R19), R4
	MOVD 48(R19), R5
	MOVD 56(R19), R6
	MOVD 64(R19), R7

	// Push outgoing stack argument area (96 bytes for args[8]-args[19]).
	SUB $96, RSP
	MOVD 72(R19), R10
	MOVD R10, 0(RSP)
	MOVD 80(R19), R10
	MOVD R10, 8(RSP)
	MOVD 88(R19), R10
	MOVD R10, 16(RSP)
	MOVD 96(R19), R10
	MOVD R10, 24(RSP)
	MOVD 104(R19), R10
	MOVD R10, 32(RSP)
	MOVD 112(R19), R10
	MOVD R10, 40(RSP)
	MOVD 120(R19), R10
	MOVD R10, 48(RSP)
	MOVD 128(R19), R10
	MOVD R10, 56(RSP)
	MOVD 136(R19), R10
	MOVD R10, 64(RSP)
	MOVD 144(R19), R10
	MOVD R10, 72(RSP)
	MOVD 152(R19), R10
	MOVD R10, 80(RSP)
	MOVD 160(R19), R10
	MOVD R10, 88(RSP)

	// Call the C function
	CALL (R9)

	// Pop outgoing stack argument area
	ADD $96, RSP

	// Store return value
	MOVD R0, 168(R19)

	// Restore LR and return to asmcgocall
	MOVD R20, R30
	RET
