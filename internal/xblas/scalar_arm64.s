#include "textflag.h"

// func VmulScalarF32(out, a *float32, scalar float32, n int)
// Layout: out=0(FP), a=8(FP), scalar=16(FP), n=24(FP)
TEXT ·VmulScalarF32(SB), NOSPLIT, $0-32
	MOVD	out+0(FP), R0
	MOVD	a+8(FP), R1
	MOVW	scalar+16(FP), R4
	MOVD	n+24(FP), R3

	CBZ	R3, smul_done

	// Broadcast scalar to V2.S4
	VDUP	R4, V2.S4

	CMP	$4, R3
	BLT	smul_tail

smul_loop4:
	VLD1.P	16(R1), [V0.S4]
	// FMUL V0.4S, V0.4S, V2.4S
	WORD	$0x6E22DC00
	VST1.P	[V0.S4], 16(R0)
	SUB	$4, R3, R3
	CMP	$4, R3
	BGE	smul_loop4

smul_tail:
	CBZ	R3, smul_done

	// Load scalar into F2 for scalar ops
	FMOVS	R4, F2

smul_scalar:
	FMOVS	(R1), F0
	FMULS	F2, F0, F0
	FMOVS	F0, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	SUB	$1, R3, R3
	CBNZ	R3, smul_scalar

smul_done:
	RET

// func VaddScalarF32(out, a *float32, scalar float32, n int)
// Layout: out=0(FP), a=8(FP), scalar=16(FP), n=24(FP)
TEXT ·VaddScalarF32(SB), NOSPLIT, $0-32
	MOVD	out+0(FP), R0
	MOVD	a+8(FP), R1
	MOVW	scalar+16(FP), R4
	MOVD	n+24(FP), R3

	CBZ	R3, sadd_done

	// Broadcast scalar to V2.S4
	VDUP	R4, V2.S4

	CMP	$4, R3
	BLT	sadd_tail

sadd_loop4:
	VLD1.P	16(R1), [V0.S4]
	// FADD V0.4S, V0.4S, V2.4S
	WORD	$0x4E22D400
	VST1.P	[V0.S4], 16(R0)
	SUB	$4, R3, R3
	CMP	$4, R3
	BGE	sadd_loop4

sadd_tail:
	CBZ	R3, sadd_done

	FMOVS	R4, F2

sadd_scalar:
	FMOVS	(R1), F0
	FADDS	F2, F0, F0
	FMOVS	F0, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	SUB	$1, R3, R3
	CBNZ	R3, sadd_scalar

sadd_done:
	RET

// func VdivScalarF32(out, a *float32, scalar float32, n int)
// Layout: out=0(FP), a=8(FP), scalar=16(FP), n=24(FP)
TEXT ·VdivScalarF32(SB), NOSPLIT, $0-32
	MOVD	out+0(FP), R0
	MOVD	a+8(FP), R1
	MOVW	scalar+16(FP), R4
	MOVD	n+24(FP), R3

	CBZ	R3, sdiv_done

	// Broadcast scalar to V2.S4
	VDUP	R4, V2.S4

	CMP	$4, R3
	BLT	sdiv_tail

sdiv_loop4:
	VLD1.P	16(R1), [V0.S4]
	// FDIV V0.4S, V0.4S, V2.4S
	WORD	$0x6E22FC00
	VST1.P	[V0.S4], 16(R0)
	SUB	$4, R3, R3
	CMP	$4, R3
	BGE	sdiv_loop4

sdiv_tail:
	CBZ	R3, sdiv_done

	FMOVS	R4, F2

sdiv_scalar:
	FMOVS	(R1), F0
	FDIVS	F2, F0, F0
	FMOVS	F0, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	SUB	$1, R3, R3
	CBNZ	R3, sdiv_scalar

sdiv_done:
	RET
