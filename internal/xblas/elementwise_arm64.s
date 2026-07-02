#include "textflag.h"

// func VaddF32(out, a, b *float32, n int)
// Layout: out=0(FP), a=8(FP), b=16(FP), n=24(FP)
TEXT ·VaddF32(SB), NOSPLIT, $0-32
	MOVD	out+0(FP), R0
	MOVD	a+8(FP), R1
	MOVD	b+16(FP), R2
	MOVD	n+24(FP), R3

	CBZ	R3, add_done
	CMP	$4, R3
	BLT	add_tail

add_loop4:
	VLD1.P	16(R1), [V0.S4]
	VLD1.P	16(R2), [V1.S4]
	// FADD V0.4S, V0.4S, V1.4S
	WORD	$0x4E21D400
	VST1.P	[V0.S4], 16(R0)
	SUB	$4, R3, R3
	CMP	$4, R3
	BGE	add_loop4

add_tail:
	CBZ	R3, add_done

add_scalar:
	FMOVS	(R1), F0
	FMOVS	(R2), F1
	FADDS	F1, F0, F0
	FMOVS	F0, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	ADD	$4, R2, R2
	SUB	$1, R3, R3
	CBNZ	R3, add_scalar

add_done:
	RET

// func VmulF32(out, a, b *float32, n int)
// Layout: out=0(FP), a=8(FP), b=16(FP), n=24(FP)
TEXT ·VmulF32(SB), NOSPLIT, $0-32
	MOVD	out+0(FP), R0
	MOVD	a+8(FP), R1
	MOVD	b+16(FP), R2
	MOVD	n+24(FP), R3

	CBZ	R3, mul_done
	CMP	$4, R3
	BLT	mul_tail

mul_loop4:
	VLD1.P	16(R1), [V0.S4]
	VLD1.P	16(R2), [V1.S4]
	// FMUL V0.4S, V0.4S, V1.4S
	WORD	$0x6E21DC00
	VST1.P	[V0.S4], 16(R0)
	SUB	$4, R3, R3
	CMP	$4, R3
	BGE	mul_loop4

mul_tail:
	CBZ	R3, mul_done

mul_scalar:
	FMOVS	(R1), F0
	FMOVS	(R2), F1
	FMULS	F1, F0, F0
	FMOVS	F0, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	ADD	$4, R2, R2
	SUB	$1, R3, R3
	CBNZ	R3, mul_scalar

mul_done:
	RET

// func VsubF32(out, a, b *float32, n int)
// Layout: out=0(FP), a=8(FP), b=16(FP), n=24(FP)
TEXT ·VsubF32(SB), NOSPLIT, $0-32
	MOVD	out+0(FP), R0
	MOVD	a+8(FP), R1
	MOVD	b+16(FP), R2
	MOVD	n+24(FP), R3

	CBZ	R3, sub_done
	CMP	$4, R3
	BLT	sub_tail

sub_loop4:
	VLD1.P	16(R1), [V0.S4]
	VLD1.P	16(R2), [V1.S4]
	// FSUB V0.4S, V0.4S, V1.4S
	WORD	$0x4EA1D400
	VST1.P	[V0.S4], 16(R0)
	SUB	$4, R3, R3
	CMP	$4, R3
	BGE	sub_loop4

sub_tail:
	CBZ	R3, sub_done

sub_scalar:
	FMOVS	(R1), F0
	FMOVS	(R2), F1
	FSUBS	F1, F0, F0
	FMOVS	F0, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	ADD	$4, R2, R2
	SUB	$1, R3, R3
	CBNZ	R3, sub_scalar

sub_done:
	RET

// func VdivF32(out, a, b *float32, n int)
// Layout: out=0(FP), a=8(FP), b=16(FP), n=24(FP)
TEXT ·VdivF32(SB), NOSPLIT, $0-32
	MOVD	out+0(FP), R0
	MOVD	a+8(FP), R1
	MOVD	b+16(FP), R2
	MOVD	n+24(FP), R3

	CBZ	R3, div_done
	CMP	$4, R3
	BLT	div_tail

div_loop4:
	VLD1.P	16(R1), [V0.S4]
	VLD1.P	16(R2), [V1.S4]
	// FDIV V0.4S, V0.4S, V1.4S
	WORD	$0x6E21FC00
	VST1.P	[V0.S4], 16(R0)
	SUB	$4, R3, R3
	CMP	$4, R3
	BGE	div_loop4

div_tail:
	CBZ	R3, div_done

div_scalar:
	FMOVS	(R1), F0
	FMOVS	(R2), F1
	FDIVS	F1, F0, F0
	FMOVS	F0, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	ADD	$4, R2, R2
	SUB	$1, R3, R3
	CBNZ	R3, div_scalar

div_done:
	RET
