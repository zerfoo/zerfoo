#include "textflag.h"

// func sgemmAccRowNeon(c, b unsafe.Pointer, aVal float32, n int)
//
// Computes c[j] += aVal * b[j] for j = 0..n-1 using NEON FMLA.
// Layout: c=0(FP), b=8(FP), aVal=16(FP), n=24(FP)
TEXT ·sgemmAccRowNeon(SB), NOSPLIT, $0-32
	MOVD	c+0(FP), R0
	MOVD	b+8(FP), R1
	FMOVS	aVal+16(FP), F6
	MOVD	n+24(FP), R2

	// Broadcast F6 to all 4 lanes of V6
	VDUP	V6.S[0], V6.S4

	CMP	$8, R2
	BLT	acc_tail4

acc_loop8:
	VLD1	(R0), [V0.S4, V1.S4]	// load c[0:8]
	VLD1	(R1), [V2.S4, V3.S4]	// load b[0:8]
	VFMLA	V6.S4, V2.S4, V0.S4	// c[0:4] += aVal * b[0:4]
	VFMLA	V6.S4, V3.S4, V1.S4	// c[4:8] += aVal * b[4:8]
	VST1	[V0.S4, V1.S4], (R0)
	ADD	$32, R0, R0
	ADD	$32, R1, R1
	SUB	$8, R2, R2
	CMP	$8, R2
	BGE	acc_loop8

acc_tail4:
	CMP	$4, R2
	BLT	acc_tail1

	VLD1	(R0), [V0.S4]
	VLD1	(R1), [V2.S4]
	VFMLA	V6.S4, V2.S4, V0.S4
	VST1	[V0.S4], (R0)
	ADD	$16, R0, R0
	ADD	$16, R1, R1
	SUB	$4, R2, R2

acc_tail1:
	CBZ	R2, acc_done

acc_scalar:
	FMOVS	(R1), F1
	FMULS	F6, F1, F1
	FMOVS	(R0), F2
	FADDS	F1, F2, F2
	FMOVS	F2, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	SUB	$1, R2, R2
	CBNZ	R2, acc_scalar

acc_done:
	RET
