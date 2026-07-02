#include "textflag.h"

// func sgemmAccRow(c, b unsafe.Pointer, aVal float32, n int)
//
// Computes c[j] += aVal * b[j] for j = 0..n-1 using AVX2 FMA.
// Layout: c=0(FP), b=8(FP), aVal=16(FP), n=24(FP)
// Total frame args: 32 bytes.
TEXT ·sgemmAccRow(SB), NOSPLIT, $0-32
	MOVQ	c+0(FP), DI
	MOVQ	b+8(FP), SI
	MOVSS	aVal+16(FP), X6
	MOVQ	n+24(FP), CX

	// Broadcast aVal to all 8 lanes of Y6
	VBROADCASTSS	X6, Y6

	CMPQ	CX, $16
	JLT	acc_tail8

acc_loop16:
	VMOVUPS	(DI), Y2
	VMOVUPS	32(DI), Y3
	VMOVUPS	(SI), Y4
	VMOVUPS	32(SI), Y5
	VFMADD231PS	Y6, Y4, Y2
	VFMADD231PS	Y6, Y5, Y3
	VMOVUPS	Y2, (DI)
	VMOVUPS	Y3, 32(DI)
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$16, CX
	CMPQ	CX, $16
	JGE	acc_loop16

acc_tail8:
	CMPQ	CX, $8
	JLT	acc_tail4

	VMOVUPS	(DI), Y2
	VMOVUPS	(SI), Y3
	VFMADD231PS	Y6, Y3, Y2
	VMOVUPS	Y2, (DI)
	ADDQ	$32, SI
	ADDQ	$32, DI
	SUBQ	$8, CX

acc_tail4:
	CMPQ	CX, $4
	JLT	acc_tail1

	VMOVUPS	(DI), X2
	VMOVUPS	(SI), X3
	VFMADD231PS	X6, X3, X2
	VMOVUPS	X2, (DI)
	ADDQ	$16, SI
	ADDQ	$16, DI
	SUBQ	$4, CX

acc_tail1:
	CMPQ	CX, $0
	JEQ	acc_done

acc_scalar:
	MOVSS	(SI), X1
	MULSS	X6, X1
	ADDSS	(DI), X1
	MOVSS	X1, (DI)
	ADDQ	$4, SI
	ADDQ	$4, DI
	SUBQ	$1, CX
	JNZ	acc_scalar

acc_done:
	VZEROUPPER
	RET
