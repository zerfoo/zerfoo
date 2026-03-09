#include "textflag.h"

// func q4DotBlockSIMD(packed *byte, scale float32, x *float32) float32
//
// Computes the dot product of one Q4 block (16 packed bytes = 32 nibbles)
// with 32 float32 activation values, using NEON.
//
// Layout: packed=0(FP), scale=8(FP), x=16(FP), ret=24(FP)
TEXT ·q4DotBlockSIMD(SB), NOSPLIT, $0-28
	MOVD	packed+0(FP), R0
	FMOVS	scale+8(FP), F7
	MOVD	x+16(FP), R1

	// Load 16 packed bytes.
	VLD1	(R0), [V0.B16]

	// Create 0x0F mask.
	MOVD	$0x0F0F0F0F0F0F0F0F, R3
	VMOV	R3, V16.D[0]
	VMOV	R3, V16.D[1]

	// Extract nibbles.
	VAND	V0.B16, V16.B16, V1.B16
	WORD	$0x6F0C0402				// USHR V2.16B, V0.16B, #4

	// Interleave: [lo0, hi0, lo1, hi1, ...]
	WORD	$0x4E023823				// ZIP1 V3.16B, V1.16B, V2.16B
	WORD	$0x4E027824				// ZIP2 V4.16B, V1.16B, V2.16B

	// Create uint16 vector of 8s.
	MOVD	$0x0008000800080008, R4
	VMOV	R4, V17.D[0]
	VMOV	R4, V17.D[1]

	// Zero accumulators.
	VEOR	V30.B16, V30.B16, V30.B16
	VEOR	V31.B16, V31.B16, V31.B16

	// --- Process V3: positions 0-15 ---
	WORD	$0x2F08A465				// USHLL V5.8H, V3.8B, #0
	WORD	$0x6F08A466				// USHLL2 V6.8H, V3.16B, #0
	WORD	$0x6E7184A5				// SUB V5.8H, V5.8H, V17.8H
	WORD	$0x6E7184C6				// SUB V6.8H, V6.8H, V17.8H
	WORD	$0x0F10A4B8				// SSHLL V24.4S, V5.4H, #0
	WORD	$0x4F10A4B9				// SSHLL2 V25.4S, V5.8H, #0
	WORD	$0x0F10A4DA				// SSHLL V26.4S, V6.4H, #0
	WORD	$0x4F10A4DB				// SSHLL2 V27.4S, V6.8H, #0
	WORD	$0x4E21DB18				// SCVTF V24.4S, V24.4S
	WORD	$0x4E21DB39				// SCVTF V25.4S, V25.4S
	WORD	$0x4E21DB5A				// SCVTF V26.4S, V26.4S
	WORD	$0x4E21DB7B				// SCVTF V27.4S, V27.4S

	VLD1.P	64(R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	VFMLA	V24.S4, V20.S4, V30.S4
	VFMLA	V25.S4, V21.S4, V31.S4
	VFMLA	V26.S4, V22.S4, V30.S4
	VFMLA	V27.S4, V23.S4, V31.S4

	// --- Process V4: positions 16-31 ---
	WORD	$0x2F08A485				// USHLL V5.8H, V4.8B, #0
	WORD	$0x6F08A486				// USHLL2 V6.8H, V4.16B, #0
	WORD	$0x6E7184A5				// SUB V5.8H, V5.8H, V17.8H
	WORD	$0x6E7184C6				// SUB V6.8H, V6.8H, V17.8H
	WORD	$0x0F10A4B8				// SSHLL V24.4S, V5.4H, #0
	WORD	$0x4F10A4B9				// SSHLL2 V25.4S, V5.8H, #0
	WORD	$0x0F10A4DA				// SSHLL V26.4S, V6.4H, #0
	WORD	$0x4F10A4DB				// SSHLL2 V27.4S, V6.8H, #0
	WORD	$0x4E21DB18				// SCVTF V24.4S, V24.4S
	WORD	$0x4E21DB39				// SCVTF V25.4S, V25.4S
	WORD	$0x4E21DB5A				// SCVTF V26.4S, V26.4S
	WORD	$0x4E21DB7B				// SCVTF V27.4S, V27.4S

	VLD1	(R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	VFMLA	V24.S4, V20.S4, V30.S4
	VFMLA	V25.S4, V21.S4, V31.S4
	VFMLA	V26.S4, V22.S4, V30.S4
	VFMLA	V27.S4, V23.S4, V31.S4

	// Horizontal reduction.
	WORD	$0x4E3FD7DE				// FADD V30.4S, V30.4S, V31.4S
	WORD	$0x6E3ED7DE				// FADDP V30.4S, V30.4S, V30.4S
	WORD	$0x6E3ED7DE				// FADDP V30.4S → scalar in V30.S[0]

	FMULS	F30, F7, F0
	FMOVS	F0, ret+24(FP)
	RET

// func q4DotRowSIMD(blockPtr unsafe.Pointer, x *float32, numBlocks int) float32
//
// Processes numBlocks consecutive Q4 blocks against activation vector x.
// Each block is 18 bytes: 2B float16 scale (LE) + 16B packed nibbles.
// This eliminates per-block Go function call overhead and converts float16
// scales using NEON FCVT directly.
//
// Layout: blockPtr=0(FP), x=8(FP), numBlocks=16(FP), ret=24(FP)
TEXT ·q4DotRowSIMD(SB), NOSPLIT, $0-28
	MOVD	blockPtr+0(FP), R0
	MOVD	x+8(FP), R1
	MOVD	numBlocks+16(FP), R2

	// Zero total accumulator (F28 = 0.0).
	VEOR	V28.B16, V28.B16, V28.B16

	// Set up constants (persist across loop iterations).
	MOVD	$0x0F0F0F0F0F0F0F0F, R3
	VMOV	R3, V16.D[0]
	VMOV	R3, V16.D[1]
	MOVD	$0x0008000800080008, R4
	VMOV	R4, V17.D[0]
	VMOV	R4, V17.D[1]

	CBZ	R2, row_done

row_loop:
	// Load float16 scale from block [R0], convert to float32.
	WORD	$0x7C400007				// LDR H7, [X0]
	WORD	$0x1EE240E7				// FCVT S7, H7

	// Load 16 packed bytes from block [R0+2].
	ADD	$2, R0, R5
	VLD1	(R5), [V0.B16]

	// Extract nibbles.
	VAND	V0.B16, V16.B16, V1.B16
	WORD	$0x6F0C0402				// USHR V2.16B, V0.16B, #4

	// Interleave.
	WORD	$0x4E023823				// ZIP1 V3.16B, V1.16B, V2.16B
	WORD	$0x4E027824				// ZIP2 V4.16B, V1.16B, V2.16B

	// Zero block accumulators.
	VEOR	V30.B16, V30.B16, V30.B16
	VEOR	V31.B16, V31.B16, V31.B16

	// --- Positions 0-15 (from V3) ---
	WORD	$0x2F08A465				// USHLL V5.8H, V3.8B, #0
	WORD	$0x6F08A466				// USHLL2 V6.8H, V3.16B, #0
	WORD	$0x6E7184A5				// SUB V5.8H, V5.8H, V17.8H
	WORD	$0x6E7184C6				// SUB V6.8H, V6.8H, V17.8H
	WORD	$0x0F10A4B8				// SSHLL V24.4S, V5.4H, #0
	WORD	$0x4F10A4B9				// SSHLL2 V25.4S, V5.8H, #0
	WORD	$0x0F10A4DA				// SSHLL V26.4S, V6.4H, #0
	WORD	$0x4F10A4DB				// SSHLL2 V27.4S, V6.8H, #0
	WORD	$0x4E21DB18				// SCVTF V24.4S, V24.4S
	WORD	$0x4E21DB39				// SCVTF V25.4S, V25.4S
	WORD	$0x4E21DB5A				// SCVTF V26.4S, V26.4S
	WORD	$0x4E21DB7B				// SCVTF V27.4S, V27.4S

	VLD1.P	64(R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	VFMLA	V24.S4, V20.S4, V30.S4
	VFMLA	V25.S4, V21.S4, V31.S4
	VFMLA	V26.S4, V22.S4, V30.S4
	VFMLA	V27.S4, V23.S4, V31.S4

	// --- Positions 16-31 (from V4) ---
	WORD	$0x2F08A485				// USHLL V5.8H, V4.8B, #0
	WORD	$0x6F08A486				// USHLL2 V6.8H, V4.16B, #0
	WORD	$0x6E7184A5				// SUB V5.8H
	WORD	$0x6E7184C6				// SUB V6.8H
	WORD	$0x0F10A4B8				// SSHLL V24.4S
	WORD	$0x4F10A4B9				// SSHLL2 V25.4S
	WORD	$0x0F10A4DA				// SSHLL V26.4S
	WORD	$0x4F10A4DB				// SSHLL2 V27.4S
	WORD	$0x4E21DB18				// SCVTF V24.4S, V24.4S
	WORD	$0x4E21DB39				// SCVTF V25.4S, V25.4S
	WORD	$0x4E21DB5A				// SCVTF V26.4S, V26.4S
	WORD	$0x4E21DB7B				// SCVTF V27.4S, V27.4S

	VLD1.P	64(R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	VFMLA	V24.S4, V20.S4, V30.S4
	VFMLA	V25.S4, V21.S4, V31.S4
	VFMLA	V26.S4, V22.S4, V30.S4
	VFMLA	V27.S4, V23.S4, V31.S4

	// Horizontal reduce block.
	WORD	$0x4E3FD7DE				// FADD V30.4S, V30.4S, V31.4S
	WORD	$0x6E3ED7DE				// FADDP V30.4S
	WORD	$0x6E3ED7DE				// FADDP → F30 = block dot sum

	// total += block_sum * scale
	FMULS	F30, F7, F30
	FADDS	F30, F28, F28

	// Advance block pointer by 18 bytes, decrement counter.
	ADD	$18, R0, R0
	SUB	$1, R2, R2
	CBNZ	R2, row_loop

row_done:
	FMOVS	F28, ret+24(FP)
	RET
