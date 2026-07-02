#include "textflag.h"

// Assembly trampolines for dynamically imported C library functions.
// Each trampoline jumps to the corresponding symbol resolved by
// //go:cgo_import_dynamic. This is the standard Go pattern for
// calling C library functions without CGo (used by syscall package).

TEXT ·libc_dlopen_trampoline(SB),NOSPLIT,$0-0
	JMP libc_dlopen(SB)

TEXT ·libc_dlsym_trampoline(SB),NOSPLIT,$0-0
	JMP libc_dlsym(SB)

TEXT ·libc_dlclose_trampoline(SB),NOSPLIT,$0-0
	JMP libc_dlclose(SB)

TEXT ·libc_dlerror_trampoline(SB),NOSPLIT,$0-0
	JMP libc_dlerror(SB)
