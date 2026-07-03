#include "textflag.h"

// Assembly trampolines for dynamically imported C library functions.
// Each trampoline jumps to the corresponding symbol resolved by
// //go:cgo_import_dynamic. This mirrors the golang.org/x/sys/unix darwin
// idiom: the trampoline itself is a file-local (<>) symbol, and its raw
// ABI0 address is exported to Go through a DATA directive so that
// syscall.syscall6 receives the true entry point (not a func-value PC).

TEXT libc_dlopen_trampoline<>(SB),NOSPLIT,$0-0
	JMP libc_dlopen(SB)
GLOBL ·libc_dlopen_trampoline_addr(SB), RODATA, $8
DATA  ·libc_dlopen_trampoline_addr(SB)/8, $libc_dlopen_trampoline<>(SB)

TEXT libc_dlsym_trampoline<>(SB),NOSPLIT,$0-0
	JMP libc_dlsym(SB)
GLOBL ·libc_dlsym_trampoline_addr(SB), RODATA, $8
DATA  ·libc_dlsym_trampoline_addr(SB)/8, $libc_dlsym_trampoline<>(SB)

TEXT libc_dlclose_trampoline<>(SB),NOSPLIT,$0-0
	JMP libc_dlclose(SB)
GLOBL ·libc_dlclose_trampoline_addr(SB), RODATA, $8
DATA  ·libc_dlclose_trampoline_addr(SB)/8, $libc_dlclose_trampoline<>(SB)

TEXT libc_dlerror_trampoline<>(SB),NOSPLIT,$0-0
	JMP libc_dlerror(SB)
GLOBL ·libc_dlerror_trampoline_addr(SB), RODATA, $8
DATA  ·libc_dlerror_trampoline_addr(SB)/8, $libc_dlerror_trampoline<>(SB)
