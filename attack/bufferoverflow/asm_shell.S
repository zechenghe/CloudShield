/* victim.c */
/* Author: Zecheng He @ Princeton University */
/X86_64 assembly code to invoke a shell/

#include <sys/syscall.h>

.globl main
        .type   main, @function

main:
        movq    $file_to_run, %rcx
        xor     %rax, %rax
        mov     $59, %al                /* syscall arg 1: syscall number execve(59) */
        movq    %rcx, %rdi               /* syscall arg 2: string pathname */
        leaq    8(%rcx), %rsi            /* syscall arg 3: argv ptr to ['/bin/sh']*/
        xor     %rdx, %rdx                 /* syscall arg 4: envp (NULL) */
        syscall                          /* Call execve("/bin/sh", ["/bin/sh"], []) */

        xor     %rax, %rax
        mov     $60, %al                /* syscall arg 1: SYS_exit (60) */
        xorq    %rdi,%rdi                /* syscall arg 2: 0 */
        syscall                          /* invoke syscall */

file_to_run:
        .ascii  "/bin/sh\0"
        .quad   file_to_run              /* Pointer to argv array.*/
        .quad   0x0
