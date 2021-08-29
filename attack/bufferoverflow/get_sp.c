/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

/* This program has a buffer overflow vulnerability. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


unsigned long get_sp(void)
{
    /* This function (suggested in alephOne's paper) prints the
       stack pointer using assembly code. */
    __asm__("movl %rsp,%rax");
}

int main(int argc, char **argv)
{
    char *addr;
    addr = get_sp();
    printf("%p", addr);
	return 0;
}
