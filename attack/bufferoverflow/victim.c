/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

/* This program has a buffer overflow vulnerability. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void* get_sp(void)
{
    /* This function (suggested in alephOne's paper) prints the
       stack pointer using assembly code. */
    __asm__("mov %rsp, %rax");
}


int vulnerable_func(char *str)
{
	char buffer[128];
    //void *addr;
    //addr = get_sp();
    printf("%p\n", buffer);

    /* The following strcpy function has a buffer overflow problem */

    strcpy(buffer, str);
	return 0;
}

int main(int argc, char **argv)
{
	char str[1024];
	FILE *badfile;
	badfile = fopen("/home/zechengh/Mastik/ad/attack/bufferoverflow/badfile", "r");
	fread(str, sizeof(char), 1024, badfile);
	vulnerable_func(str);
	printf("If you see this, payload is NOT executed. Attack fails.\n");
	return 0;
}
