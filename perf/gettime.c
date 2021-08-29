#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>


int main(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    printf("%f \n", t.tv_sec + (float)t.tv_usec / 1000000.0);
}
