CC = gcc
CC44 = gcc-4.4
FLAGS = -static -std=c99 -g
FLAGS44 = -static -std=c99 -msse2

spectre: spectre.c
	$(CC) $(FLAGS) spectre.c -o spectrev1
hello:   hello.c
	$(CC) $(FLAGS) hello.c -o hello_static
spectre44: spectre44.c
	$(CC44) $(FLAGS44) spectre44.c -o spectre44_static

clean:
	rm spectrev1 spectre_static spectre_static44
