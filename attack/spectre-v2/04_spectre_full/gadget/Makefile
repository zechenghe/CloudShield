all:  
	objcopy  --input binary --output elf64-x86-64 --binary-architecture i386:x86-64 1.bmp data.o
	gcc -fPIC -g -c -Wall spectrelib.c
	gcc -shared -Wl,-soname,libmyspectreattack.so.1 -o libmyspectreattack.so.1.0.1 data.o spectrelib.o 
	ln -sf libmyspectreattack.so.1.0.1 libmyspectreattack.so
