all: main.cpp 01.asm
	nasm -f elf64 01.asm
	g++ -c main.cpp -std=c++11 -O0
	g++ main.o 01.o -o spectrev2

clean:
	rm *.o spectre
