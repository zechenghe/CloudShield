all:   
	g++ -c main.cpp -std=c++17 -O0 
	nasm -f elf64 01.asm
	g++ main.o 01.o -o spectre -ldl -lpthread -L../gadget/ -lmyspectreattack
clean:
	rm *.o spectre
