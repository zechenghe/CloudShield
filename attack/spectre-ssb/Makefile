CC=gcc
CYCLES?=100
#CFLAGS=-Wall -DHIT_THRESHOLD=$(CYCLES) -DNO_INTERRUPTS
CFLAGS=-Wall -DHIT_THRESHOLD=$(CYCLES)

ssb: ssb.c
	$(CC) -o spectrev4 ssb.c $(CFLAGS)

clean:
	rm spectrev4 ssb_*
