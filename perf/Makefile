KDIR := /lib/modules/$(shell uname -r)/build
CC = gcc
TARGET = event_open_user
obj-m += event_open_kernel.o

all: kernel user gettime

kernel:
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) modules

user: $(TARGET).c
	$(CC) -o $(TARGET) $(TARGET).c

gettimer: gettime.c
	$(CC) -o gettimer gettime.c

clean:
	rm -rf *.o *.ko *.mod.* *.cmd .module* modules* Module* .*.cmd .tmp* $(TARGET) gettime
