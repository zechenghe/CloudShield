#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <fr.h>
#include <util.h>
#include <symbol.h>
#include <sys/io.h>

#define PAGE_SIZE 4096
#define CACHELINE_SIZE 64
#define NPAGES 1024

char *monitor[] = {
  "mpih-mul.c:85",
  "mpih-mul.c:271",
  "mpih-div.c:356"
};
int nmonitor = sizeof(monitor)/sizeof(monitor[0]);

int main(int argc, char **argv) {
  char temp = 0;
  int fd = open(argv[1], O_RDONLY);
  printf("sysconf(SC_PAGE_SIZE) %ld\n", sysconf(_SC_PAGE_SIZE));
  //mapaddress = mmap(0, sysconf(_SC_PAGE_SIZE), PROT_READ, MAP_PRIVATE, fd, offset & ~(sysconf(_SC_PAGE_SIZE) -1));
  //(char*)mmap(0, NPAGES * PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
  //char* buffer = (char*)mmap(0, NPAGES * PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, fd, 0);

  char* buffer = (char*)mmap(0, NPAGES * PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (buffer == MAP_FAILED){
    printf("mmap error\n");
    exit(1);
  }
  printf("Buffer %p\n",buffer);

  char **p = malloc(nmonitor*sizeof(char*));
  fr_t fr = fr_prepare();
  uint64_t offset = 0;
  for (int i = 0; i < nmonitor; i++) {
    offset = sym_getsymboloffset(argv[1], monitor[i]);
    if (offset == ~0ULL) {
      fprintf(stderr, "Cannot find %s in %s\n", monitor[i], argv[1]);
      exit(1);
    }
    p[i] = map_offset(argv[1], offset);
    printf("%s %p, offset %p\n",monitor[i], p[i], (void*)offset);
  }

  srand(0);
  for (int i = 0; i < NPAGES * PAGE_SIZE; i++){
    asm volatile ("clflush 0(%0)": : "r" (buffer + i):);
  }

  asm volatile("mfence");
  asm volatile("mfence");

  printf("RAND_MAX %d\n", RAND_MAX);
  printf("Start For Loop\n");

  while(1){
      //temp = buffer[offset];
      //temp = *p[2];
      temp = buffer[rand() % (NPAGES * PAGE_SIZE)];
      asm volatile("mfence");
  }
}
