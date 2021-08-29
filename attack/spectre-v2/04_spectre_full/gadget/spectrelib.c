#include <stdint.h>
  void IndirectCall(const void *proc,
                     const void *target_memory,
                     const void *probe_start);
  void Touch(uint8_t *target_memory, uint8_t *probe_start);

  void LoadGadget() {}

