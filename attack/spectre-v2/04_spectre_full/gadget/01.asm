global IndirectCall:function
global Touch:function

section .text

IndirectCall:
  mov r9, rdi
  mov rdi, rsi
  mov rsi, rdx
  clflush [r9]
  movzx eax, byte [rdi]
  shl rax, 0Ch
  call [r9]
  ret

Touch:
  mov al, byte [rax+rsi]
  ret
