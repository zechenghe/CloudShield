#!/bin/bash

# If attack not success (e.g., segmentation fault),
# change BUFFER_PTR_L in payload.S to the printed address (buffer_addr).
# Then 'make payload'

set -u

while true
  do
    taskset 0x8 env - /home/zechengh/Mastik/ad/attack/bufferoverflow/victim.o
  done
