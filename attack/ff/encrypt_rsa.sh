#!/bin/bash

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg

while true
  do
    taskset 0x8000 $GPG -r zechengh_key1 -o /dev/null -e 'hello'
    echo "Encrypt..."
    sleep 0.1
  done
