#!/bin/bash

set -u

while true
  do
    taskset 0x8 gpg --yes --batch -r zechengh_key1 -e /home/zechengh/sample.txt
    sleep 1
  done
