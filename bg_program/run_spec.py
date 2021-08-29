#!/usr/bin/python2

import os
import subprocess
import time
import random

def spec_cmd(spec_prog):
    return "taskset 0x8 runspec --config=test.cfg --size=train" \
    " --noreportable --tune=base --iterations=1 {spec_prog}".format(
        spec_prog=spec_prog)

spec_benchmarks = ('perlbench', 'bzip2', 'gcc', 'mcf', 'milc', 'namd',
'gobmk', 'soplex', 'povray', 'hmmer', 'sjeng', 'libquantum',
'h264ref', 'lbm', 'omnetpp', 'astar')

while True:
    num_spec_processes = random.randint(1, 5)

    benchmark_run = []
    for i in range(num_spec_processes):
        benchmark_idx = random.randrange(len(spec_benchmarks))
        benchmark_run.append(spec_benchmarks[benchmark_idx])
    print("Run {l} benchmarks:".format(l=len(benchmark_run)), benchmark_run)

    running_processes = []
    for spec_prog in benchmark_run:
        cmd = spec_cmd(spec_prog)
        print(cmd)
        spec_process = subprocess.Popen(cmd.split())
        running_processes.append(spec_process)

    for p in running_processes:
        p.wait()
    time.sleep(1)
