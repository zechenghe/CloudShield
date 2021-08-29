import os
import signal
import subprocess
import time
import argparse
import functools
import utils
import random
import json
import collections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--core', type = int, default = 3, help='core index')
    parser.add_argument('--us', type = int, default = 100, help='interval in us')
    parser.add_argument('--n_readings', type = int, default = 300000, help='number of HPC readings')
    parser.add_argument('--bg_program', type = str, default = 'webserver', help='background program')

    args = parser.parse_args()

    interval_cycles = int(args.us / 3)
    schedule = collections.defaultdict(lambda: collections.defaultdict(lambda: list()))

    attacks = utils.attacks
    gpg_command = utils.gpg_command

    save_data_dir = 'data/{bg_program}/{us}us/'.format(
        bg_program=args.bg_program,
        us=args.us
    )

    os.system('mkdir -p {save_data_dir}'.format(save_data_dir=save_data_dir))

    monitor_cmd_fn=functools.partial(
        utils.monitor_cmd,
        core=args.core,
        interval_cycles=interval_cycles,
        n_readings=args.n_readings,
        save_data_dir=save_data_dir,
    )

    attack_processes = {}

    # Start attack processes and pause them
    for k in attacks.keys():
        attack_processes[k] = subprocess.Popen(attacks[k].split())
        os.kill(attack_processes[k].pid, signal.SIGSTOP)

    cmd = monitor_cmd_fn(save_data_name='eval_sequence.csv')
    monitor_process = subprocess.Popen(cmd.split())

    # Start of HPC collection
    time.sleep(20)

    # Randomly run multiple attacks
    for k in ['l1pp', 'spectrev4', 'spectrev1', 'ff', 'l3pp', 'bufferoverflow', 'spectrev2', 'spectrev3', 'fr']:
        os.kill(attack_processes[k].pid, signal.SIGCONT)
        schedule[k]['start'].append(utils.get_time())
        time.sleep(10)
        os.kill(attack_processes[k].pid, signal.SIGSTOP)
        schedule[k]['end'].append(utils.get_time())
        time.sleep(random.randint(10,40))


    """
    # Run flush reload attack and gpg concurrently
    # First, run flush_reload attack for 30s
    os.kill(attack_processes['fr'].pid, signal.SIGCONT)
    schedule[k]['start'].append(utils.get_time())
    time.sleep(30)
    os.kill(attack_processes['fr'].pid, signal.SIGSTOP)
    schedule[k]['end'].append(utils.get_time())
    time.sleep(30)

    # Second, run gpg for 45s
    gpg_process = subprocess.Popen(gpg_command.split())
    schedule[k]['start'].append(utils.get_time())
    time.sleep(45)

    # Third, run flush_reload+gpg for 15s
    os.kill(attack_processes['fr'].pid, signal.SIGCONT)
    time.sleep(15)
    gpg_process.terminate()

    # Fourth, run flush_reload for 15s
    time.sleep(15)
    os.kill(attack_processes['fr'].pid, signal.SIGSTOP)
    schedule[k]['end'].append(utils.get_time())
    time.sleep(20)


    # Run spectrev3 attack and gcc
    os.kill(attack_processes['spectrev3'].pid, signal.SIGCONT)
    schedule[k]['start'].append(utils.get_time())
    time.sleep(10)
    os.kill(attack_processes['spectrev3'].pid, signal.SIGSTOP)
    schedule[k]['end'].append(utils.get_time())
    time.sleep(20)

    cmd = utils.spec_cmd('gcc', iterations=20)
    print(cmd)
    spec_process = subprocess.Popen(cmd.split())
    schedule[k]['start'].append(utils.get_time())
    time.sleep(60)
    schedule[k]['end'].append(utils.get_time())
    spec_process.terminate()

    time.sleep(20)

    os.kill(attack_processes['spectrev3'].pid, signal.SIGCONT)
    schedule[k]['start'].append(utils.get_time())
    time.sleep(10)
    os.kill(attack_processes['spectrev3'].pid, signal.SIGSTOP)
    schedule[k]['end'].append(utils.get_time())
    time.sleep(20)
    """

    monitor_process.wait()

    for k, p in attack_processes.items():
        p.terminate()

    # Clean up
    cmd = 'sudo chown zechengh ../ -R'
    print(cmd)
    p = subprocess.Popen(cmd.split())
    p_status = p.wait()

    cmd = 'python2 ../detector/preprocess.py --data_dir {save_data_dir} --window_size 100'.format(
        save_data_dir=save_data_dir
    )
    print(cmd)
    p = subprocess.Popen(cmd.split())
    p_status = p.wait()

    with open(os.path.join(save_data_dir, 'schedule'), 'w+') as f:
        json.dump(dict(schedule), f)
