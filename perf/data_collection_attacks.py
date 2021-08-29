import os
import subprocess
import time
import argparse
import functools
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--core', type = int, default = 3, help='core index')
    parser.add_argument('--us', type = int, default = 100, help='interval in us')
    parser.add_argument('--n_readings', type = int, default = 300000, help='number of HPC readings')
    parser.add_argument('--bg_program', type = str, default = 'webserver', help='background program')

    parser.add_argument('--dryrun', dest='dryrun', action='store_true', help='Dry run')
    parser.set_defaults(dryrun=False)

    args = parser.parse_args()

    # Adjust to CPU frequency
    interval_cycles = int(args.us / 3)


    attacks = {
        'l1pp': 'taskset 0x8 /home/zechengh/Mastik/exp/test_workspace/spy_l1pp 1000 &',
        'l3pp': 'taskset 0x8 /home/zechengh/Mastik/exp/test_workspace/spy_l3pp 1000 &',
        'fr': 'taskset 0x8 /home/zechengh/Mastik/exp/test_workspace/spy_fr /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &',
        'ff': 'taskset 0x8 /home/zechengh/Mastik/exp/test_workspace/spy_ff /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &',
        'spectrev1': 'taskset 0x8 /home/zechengh/Mastik/ad/attack/spectre-v1/spectrev1 &',
        'spectrev2': 'taskset 0x8 /home/zechengh/Mastik/ad/attack/spectre-v2/spectrev2 &',
        'spectrev3': 'taskset 0x8 /home/zechengh/Mastik/ad/attack/meltdown/memdump &',
        'spectrev4': 'taskset 0x8 /home/zechengh/Mastik/ad/attack/spectre-ssb/spectrev4 &',
        'bufferoverflow': 'taskset 0x8 /home/zechengh/Mastik/ad/bg_program/run_bufferoverflow.sh &',
        }

    gpg_command = 'taskset 0x8 /home/zechengh/Mastik/ad/bg_program/run_gpg.sh'

    spec_benchmarks = ('gcc', 'libquantum') # 'bzip2', 'mcf', 'milc', 'namd', 'gobmk', 'soplex', 'hmmer', 'h264ref'

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

    # Dry run to test the commands
    if args.dryrun:
        dryrun_commands = [
            monitor_cmd_fn(save_data_name='train_normal.csv'),
            gpg_command,
        ] + [
            utils.spec_cmd(spec_benchmarks[0]),
            utils.spec_cmd(spec_benchmarks[1])
        ] + [v for k, v in attacks.items()]

        for dryrun_command in dryrun_commands:
            print(dryrun_command)
            dryrun_process = subprocess.Popen(dryrun_command.split())
            time.sleep(10)
            os.system(f"sudo kill {dryrun_process.pid}")
            #dryrun_process.terminate()
            utils.clean_spec()
            time.sleep(1)
        exit(0)

    time.sleep(20)

    # With SPEC running
    for split in ['train', 'test']:
        for k in attacks.keys():
            attack_process = subprocess.Popen(attacks[k].split())
            # To make the attack actually run
            time.sleep(10)
            cmd = monitor_cmd_fn(save_data_name=f'{split}_abnormal_{k}.csv')
            print(cmd)
            monitor_process = subprocess.Popen(cmd.split())
            monitor_status = monitor_process.wait()

            if split == 'test':

                # Attack with spec running
                for spec_prog in spec_benchmarks:
                    cmd = utils.spec_cmd(spec_prog)
                    print(cmd)
                    spec_process = subprocess.Popen(cmd.split())
                    cmd = monitor_cmd_fn(save_data_name=f'test_abnormal_{k}_with_{spec_prog}.csv')
                    monitor_process = subprocess.Popen(cmd.split())
                    monitor_status = monitor_process.wait()
                    spec_process.terminate()
                    utils.clean_spec()

                # Attack with gpg running
                gpg_process = subprocess.Popen(gpg_command.split())
                time.sleep(10)
                cmd = monitor_cmd_fn(save_data_name=f'test_abnormal_{k}_with_gpg.csv')
                print(cmd)
                monitor_process = subprocess.Popen(cmd.split())
                monitor_status = monitor_process.wait()
                gpg_process.terminate()

            attack_process.terminate()


    # Clean up
    cmd = 'sudo chown zechengh ../ -R'
    print(cmd)
    monitor_process = subprocess.Popen(cmd.split())
    monitor_status = monitor_process.wait()

    cmd = 'python2 ../detector/preprocess.py --data_dir {save_data_dir}'.format(
        save_data_dir=save_data_dir
    )
    print(cmd)
    monitor_process = subprocess.Popen(cmd.split())
    monitor_status = monitor_process.wait()
