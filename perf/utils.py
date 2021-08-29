import subprocess
import time

def monitor_cmd(
    core,
    interval_cycles,
    n_readings,
    save_data_dir,
    save_data_name
    ):

    # Collect normal data
    cmd = 'sudo ./event_open_user {core} {interval_cycles} {n_readings} {save_data_dir}{save_data_name}'.format(
        core=core,
        interval_cycles=interval_cycles,
        n_readings=n_readings,
        save_data_dir=save_data_dir,
        save_data_name=save_data_name,
    )
    return cmd

def clean_spec():

    spec_clean_cmd="/home/zechengh/Mastik/ad/bg_program/clean_spec.sh"
    print(spec_clean_cmd)
    spec_clean_process=subprocess.Popen(spec_clean_cmd.split())
    spec_clean_process.wait()
    return

def get_time():
    return int(time.time()*1000000)

def spec_cmd(spec_prog, iterations=1):
    return f"taskset 0x8 runspec --config=test.cfg --size=train" \
    f" --noreportable --tune=base --iterations={iterations} {spec_prog}"

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
