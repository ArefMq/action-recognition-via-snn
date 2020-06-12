#!/usr/bin/env python3

import subprocess
from time import sleep


def wait_until_gpu_is_free(wait_sec=10):
    while True:
        process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout = stdout.replace('\\n', '\n')

        if 'no running processes found' in stdout.lower():
            break
        sleep(wait_sec)


def signal_gpu_state_change(free_signal=None, busy_signal=None, wait_sec=10):
    if free_signal is None and busy_signal is None:
        raise Exception('At least one signal needs to be specified.')

    last_gpu_free = None
    while True:
        process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout = stdout.replace('\\n', '\n')

        if 'no running processes found' in stdout.lower():
            gpu_free = True
        else:
            gpu_free = False

        if last_gpu_free is None:
            last_gpu_free = gpu_free
            continue

        if last_gpu_free != gpu_free:
            if gpu_free:
                free_signal()
            else:
                busy_signal()

        last_gpu_free = gpu_free
        sleep(wait_sec)


if __name__ == "__main__":
    from tools.notify import notify
    wait_until_gpu_is_free()
    notify('GPU Available')
