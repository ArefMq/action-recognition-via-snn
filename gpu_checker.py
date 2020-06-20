#!/usr/bin/env python3

import subprocess
from time import sleep


def is_gpu_free():
    process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = str(stdout).replace('\\n', '\n')
    return 'no running processes found' in stdout.lower()


def wait_until_gpu_is_free(wait_sec=10, secondary_wait=60):
    if is_gpu_free():
        return

    print('Waiting for GPU to free...')
    while True:
        if is_gpu_free():
            print('GPU got free, process will be run in %d seconds' % secondary_wait)
            sleep(secondary_wait)
            if not is_gpu_free():
                print('GPU is still busy, waiting for it to free...')
                continue
            return
        sleep(wait_sec)


def signal_gpu_state_change(free_signal=None, busy_signal=None, wait_sec=10):
    if free_signal is None and busy_signal is None:
        raise Exception('At least one signal needs to be specified.')

    last_gpu_free = None
    while True:
        gpu_free = is_gpu_free()

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
