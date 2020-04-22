from tools.notify import notify
import subprocess
from time import sleep


while True:
    process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.replace('\\n', '\n')

    if 'no running processes found' in stdout.lower(): 
        notify('GPU Available')
        break
    sleep(10)
