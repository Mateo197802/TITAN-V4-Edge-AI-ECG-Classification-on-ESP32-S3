import paramiko
import os
import sys

# Forzar salida en UTF-8 para evitar errores con caracteres especiales como ✗
sys.stdout.reconfigure(encoding='utf-8')

HOSTNAME = 'hpc.cedia.edu.ec'
USERNAME = 'kevin.landazuri__yachaytech.edu.ec'
KEY_FILE = os.path.expanduser(r'~/.ssh/cedia_rsa')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOSTNAME, username=USERNAME, key_filename=KEY_FILE)

# Tail the end of the full_8859.out file
_, out, err = ssh.exec_command('tail -n 20 "Mateo Gavilanes/TEST_CEDIA/03_OUTPUTS/full_8859.out"')
stdout = out.read().decode('utf-8', errors='replace').strip()
stderr = err.read().decode('utf-8', errors='replace').strip()

print("STDOUT:")
print(stdout)
if stderr:
    print("STDERR:")
    print(stderr)

ssh.close()
