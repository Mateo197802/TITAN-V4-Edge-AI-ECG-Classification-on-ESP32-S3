import paramiko
import os

HOSTNAME = 'hpc.cedia.edu.ec'
USERNAME = 'kevin.landazuri__yachaytech.edu.ec'
KEY_FILE = os.path.expanduser(r'~/.ssh/cedia_rsa')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(HOSTNAME, username=USERNAME, key_filename=KEY_FILE)
    print("Conectado a CEDIA...")
    # Verificar estado
    command = "cat 'Mateo Gavilanes/TEST_CEDIA/slurm-8802.out' && ls -la 'Mateo Gavilanes/TEST_CEDIA/03_OUTPUTS/'"
    stdin, stdout, stderr = ssh.exec_command(command)
    print("OUT:\n", stdout.read().decode('utf-8'))
    print("ERR:\n", stderr.read().decode('utf-8'))
finally:
    ssh.close()
