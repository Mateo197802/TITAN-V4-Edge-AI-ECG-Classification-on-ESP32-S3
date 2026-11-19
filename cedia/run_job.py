from __future__ import annotations

import argparse
import sys

from cedia.ssh_config import connect_ssh


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an explicitly supplied command on the CEDIA cluster.")
    parser.add_argument("--command", required=True, help="Remote shell command to execute.")
    args = parser.parse_args()

    ssh = connect_ssh()
    try:
        stdin, stdout, stderr = ssh.exec_command(args.command)
        del stdin
        output = stdout.read().decode("utf-8", errors="replace")
        errors = stderr.read().decode("utf-8", errors="replace")
        if output:
            print(output, end="")
        if errors:
            print(errors, end="", file=sys.stderr)
        return stdout.channel.recv_exit_status()
    finally:
        ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())
