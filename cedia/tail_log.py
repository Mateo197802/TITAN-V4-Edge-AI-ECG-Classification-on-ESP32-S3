from __future__ import annotations

import argparse
import posixpath
from pathlib import PurePosixPath

from cedia.ssh_config import connect_ssh, required_env


def quote_remote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description="Display the end of a log beneath the configured CEDIA project root.")
    parser.add_argument("log_path", help="Remote path relative to TITAN_CEDIA_REMOTE_V4.")
    parser.add_argument("--lines", type=int, default=20)
    args = parser.parse_args()
    relative_path = PurePosixPath(args.log_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        parser.error("log_path must stay inside TITAN_CEDIA_REMOTE_V4")
    if args.lines < 1:
        parser.error("--lines must be a positive integer")

    remote_root = required_env("TITAN_CEDIA_REMOTE_V4")
    remote_path = posixpath.join(remote_root, relative_path.as_posix())
    ssh = connect_ssh()
    try:
        _, stdout, stderr = ssh.exec_command(f"tail -n {args.lines} -- {quote_remote(remote_path)}")
        output = stdout.read().decode("utf-8", errors="replace").strip()
        errors = stderr.read().decode("utf-8", errors="replace").strip()
        print("STDOUT:")
        print(output)
        if errors:
            print("STDERR:")
            print(errors)
        return stdout.channel.recv_exit_status()
    finally:
        ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())
