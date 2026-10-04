import argparse
import posixpath
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko

from cedia.ssh_config import connect_ssh, required_env

DELTA_DIRS = [
    "DATA/ptb-xl",
    "DATA/segments_wfdb/avb_segments",
    "DATA/virtual_windows",
]

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache"}


def quote_remote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def progress(done: int, total: int, label: str) -> None:
    width = 24
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "." * (width - filled)
    print(f"[{bar}] {done}/{total} {label}", flush=True)


def iter_delta_files(local_v4: Path) -> list[Path]:
    files: list[Path] = []
    for rel_dir in DELTA_DIRS:
        base = local_v4 / Path(rel_dir)
        if not base.exists():
            raise FileNotFoundError(str(base))
        for path in sorted(base.rglob("*")):
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.is_file():
                files.append(path)
    return files


def build_archive(files: list[Path], local_v4: Path) -> Path:
    archive_path = Path(tempfile.gettempdir()) / f"titan_v4_cedia_data_delta_{int(time.time())}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for idx, path in enumerate(files, start=1):
            arcname = path.relative_to(local_v4).as_posix()
            tar.add(path, arcname=arcname)
            if idx == 1 or idx == len(files) or idx % 500 == 0:
                progress(idx, len(files), "archive")
    print(f"Archive: {archive_path} ({archive_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return archive_path


def run_remote(ssh: paramiko.SSHClient, command: str) -> None:
    stdin, stdout, stderr = ssh.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out.rstrip(), flush=True)
    if err:
        print(err.rstrip(), flush=True)
    if code != 0:
        raise RuntimeError(f"Remote command failed with exit code {code}: {command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sincroniza archivos de datos ECG a CEDIA.")
    parser.add_argument(
        "--virtual_only",
        action="store_true",
        help="Sube solo DATA/virtual_windows para evitar reenviar PTB-XL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_v4 = Path(required_env("TITAN_CEDIA_LOCAL_V4")).expanduser().resolve()
    remote_v4 = required_env("TITAN_CEDIA_REMOTE_V4")
    if not local_v4.is_dir():
        raise FileNotFoundError(f"Local CEDIA project directory does not exist: {local_v4}")

    global DELTA_DIRS
    if args.virtual_only:
        DELTA_DIRS = ["DATA/virtual_windows"]
    files = iter_delta_files(local_v4)
    print(f"Archivos del delta: {len(files)}")
    archive_path = build_archive(files, local_v4)

    remote_sync_dir = posixpath.join(remote_v4, "_sync")
    remote_archive = posixpath.join(remote_sync_dir, archive_path.name)

    ssh = connect_ssh()
    try:
        run_remote(ssh, f"mkdir -p {quote_remote(remote_sync_dir)}")
        sftp = ssh.open_sftp()
        try:
            print("Subiendo archive a CEDIA...")
            sftp.put(str(archive_path), remote_archive)
        finally:
            sftp.close()
        print("[########################] upload archive OK", flush=True)
        run_remote(
            ssh,
            f"cd {quote_remote(remote_v4)} && tar -xzf {quote_remote(remote_archive)}",
        )
        print("[########################] extract remote OK", flush=True)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
