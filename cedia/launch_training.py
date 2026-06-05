import argparse
import os
import posixpath
import stat
import sys
from pathlib import Path

import paramiko


HOSTNAME = "hpc.cedia.edu.ec"
USERNAME = "kevin.landazuri__yachaytech.edu.ec"
KEY_FILE = os.path.expanduser(r"~/.ssh/cedia_rsa")

PROJECT_ROOT = Path(
    r"c:\Users\intel\OneDrive - yachaytech.edu.ec\Escritorio\UITEY\Semestre VII\Matematica superior\Proyecto final"
)
LOCAL_V4 = PROJECT_ROOT / "V4" / "V4 CEDIA"
REMOTE_V4 = "/home/kevin.landazuri__yachaytech.edu.ec/Mateo Gavilanes/V4_CEDIA"

ROOT_FILES = [
    "lanzar_titan.sh",
    "lanzar_preflight_audit.sh",
    "lanzar_optuna.sh",
    "lanzar_benchmark_workers.sh",
    "setup_cedia_env.sh",
]

EXTRA_FILES = [
    "DATA/ptb-xl/ptbxl_pathology_label_map.json",
    "03_OUTPUTS/label_coverage_report.json",
    "03_OUTPUTS/label_coverage_report.csv",
    "03_OUTPUTS/pathology_coverage_report.json",
    "03_OUTPUTS/pathology_coverage_report.csv",
]

DATA_DIRS = [
    "DATA/georgia",
    "DATA/chapman_shaoxing",
    "DATA/cpsc2018_extra",
    "DATA/ningbo",
    "DATA/ningbo_3avb_pull",
    "DATA/cpsc2018_extra_3avb_pull",
    "DATA/georgia_3avb_pull",
    "DATA/segments_wfdb/3avb_dx_windows",
]

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache"}


def progress(done: int, total: int, label: str) -> None:
    width = 24
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "." * (width - filled)
    print(f"[{bar}] {done}/{total} {label}", flush=True)


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    current = "/" if remote_dir.startswith("/") else "."
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def remote_is_dir(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        return stat.S_ISDIR(sftp.stat(remote_path).st_mode)
    except OSError:
        return False


def iter_code_files() -> list[str]:
    files: list[str] = []
    source_root = LOCAL_V4 / "01_CODIGO_FUENTE"
    for path in sorted(source_root.rglob("*.py")):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        files.append(path.relative_to(LOCAL_V4).as_posix())
    return files


def iter_files_in_dir(rel_dir: str) -> list[str]:
    base = LOCAL_V4 / Path(rel_dir)
    if not base.exists():
        raise FileNotFoundError(str(base))
    files: list[str] = []
    for path in sorted(base.rglob("*")):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.is_file():
            files.append(path.relative_to(LOCAL_V4).as_posix())
    return files


def upload_file(sftp: paramiko.SFTPClient, rel: str) -> None:
    local_path = LOCAL_V4 / Path(rel)
    remote_path = posixpath.join(REMOTE_V4, rel)
    if not local_path.exists():
        raise FileNotFoundError(str(local_path))
    mkdir_p(sftp, posixpath.dirname(remote_path))
    sftp.put(str(local_path), remote_path)


def collect_uploads() -> list[str]:
    files: list[str] = []
    files.extend(ROOT_FILES)
    files.extend(iter_code_files())
    files.extend(EXTRA_FILES)
    for rel_dir in DATA_DIRS:
        files.extend(iter_files_in_dir(rel_dir))
    return sorted(dict.fromkeys(files))


def run_remote(ssh: paramiko.SSHClient, command: str, *, check: bool = True) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out.rstrip(), flush=True)
    if err:
        print(err.rstrip(), file=sys.stderr, flush=True)
    if check and code != 0:
        raise RuntimeError(f"Remote command failed with exit code {code}: {command}")
    return code, out, err


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sincroniza TITAN V4 y lanza entrenamiento controlado en CEDIA."
    )
    parser.add_argument("--no_launch", action="store_true", help="Sincroniza y audita, pero no ejecuta sbatch.")
    parser.add_argument("--skip_sync", action="store_true", help="No sube archivos; solo audita y/o lanza.")
    parser.add_argument("--status", action="store_true", help="Solo consulta squeue y logs recientes.")
    parser.add_argument("--cancel_job", help="Cancela un job SLURM especifico y muestra squeue.")
    parser.add_argument("--audit_only", action="store_true", help="Envia solo lanzar_preflight_audit.sh.")
    args = parser.parse_args()

    uploads = [] if args.skip_sync else collect_uploads()
    print(f"Archivos a sincronizar: {len(uploads)}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, key_filename=KEY_FILE, timeout=30)
    try:
        if args.cancel_job:
            run_remote(ssh, f"scancel {args.cancel_job}", check=False)
            run_remote(ssh, f"squeue -u {USERNAME}", check=False)
            return
        sftp = ssh.open_sftp()
        try:
            if args.status:
                sftp.close()
                run_remote(ssh, f"squeue -u {USERNAME}", check=False)
                run_remote(
                    ssh,
                    f"cd {quote_remote(REMOTE_V4)} && ls -1t titan_audit_log_*.out titan_train_log_*.out 2>/dev/null | head -n 5",
                    check=False,
                )
                run_remote(
                    ssh,
                    f"cd {quote_remote(REMOTE_V4)} && tail -n 120 $(ls -1t titan_audit_log_*.out 2>/dev/null | head -n 1)",
                    check=False,
                )
                run_remote(
                    ssh,
                    f"cd {quote_remote(REMOTE_V4)} && tail -n 160 $(ls -1t titan_train_log_*.out 2>/dev/null | head -n 1)",
                    check=False,
                )
                run_remote(
                    ssh,
                    f"cd {quote_remote(REMOTE_V4)} && tail -n 120 $(ls -1t titan_audit_err_*.err 2>/dev/null | head -n 1)",
                    check=False,
                )
                code = (
                    "import json; "
                    "r=json.load(open('03_OUTPUTS/label_coverage_report.json')); "
                    "print('passed=', r.get('passed')); "
                    "print('missing=', r.get('missing_classes')); "
                    "print('below=', r.get('below_minimum_classes')); "
                    "print('total_train_windows=', sum(x.get('train_windows',0) for x in r.get('class_rows', []))); "
                    "print('total_val_windows=', sum(x.get('val_windows',0) for x in r.get('class_rows', []))); "
                    "print('total_windows=', sum(x.get('total_windows',0) for x in r.get('class_rows', []))); "
                    "[print(x) for x in r.get('class_rows', []) if x.get('class_name') in set(r.get('missing_classes', []) + r.get('below_minimum_classes', []))]"
                )
                run_remote(ssh, f"cd {quote_remote(REMOTE_V4)} && python -c {quote_remote(code)}", check=False)
                return
            if uploads and not remote_is_dir(sftp, REMOTE_V4):
                mkdir_p(sftp, REMOTE_V4)
            for idx, rel in enumerate(uploads, start=1):
                upload_file(sftp, rel)
                if idx == 1 or idx == len(uploads) or idx % 50 == 0:
                    progress(idx, len(uploads), "sync")
        finally:
            sftp.close()

        run_remote(ssh, f"cd {quote_remote(REMOTE_V4)} && chmod +x lanzar_titan.sh lanzar_preflight_audit.sh")

        if args.no_launch:
            print("NO_LAUNCH activo: no se envio sbatch.")
            return

        _, audit_out, _ = run_remote(ssh, f"cd {quote_remote(REMOTE_V4)} && sbatch --parsable lanzar_preflight_audit.sh")
        audit_job_id = audit_out.strip().splitlines()[-1].strip() if audit_out.strip() else "UNKNOWN"
        print(f"[##########..............] auditoria SLURM enviada job_id={audit_job_id}", flush=True)
        if args.audit_only:
            run_remote(ssh, f"squeue -u {USERNAME}", check=False)
            return

        _, train_out, _ = run_remote(
            ssh,
            f"cd {quote_remote(REMOTE_V4)} && sbatch --parsable --dependency=afterok:{audit_job_id} lanzar_titan.sh",
        )
        train_job_id = train_out.strip().splitlines()[-1].strip() if train_out.strip() else "UNKNOWN"
        print(f"[########################] entrenamiento dependiente enviado job_id={train_job_id}", flush=True)
        run_remote(ssh, f"squeue -u {USERNAME}", check=False)
    finally:
        ssh.close()


def quote_remote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
