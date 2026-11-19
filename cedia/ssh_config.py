from __future__ import annotations

import os
from pathlib import Path


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this command.")
    return value


def connect_ssh():
    import paramiko

    host = os.environ.get("TITAN_CEDIA_HOST", "hpc.cedia.edu.ec").strip()
    username = required_env("TITAN_CEDIA_USERNAME")
    key_file = Path(os.environ.get("TITAN_CEDIA_KEY_FILE", "~/.ssh/cedia_rsa")).expanduser()
    known_hosts = os.environ.get("TITAN_CEDIA_KNOWN_HOSTS")
    known_hosts_path = Path(known_hosts).expanduser() if known_hosts else None
    if known_hosts_path is not None and not known_hosts_path.is_file():
        raise FileNotFoundError(f"Known-hosts file does not exist: {known_hosts_path}")

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if known_hosts_path is not None:
        client.load_host_keys(str(known_hosts_path))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(host, username=username, key_filename=str(key_file), timeout=30)
    except Exception:
        client.close()
        raise
    return client
