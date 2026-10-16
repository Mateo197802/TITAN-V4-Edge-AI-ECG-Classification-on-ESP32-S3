from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cedia.ssh_config import connect_ssh, required_env


class FakeSSHClient:
    def __init__(self):
        self.system_host_keys_loaded = False
        self.loaded_host_keys = None
        self.policy = None
        self.connection = None
        self.closed = False

    def load_system_host_keys(self):
        self.system_host_keys_loaded = True

    def load_host_keys(self, path):
        self.loaded_host_keys = path

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, *args, **kwargs):
        self.connection = (args, kwargs)

    def close(self):
        self.closed = True


def test_connect_ssh_loads_trusted_keys_and_rejects_unknown_hosts(tmp_path):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("verified-key\n", encoding="utf-8")
    client = FakeSSHClient()
    reject_policy = object()
    fake_paramiko = SimpleNamespace(SSHClient=lambda: client, RejectPolicy=lambda: reject_policy)
    environment = {
        "TITAN_CEDIA_HOST": "cluster.example.org",
        "TITAN_CEDIA_USERNAME": "test-user",
        "TITAN_CEDIA_KEY_FILE": "~/keys/test-key",
        "TITAN_CEDIA_KNOWN_HOSTS": str(known_hosts),
        "USERPROFILE": str(tmp_path),
        "HOME": str(tmp_path),
    }

    with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, {"paramiko": fake_paramiko}):
        result = connect_ssh()

    assert result is client
    assert client.system_host_keys_loaded
    assert client.loaded_host_keys == str(known_hosts)
    assert client.policy is reject_policy
    assert client.connection == (
        ("cluster.example.org",),
        {"username": "test-user", "key_filename": str(tmp_path / "keys/test-key"), "timeout": 30},
    )


def test_connect_ssh_fails_closed_when_configured_known_hosts_is_missing(tmp_path):
    client = FakeSSHClient()
    fake_paramiko = SimpleNamespace(SSHClient=lambda: client, RejectPolicy=object)
    environment = {
        "TITAN_CEDIA_USERNAME": "test-user",
        "TITAN_CEDIA_KNOWN_HOSTS": str(tmp_path / "missing-known-hosts"),
        "USERPROFILE": str(tmp_path),
        "HOME": str(tmp_path),
    }

    with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, {"paramiko": fake_paramiko}):
        with pytest.raises(FileNotFoundError, match="Known-hosts file does not exist"):
            connect_ssh()


def test_required_environment_variable_has_actionable_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="TITAN_CEDIA_USERNAME"):
            required_env("TITAN_CEDIA_USERNAME")
