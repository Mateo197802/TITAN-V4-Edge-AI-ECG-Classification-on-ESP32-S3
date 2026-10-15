import importlib.util
from pathlib import Path

import pytest


def api():
    path = Path(__file__).resolve().parents[1] / 'scripts/package_founder_transfer.py'
    spec = importlib.util.spec_from_file_location('package_founder_transfer', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_packet_copy_never_overwrites_different_published_bytes(tmp_path):
    source = tmp_path / 'source'; source.mkdir(); (source / 'data.csv').write_bytes(b'original')
    target = tmp_path / 'target'; target.mkdir(); (target / 'data.csv').write_bytes(b'changed')
    with pytest.raises(ValueError, match='different'):
        api().copy_exact(source, target)
    assert (target / 'data.csv').read_bytes() == b'changed'


def test_packet_copy_is_restartable_without_changing_bytes(tmp_path):
    source = tmp_path / 'source'; source.mkdir(); (source / 'data.csv').write_bytes(b'original\r\n')
    target = tmp_path / 'target'
    api().copy_exact(source, target); api().copy_exact(source, target)
    assert (target / 'data.csv').read_bytes() == b'original\r\n'


def test_packet_copy_rejects_unlisted_existing_files(tmp_path):
    source = tmp_path / 'source'; source.mkdir(); (source / 'data.csv').write_bytes(b'original')
    target = tmp_path / 'target'; target.mkdir(); (target / 'unlisted').write_bytes(b'extra')
    with pytest.raises(ValueError, match='different'):
        api().copy_exact(source, target)
