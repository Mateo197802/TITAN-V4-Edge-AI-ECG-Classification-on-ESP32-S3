import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tarfile

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


@pytest.mark.parametrize('stage', ['bootstrap', 'teachers'])
def test_legacy_installation_snapshot_is_never_a_submission_bound_model_result(stage):
    data = b'pass\n'; manifest = json.dumps({'code/worker.py': hashlib.sha256(data).hexdigest()}).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, raw in {'code/worker.py': data, 'INPUT_SHA256.json': manifest}.items():
            info = tarfile.TarInfo(name); info.size = len(raw); archive.addfile(info, io.BytesIO(raw))
    if stage == 'bootstrap':
        receipt = api().snapshot_identity({'stage': stage}, buffer.getvalue())
        assert receipt == {'code_sha256': hashlib.sha256(manifest).hexdigest(), 'submission_bound': False}
    else:
        with pytest.raises(ValueError, match='identity'):
            api().snapshot_identity({'stage': stage}, buffer.getvalue())


def test_collected_legacy_snapshot_keeps_unbound_installation_flag(tmp_path, monkeypatch):
    import contextlib
    from types import SimpleNamespace
    stage = tmp_path / 'stages/bootstrap'; (stage / 'code').mkdir(parents=True)
    source = stage / 'code/worker.py'; source.write_bytes(b'pass\n')
    (stage / 'INPUT_SHA256.json').write_text(json.dumps({'code/worker.py': hashlib.sha256(source.read_bytes()).hexdigest()}))
    for suffix in ('out', 'err'): (stage / ('slurm-1.' + suffix)).write_text('')
    module = api()
    def local_rpc(program):
        buffer = io.BytesIO()
        with contextlib.redirect_stdout(SimpleNamespace(buffer=buffer)): exec(program, {})
        return buffer.getvalue()
    monkeypatch.setattr(module, 'rpc', local_rpc)
    files = module.collect_execution_assets([{'stage': 'bootstrap', 'remote_root': str(tmp_path),
        'stage_root': str(stage), 'job_id': 1}])
    assert json.loads(files['logs/bootstrap/receipt.json'])['snapshot']['submission_bound'] is False
