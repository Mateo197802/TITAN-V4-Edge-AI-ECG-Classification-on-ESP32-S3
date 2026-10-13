import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile


def test_replay_does_not_hash_its_existing_input_manifest(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('joint_manager_test',Path(__file__).resolve().parents[1]/'scripts/manage_joint_cedia_run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    training=tmp_path/'training';training.mkdir()
    (training/'protocol.json').write_text('{}')
    (training/'INPUT_SHA256.json').write_text('{"old":"manifest"}')
    path=tmp_path/'path';path.mkdir()
    for name in ['path_targets.npz','path_extra.npz']:(path/name).write_bytes(b'known-input')
    raw=tmp_path/'raw';raw.mkdir()
    for name in ['train.npz','validation.npz','evaluation.npz']:(raw/name).write_bytes(b'known-raw-input')
    captured=[];manifest_bytes=[]
    def rpc(script,payload):
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            captured.extend(archive.getnames())
            manifest_bytes.append(archive.extractfile('INPUT_SHA256.json').read())
            manifest=json.loads(manifest_bytes[-1])
        assert 'INPUT_SHA256.json' not in manifest
        return json.dumps({'exit_code':0,'stdout':'123\n','remote_root':'safe/run'}).encode()
    monkeypatch.setattr(module,'TRAINING',training);monkeypatch.setattr(module,'rpc',rpc)
    monkeypatch.setattr(sys,'argv',['manager','submit','--state-dir',str(tmp_path/'state'),'--path-inputs',str(path),'--raw-inputs',str(raw)])
    module.main()
    assert captured.count('INPUT_SHA256.json')==1
    assert (tmp_path/'state/INPUT_SHA256.json').read_bytes()==manifest_bytes[0]
