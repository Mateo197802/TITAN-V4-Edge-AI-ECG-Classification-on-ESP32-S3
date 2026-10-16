import importlib.util
from pathlib import Path

import pytest


def script():
    spec=importlib.util.spec_from_file_location('joint_prepare_test',Path(__file__).resolve().parents[1]/'scripts/prepare_joint_pathology_inputs.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_transient_source_failure_retries_without_weakening_hash_check(monkeypatch):
    module=script();calls=[]
    def operation(*args):
        calls.append(args)
        if len(calls)==1:raise OSError('transient transport error')
    monkeypatch.setattr(module,'_ensure_file',operation)
    module.fetch_pinned_file(Path('fake.hea'),'https://example.org','a'*64,True,sleep=lambda _:None)
    assert len(calls)==2
    assert calls[0]==calls[1]


def test_hash_mismatch_is_never_retried_or_accepted(monkeypatch):
    module=script();calls=[]
    def operation(*args):
        calls.append(args);raise ValueError('hash mismatch')
    monkeypatch.setattr(module,'_ensure_file',operation)
    with pytest.raises(ValueError,match='hash mismatch'):
        module.fetch_pinned_file(Path('fake.hea'),'https://example.org','a'*64,True,sleep=lambda _:None)
    assert len(calls)==1
