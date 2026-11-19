import hashlib
import importlib
from pathlib import Path

import pytest


def test_mirror_does_not_accept_a_different_file(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('prepare_expanded_pathology')
    class Response:
        content = b'wrong'
        def raise_for_status(self): pass
    class Session:
        def get(self, url, timeout): return Response()
    target = tmp_path / 'signal.dat'
    with pytest.raises(ValueError, match='hash'):
        module.fetch_s3_file(target, 'https://physionet-open.s3.amazonaws.com/example', hashlib.sha256(b'right').hexdigest(), Session())
    assert not target.exists()


def test_mirror_writes_matching_bytes_and_reuses_verified_cache(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('prepare_expanded_pathology')
    class Response:
        content = b'right'
        def raise_for_status(self): pass
    class Session:
        calls = 0
        def get(self, url, timeout):
            self.calls += 1
            return Response()
    session = Session(); target = tmp_path / 'signal.dat'; h = hashlib.sha256(b'right').hexdigest()
    module.fetch_s3_file(target, 'https://physionet-open.s3.amazonaws.com/example', h, session)
    module.fetch_s3_file(target, 'https://physionet-open.s3.amazonaws.com/example', h, session)
    assert session.calls == 1
    assert target.read_bytes() == b'right'
