from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('script',[
    'run_joint_inference.py','run_joint_historical.py',
    'replay_joint_validation_auxiliary.py','verify_joint_evidence.py','verify_joint_sources.py',
])
def test_replay_cli_accepts_an_explicit_isolated_bundle(script):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts'/script),'--bundle','isolated-evidence','--help'],
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert '--bundle' in result.stdout
