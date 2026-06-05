from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from scipy.io import loadmat


ESP32_ROOT = Path(__file__).resolve().parents[1]
RECORDER = ESP32_ROOT / "07_record_wifi_ecg.py"


def load_recorder():
    spec = importlib.util.spec_from_file_location("wifi_ecg_recorder", RECORDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {RECORDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WiFiEcgRecorderTests(unittest.TestCase):
    def test_exports_csv_mat_hea_and_metadata(self):
        module = load_recorder()
        raw_csv = (
            "cycle,sample_index,lead_i_adc,lead_ii_adc,lead_iii_adc\n"
            "42,0,100,300,200\n"
            "42,1,110,320,210\n"
        )
        window = module.parse_window_csv(raw_csv)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = module.write_window_artifacts(
                output_dir=Path(tmp),
                window=window,
                quality={"signal_usable": True},
                sample_rate_hz=200,
            )
            for artifact in artifacts.values():
                self.assertTrue(Path(artifact).exists())

            mat = loadmat(artifacts["mat"])
            self.assertEqual(mat["val"].shape, (6, 2))
            self.assertEqual(mat["raw_adc"].shape, (3, 2))

            hea = Path(artifacts["hea"]).read_text(encoding="ascii")
            self.assertIn(" 6 200 2", hea)
            self.assertIn(" aVF", hea)


if __name__ == "__main__":
    unittest.main()
