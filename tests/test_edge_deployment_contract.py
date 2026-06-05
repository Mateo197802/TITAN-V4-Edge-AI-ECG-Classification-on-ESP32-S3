from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ESP32 = ROOT / "hardware" / "esp32"


def test_esp32_firmware_has_public_wifi_placeholders():
    config = (ESP32 / "firmware/src/config.h").read_text(encoding="utf-8")
    platformio = (ESP32 / "firmware/platformio.ini").read_text(encoding="utf-8")
    assert "YOUR_PRIVATE_2G4_SSID" in config
    assert "YOUR_PRIVATE_WIFI_PASSWORD" in config
    assert "YOUR_PRIVATE_2G4_SSID" in platformio
    assert "YOUR_PRIVATE_WIFI_PASSWORD" in platformio
    assert "Mateo1978" not in config + platformio


def test_esp32_recorder_exports_required_formats():
    recorder = (ESP32 / "recorder/record_wifi_ecg.py").read_text(encoding="utf-8")
    assert ".csv" in recorder
    assert ".mat" in recorder
    assert ".hea" in recorder
    assert "session_summary.json" in recorder

