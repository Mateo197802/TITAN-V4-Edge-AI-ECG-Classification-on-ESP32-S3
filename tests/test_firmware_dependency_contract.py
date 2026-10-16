from configparser import ConfigParser
import json
from pathlib import Path


def test_esp32_firmware_pins_its_tensorflow_lite_dependency():
    config = ConfigParser(interpolation=None)
    config.read(Path(__file__).parents[1] / "hardware/esp32/firmware/platformio.ini")

    dependencies = config.get("env", "lib_deps", fallback="")

    assert "Arduino_TensorFlowLite_ESP32.git#e88e0ebee0430ed716ff5b49854795db90066e59" in dependencies


def test_esp32_firmware_does_not_enable_incompatible_tflm_delete_guard():
    config = ConfigParser(interpolation=None)
    config.read(Path(__file__).parents[1] / "hardware/esp32/firmware/platformio.ini")

    build_flags = config.get("common", "build_flags")

    assert "-DTF_LITE_STATIC_MEMORY" not in build_flags
    assert "-fpermissive" not in build_flags


def test_esp32_firmware_target_declares_the_required_octal_psram_board():
    firmware = Path(__file__).parents[1] / "hardware/esp32/firmware"
    config = ConfigParser(interpolation=None)
    config.read(firmware / "platformio.ini")
    board_id = config.get("env", "board")
    board_path = firmware / "boards" / f"{board_id}.json"
    board = json.loads(board_path.read_text(encoding="utf-8"))

    assert "N8R8" in board["name"]
    assert "-DBOARD_HAS_PSRAM" in board["build"]["extra_flags"]
    assert config.get("env", "board_build.arduino.memory_type") == "qio_opi"
