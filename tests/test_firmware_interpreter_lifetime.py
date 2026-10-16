from pathlib import Path


def test_static_tflite_interpreter_is_not_deleted_by_the_wrapper():
    source = (
        Path(__file__).parents[1] / "hardware/esp32/firmware/src/ecg_inference.cpp"
    ).read_text(encoding="utf-8")

    assert "_interpreter = &static_interpreter;" in source
    assert "delete _interpreter" not in source
